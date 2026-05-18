import os
import sys
import time
import signal
import requests
import pandas as pd
from risk_controller import DynamicRiskController
from pnl_utils import _get_realized_net_pnl, get_realized_pnl_record
import logging
from datetime import datetime, timezone, timedelta
import numpy as np
from collections import deque
import config
from ai_advisor import AITradingAdvisor
from market_analyzer import MarketAnalyzer
from tool import ToolKit
from review_analytics import (
    build_ai_shadow_recommendations,
    build_backtest_parameter_advice,
    build_live_ai_review_recommendations,
    build_live_ai_review_rows,
    summarize_ai_shadow_quality,
)

try:
    from okx_engine import OKXExecutionEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _configure_stdio_utf8():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            reconfigure(encoding='utf-8', errors='replace')


def _safe_ai_reason(reason):
    text = str(reason or '').strip()
    if not text or text == 'æªç¥':
        return '未知'
    return text


def _extract_funding_bps(data):
    funding_snapshot = data.get('funding_snapshot') or {}
    try:
        funding_rate = float(funding_snapshot.get('funding_rate') or 0.0)
    except (TypeError, ValueError):
        funding_rate = 0.0
    hold_hours = max(float(getattr(config, 'FUNDING_HOLD_HOURS', 8.0) or 0.0), 0.0)
    if hold_hours <= 0:
        return 0.0
    return abs(funding_rate) * 10000.0 * (hold_hours / 8.0)



def _compute_rr_costs_bps(data):
    fee_bps = float(getattr(config, 'FEE_RATE', 0.0004) or 0.0) * 10000.0
    if bool(getattr(config, 'RR_INCLUDE_FEES', True)):
        fee_bps *= 2.0
    else:
        fee_bps = 0.0

    slippage_bps = float(getattr(config, 'SLIPPAGE_RATE', 0.0005) or 0.0) * 10000.0
    if bool(getattr(config, 'RR_INCLUDE_SLIPPAGE', True)):
        slippage_bps *= 2.0
    else:
        slippage_bps = 0.0

    funding_bps = _extract_funding_bps(data) if bool(getattr(config, 'RR_INCLUDE_FUNDING', True)) else 0.0
    total_cost_bps = fee_bps + slippage_bps + funding_bps
    try:
        basis_price = float(data.get('price') or 0.0)
    except (TypeError, ValueError):
        basis_price = 0.0
    estimated_cost_u = basis_price * (total_cost_bps / 10000.0) if basis_price > 0 else 0.0
    return {
        'fee_bps': round(fee_bps, 4),
        'slippage_bps': round(slippage_bps, 4),
        'funding_bps': round(funding_bps, 4),
        'total_cost_bps': round(total_cost_bps, 4),
        'estimated_cost_u': round(estimated_cost_u, 6),
    }



def _classify_market_regime(data):
    try:
        adx = float(data.get('adx_1h') or 0.0)
        prev_adx = float(data.get('prev_adx_1h') or 0.0)
        vol_ratio = float(data.get('vol_ratio_1h') or 0.0)
        trend_state = str(data.get('trend_state') or '').lower()
        ema_4h = str(data.get('ema_status_4h') or '')
        ema_1d = str(data.get('ema_status_1d') or '')
        price = float(data.get('price') or 0.0)
        boll_high = float(data.get('boll_high_1h') or 0.0)
        boll_low = float(data.get('boll_low_1h') or 0.0)
        band_width_ratio = abs(boll_high - boll_low) / max(price, 1e-9) if price > 0 and boll_high > boll_low else 0.0
        funding_bps = _extract_funding_bps(data)

        aligned_long = trend_state == 'long' and '多头' in ema_4h and '多头' in ema_1d
        aligned_short = trend_state == 'short' and '空头' in ema_4h and '空头' in ema_1d
        range_bias = trend_state in {'range', 'sideways'} or ('震荡' in ema_4h and '震荡' in ema_1d)

        if (aligned_long or aligned_short) and adx >= 25 and adx >= prev_adx:
            return 'strong_trend'
        if vol_ratio >= 2.0 or funding_bps >= 20 or band_width_ratio >= 0.08:
            return 'event_volatility'
        if vol_ratio <= 0.7 and band_width_ratio <= 0.012:
            return 'low_liquidity'
        if funding_bps >= 12 or vol_ratio >= 1.8:
            return 'event_driven'
        if range_bias and (band_width_ratio <= 0.025 or vol_ratio <= 1.0):
            return 'high_vol_range'
        if adx < 18 and band_width_ratio < 0.015:
            return 'low_vol_chop'
        if aligned_long or aligned_short:
            return 'weak_trend'
        return 'high_vol_range'
    except Exception:
        return 'high_vol_range'



def _build_strategy_plan(direction, data, conflict_override_active=None):
    direction = str(direction or '').upper()
    market_regime = _classify_market_regime(data)
    trend_state = str(data.get('trend_state') or '').lower()
    m15_entry_mode = str(data.get('m15_entry_mode') or 'normal').lower()
    pullback_ok = bool(data.get('pullback_ok'))

    same_direction = (
        (direction == 'LONG' and trend_state == 'long') or
        (direction == 'SHORT' and trend_state == 'short')
    )

    if conflict_override_active:
        return {
            'market_regime': market_regime,
            'strategy_bucket': 'reversal_probe',
            'rr_label': 'reversal_probe',
            'min_net_rr': 1.5,
            'max_scaler': min(float(conflict_override_active.get('scaler', 0.2) or 0.2), 0.2),
            'allow_conflict_override': True,
        }

    if same_direction and market_regime == 'strong_trend' and m15_entry_mode == 'normal' and not pullback_ok:
        return {
            'market_regime': market_regime,
            'strategy_bucket': 'trend_continuation',
            'rr_label': 'trend_continuation',
            'min_net_rr': 1.0,
            'max_scaler': 1.0,
            'allow_conflict_override': False,
        }

    if same_direction and pullback_ok:
        return {
            'market_regime': market_regime,
            'strategy_bucket': 'pullback_continuation',
            'rr_label': 'pullback_continuation',
            'min_net_rr': 1.1,
            'max_scaler': 0.7,
            'allow_conflict_override': False,
        }

    if market_regime in {'high_vol_range', 'low_vol_chop', 'event_volatility', 'low_liquidity'}:
        if market_regime in {'event_volatility', 'low_liquidity'}:
            min_net_rr = 1.35
            max_scaler = 0.35
        elif market_regime == 'high_vol_range':
            min_net_rr = 1.4
            max_scaler = 0.4
        else:
            min_net_rr = 1.3
            max_scaler = 0.5
        return {
            'market_regime': market_regime,
            'strategy_bucket': 'mean_reversion' if market_regime in {'high_vol_range', 'low_vol_chop'} else 'reversal_probe',
            'rr_label': 'mean_reversion' if market_regime in {'high_vol_range', 'low_vol_chop'} else 'reversal_probe',
            'min_net_rr': min_net_rr,
            'max_scaler': max_scaler,
            'allow_conflict_override': False,
        }

    return {
        'market_regime': market_regime,
        'strategy_bucket': 'trend_continuation' if same_direction else 'reversal_probe',
        'rr_label': 'trend_continuation' if same_direction else 'reversal_probe',
        'min_net_rr': 1.2 if same_direction else 1.5,
        'max_scaler': 0.85 if same_direction else 0.2,
        'allow_conflict_override': bool(conflict_override_active),
    }



def _rank_trade_candidates(candidates):
    bucket_weight = {
        'trend_continuation': 4.0,
        'pullback_continuation': 3.0,
        'reversal_probe': 2.0,
        'mean_reversion': 1.0,
    }

    def sort_key(candidate):
        bucket = str(candidate.get('strategy_bucket') or '')
        net_rr = float(candidate.get('net_rr') or 0.0)
        ai_conf = float(candidate.get('ai_confidence') or 0.0)
        same_side_exposure = float(candidate.get('same_side_exposure_ratio') or 0.0)
        symbol_exposure = float(candidate.get('symbol_exposure_ratio') or 0.0)
        sizing_multiplier = float(candidate.get('sizing_multiplier') or 1.0)
        score = (
            bucket_weight.get(bucket, 0.0) * 100.0
            + net_rr * 10.0
            + ai_conf * 0.1
            + sizing_multiplier * 5.0
            - same_side_exposure * 180.0
            - symbol_exposure * 60.0
        )
        return score

    return sorted(candidates, key=sort_key, reverse=True)



def _get_symbol_trading_params(symbol):
    fallback = {
        'group': 'default',
        'min_net_rr_floor': float(getattr(config, 'MIN_NET_RR_HARD_FLOOR', 0.7) or 0.7),
        'same_side_ai_confidence': 65.0,
        'reverse_ai_confidence': 75.0,
        'position_scaler': 1.0,
        'm15_strong_signal_buffer': float(getattr(config, 'M15_CONFIRM_STRONG_SIGNAL_BUFFER', 1.0) or 1.0),
        'm15_reduced_scaler': float(getattr(config, 'M15_CONFIRM_REDUCED_SCALER', 0.7) or 0.7),
    }
    symbol = str(symbol or '').upper()
    groups = getattr(config, 'SYMBOL_TRADING_GROUPS', {}) or {}
    params_by_group = getattr(config, 'SYMBOL_TRADING_PARAMS', {}) or {}
    matched_params = None
    for group, symbols in groups.items():
        params = dict(fallback)
        params.update(params_by_group.get(group, {}) or {})
        params['group'] = group
        if symbol in {str(item).upper() for item in (symbols or [])}:
            matched_params = params
    return matched_params or fallback


def _extract_algo_side_from_signal(signal_text):
    text = str(signal_text or '')
    if any(k in text for k in ["抄底", "看多", "异动", "刺客", "反弹"]):
        return 'LONG'
    if any(k in text for k in ["逃顶", "看空", "减仓"]):
        return 'SHORT'
    return None


def _sort_market_rows_by_candidates(market_rows, ranked_candidates, symbol_order):
    candidate_order = {item['symbol']: index for index, item in enumerate(ranked_candidates)}
    return sorted(
        market_rows,
        key=lambda row: candidate_order.get(
            row[0],
            len(candidate_order) + symbol_order.index(row[0]) if row[0] in symbol_order else 9999,
        ),
    )



def _select_trade_candidates_for_available_slots(candidates, accounts):
    ranked = _rank_trade_candidates(candidates)
    available_slots = 0
    max_pos = max(1, int(getattr(config, 'MAX_OPEN_POSITIONS', 3) or 3))
    for acc in accounts or []:
        positions = getattr(acc, 'real_positions', None)
        if positions is None:
            positions = getattr(acc, 'open_positions', {})
        open_count = len(positions or {})
        available_slots += max(0, max_pos - open_count)
    if available_slots <= 0:
        return []
    return ranked[:available_slots]



def _allocate_trade_candidates_to_accounts(candidates, accounts):
    ranked = _rank_trade_candidates(candidates)
    assignments = []
    max_pos = max(1, int(getattr(config, 'MAX_OPEN_POSITIONS', 3) or 3))
    max_account_exposure_ratio = float(getattr(config, 'MAX_ACCOUNT_EXPOSURE_RATIO', 0.8) or 0.8)
    max_symbol_exposure_ratio = float(getattr(config, 'MAX_SYMBOL_EXPOSURE_RATIO', 0.35) or 0.35)
    max_direction_exposure_ratio = float(getattr(config, 'MAX_DIRECTION_EXPOSURE_RATIO', 0.65) or 0.65)
    max_cluster_exposure_ratio = float(getattr(config, 'MAX_CLUSTER_EXPOSURE_RATIO', 0.55) or 0.55)

    account_state = {}
    for acc in accounts or []:
        base_capital = max(float(acc.get_base_capital() or 0.0), 0.0)
        trade_meta = getattr(acc, 'real_trade_meta', {}) or {}
        snapshot = _compute_portfolio_exposure_snapshot(trade_meta)
        account_state[id(acc)] = {
            'base_capital': base_capital,
            'positions': getattr(acc, 'real_positions', None) or getattr(acc, 'open_positions', {}) or {},
            'notional': sum(float(meta.get('notional', 0.0) or 0.0) for meta in trade_meta.values()),
            'direction': dict(snapshot.get('direction', {}) or {}),
            'cluster': dict(snapshot.get('cluster', {}) or {}),
            'symbol': dict(snapshot.get('symbol', {}) or {}),
        }

    for candidate in ranked:
        margin_ratio = float(candidate.get('margin_ratio') or 0.0)
        expected_notional_ratio = float(candidate.get('expected_notional_ratio') or (margin_ratio * float(candidate.get('leverage') or 0.0)) or 0.0)
        assigned_account = None
        best_score = None
        symbol = str(candidate.get('symbol') or '')
        candidate_side = str(candidate.get('algo_side') or '').upper()
        candidate_cluster = str(candidate.get('risk_cluster') or _get_symbol_risk_cluster(symbol))
        for acc in accounts or []:
            state = account_state[id(acc)]
            positions = state['positions']
            if len(positions) >= max_pos:
                continue
            if symbol in positions:
                continue
            base_capital = state['base_capital']
            if base_capital <= 0:
                continue
            current_notional = state['notional']
            symbol_current_notional = state['symbol'].get(symbol.upper(), 0.0)
            direction_current_notional = state['direction'].get(candidate_side or 'UNKNOWN', 0.0)
            cluster_current_notional = state['cluster'].get(candidate_cluster, 0.0)
            account_remaining_ratio = max(max_account_exposure_ratio - (current_notional / max(base_capital, 1e-9)), 0.0)
            symbol_remaining_ratio = max(max_symbol_exposure_ratio - (symbol_current_notional / max(base_capital, 1e-9)), 0.0)
            direction_remaining_ratio = max(max_direction_exposure_ratio - (direction_current_notional / max(base_capital, 1e-9)), 0.0)
            cluster_remaining_ratio = max(max_cluster_exposure_ratio - (cluster_current_notional / max(base_capital, 1e-9)), 0.0)
            planned_notional_ratio = min(expected_notional_ratio, account_remaining_ratio, symbol_remaining_ratio, direction_remaining_ratio, cluster_remaining_ratio)
            if planned_notional_ratio <= 0:
                continue
            projected_notional = current_notional + (base_capital * planned_notional_ratio)
            score = base_capital - projected_notional
            if best_score is None or score > best_score:
                best_score = score
                assigned_account = acc
                assigned_ratio = planned_notional_ratio

        if assigned_account is not None:
            state = account_state[id(assigned_account)]
            assignments.append({'symbol': candidate.get('symbol'), 'candidate': candidate, 'account': assigned_account})
            added_notional = state['base_capital'] * float(assigned_ratio)
            state['notional'] += added_notional
            state['symbol'][symbol.upper()] = state['symbol'].get(symbol.upper(), 0.0) + added_notional
            if candidate_side:
                state['direction'][candidate_side] = state['direction'].get(candidate_side, 0.0) + added_notional
            if candidate_cluster:
                state['cluster'][candidate_cluster] = state['cluster'].get(candidate_cluster, 0.0) + added_notional

    return assignments



def _build_runtime_trade_candidate(symbol, data, spec, accounts):
    signal_text = data.get('signal', '')
    algo_side = _extract_algo_side_from_signal(signal_text)
    if not algo_side:
        return None

    current_price = float(data.get('price') or 0.0)
    rr_preview = _compute_realistic_rr_preview(algo_side, data, current_price, lambda direction, payload, price: accounts[0].monitor._rr_obstacle_levels(direction, payload, price)) if accounts else None
    net_rr = float(rr_preview.get('net_rr', 0.0) or 0.0) if rr_preview else 0.0
    strategy_plan = _build_strategy_plan(algo_side, data)

    total_capital = sum(max(float(acc.get_base_capital() or 0.0), 0.0) for acc in accounts) or 1e-9
    exposure_snapshot = {'direction': {}, 'cluster': {}, 'symbol': {}}
    for acc in accounts:
        trade_meta = getattr(acc, 'real_trade_meta', {}) or {}
        snapshot = _compute_portfolio_exposure_snapshot(trade_meta)
        for key in ('direction', 'cluster', 'symbol'):
            for name, notional in (snapshot.get(key, {}) or {}).items():
                exposure_snapshot[key][name] = exposure_snapshot[key].get(name, 0.0) + float(notional or 0.0)

    same_side_notional = exposure_snapshot['direction'].get(algo_side, 0.0)
    symbol_notional = exposure_snapshot['symbol'].get(symbol.upper(), 0.0)
    cluster_name = _get_symbol_risk_cluster(symbol)
    cluster_notional = exposure_snapshot['cluster'].get(cluster_name, 0.0)
    multiplier = _compute_dynamic_position_multiplier(
        accounts[0] if accounts else None,
        symbol,
        data,
        spec,
        net_rr=net_rr,
        ai_confidence=float(data.get('ai_confidence') or 0.0),
    )

    return {
        'symbol': symbol,
        'strategy_bucket': strategy_plan.get('strategy_bucket'),
        'market_regime': strategy_plan.get('market_regime'),
        'net_rr': net_rr,
        'ai_confidence': float(data.get('ai_confidence') or 0.0),
        'same_side_exposure_ratio': same_side_notional / total_capital,
        'direction_exposure_ratio': same_side_notional / total_capital,
        'symbol_exposure_ratio': symbol_notional / total_capital,
        'cluster_exposure_ratio': cluster_notional / total_capital,
        'risk_cluster': cluster_name,
        'algo_side': algo_side,
        'margin_ratio': float((spec or {}).get('margin_ratio', 0.0) or 0.0) * multiplier,
        'expected_notional_ratio': float((spec or {}).get('margin_ratio', 0.0) or 0.0) * float((spec or {}).get('leverage', 0.0) or 0.0) * multiplier,
        'risk_level': (spec or {}).get('risk_level'),
        'leverage': float((spec or {}).get('leverage', 0.0) or 0.0),
        'sizing_multiplier': multiplier,
        'base_margin_ratio': float((spec or {}).get('margin_ratio', 0.0) or 0.0),
    }



def _build_ai_decision_log_record(symbol, algo_signal, final_action, ai_reason, data, signal_text, ai_confidence=0, account_name='', order_id=''):
    rr_costs = data.get('rr_costs') or {}
    gross_rr = data.get('realistic_rr', data.get('gross_rr'))
    net_rr = data.get('net_expected_rr', data.get('net_rr'))
    strategy_plan = data.get('strategy_plan') or {}
    return {
        '触发时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '平仓时间': '',
        '账户': account_name or '',
        '订单ID': order_id or '',
        '交易对': symbol,
        '策略触发方向': algo_signal,
        'AI最终判定': final_action,
        'AI置信度': float(ai_confidence or 0),
        'AI决策原因': ai_reason,
        '策略数据': ToolKit.get_strategy_data_str(data, signal_text),
        '原始数据': getattr(ToolKit, 'get_clean_raw_data', lambda x: str(x))(data),
        '市场状态': strategy_plan.get('market_regime', data.get('market_regime', '')),
        '策略桶': strategy_plan.get('strategy_bucket', data.get('strategy_bucket', '')),
        '粗RR': gross_rr,
        '净RR': net_rr,
        '预估手续费bps': rr_costs.get('fee_bps'),
        '预估滑点bps': rr_costs.get('slippage_bps'),
        '预估资金费率bps': rr_costs.get('funding_bps'),
        '预估总成本bps': rr_costs.get('total_cost_bps'),
        '预估成本(U)': rr_costs.get('estimated_cost_u'),
        'BTC近端粗RR': data.get('btc_near_gross_rr'),
        'BTC近端净RR': data.get('btc_near_net_rr'),
        'BTC近端目标': data.get('btc_near_obstacle'),
        'BTC扩展粗RR': data.get('btc_extended_gross_rr'),
        'BTC扩展净RR': data.get('btc_extended_net_rr'),
        'BTC扩展目标': data.get('btc_extended_obstacle'),
        'BTC RR模式': data.get('btc_rr_mode'),
        'BTC RR仓位上限': data.get('btc_rr_cap'),
        'AI仓位档位': data.get('ai_entry_label', ''),
        'AI杠杆系数': data.get('ai_leverage_factor', 1.0),
        '后续4H涨跌(%)': data.get('后续4H涨跌(%)'),
        '拦截评估结果': data.get('拦截评估结果', ''),
        'Shadow结果': data.get('Shadow结果', ''),
        'Shadow机会收益(%)': data.get('Shadow机会收益(%)'),
        'Shadow机会方向': data.get('Shadow机会方向', ''),
        '平仓价格': data.get('平仓价格'),
        '盈亏': data.get('盈亏'),
        '余额': data.get('余额'),
    }
def _prepare_strategy_summary_frame(df):
    if df is None or df.empty:
        return None
    work_df = df.copy()
    work_df['净RR'] = pd.to_numeric(work_df['净RR'], errors='coerce') if '净RR' in work_df.columns else np.nan
    work_df['盈亏'] = pd.to_numeric(work_df['盈亏'], errors='coerce') if '盈亏' in work_df.columns else np.nan
    if 'AI最终判定' in work_df.columns:
        trade_mask = work_df['AI最终判定'].isin(['LONG', 'SHORT'])
    else:
        trade_mask = pd.Series([False] * len(work_df), index=work_df.index)
    work_df['_trade_mask'] = trade_mask
    work_df['_win_mask'] = trade_mask & (work_df['盈亏'].fillna(0) > 0)
    return work_df


def _append_ai_decision_log_row(df, expected_columns, dec_rec):
    new_row_data = {col: dec_rec.get(col, np.nan) for col in expected_columns}
    new_row = pd.DataFrame([new_row_data], columns=expected_columns)
    if df is None or df.empty:
        return new_row
    base = df.reindex(columns=expected_columns).copy()
    base.loc[len(base)] = new_row_data
    return base.reset_index(drop=True)



def _build_summary_rows(work_df, group_columns, label_keys):
    if work_df is None:
        return []
    rows = []
    for group_values, group in work_df.groupby(group_columns, dropna=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        labels = [str(value or '').strip() for value in group_values]
        if any(not label for label in labels):
            continue
        trade_count = int(group['_trade_mask'].sum())
        rows.append({
            **dict(zip(label_keys, labels)),
            'signal_count': int(len(group)),
            'trade_count': trade_count,
            'win_rate': round(float(group['_win_mask'].sum()) / trade_count * 100, 2) if trade_count else 0.0,
            'avg_net_rr': round(float(group['净RR'].dropna().mean()) if not group['净RR'].dropna().empty else 0.0, 2),
            'realized_pnl': round(float(group['盈亏'].fillna(0).sum()), 2),
        })
    return rows



def _summarize_symbols(df):
    if df is None or df.empty or '交易对' not in df.columns:
        return []
    work_df = _prepare_strategy_summary_frame(df)
    rows = _build_summary_rows(work_df, ['交易对'], ['symbol'])
    return sorted(rows, key=lambda row: (-row['signal_count'], row['symbol']))



def _summarize_strategy_buckets(df):
    if df is None or df.empty or '策略桶' not in df.columns:
        return []
    work_df = _prepare_strategy_summary_frame(df)
    rows = _build_summary_rows(work_df, ['策略桶'], ['strategy_bucket'])
    return sorted(rows, key=lambda row: (-row['signal_count'], row['strategy_bucket']))



def _summarize_market_regimes(df):
    if df is None or df.empty or '市场状态' not in df.columns:
        return []
    work_df = _prepare_strategy_summary_frame(df)
    rows = _build_summary_rows(work_df, ['市场状态'], ['market_regime'])
    return sorted(rows, key=lambda row: (-row['signal_count'], row['market_regime']))



def _summarize_regime_bucket_matrix(df):
    if df is None or df.empty or '市场状态' not in df.columns or '策略桶' not in df.columns:
        return []
    work_df = _prepare_strategy_summary_frame(df)
    rows = _build_summary_rows(work_df, ['市场状态', '策略桶'], ['market_regime', 'strategy_bucket'])
    return sorted(rows, key=lambda row: (-row['signal_count'], row['market_regime'], row['strategy_bucket']))



def _summarize_shadow_opportunities(df):
    if df is None or df.empty or 'Shadow结果' not in df.columns:
        return []
    work_df = df.copy()
    work_df['Shadow机会收益(%)'] = pd.to_numeric(work_df['Shadow机会收益(%)'], errors='coerce') if 'Shadow机会收益(%)' in work_df.columns else np.nan
    work_df = work_df[work_df['Shadow结果'].notna() & (work_df['Shadow结果'].astype(str).str.strip() != '')]
    if work_df.empty:
        return []
    rows = []
    total = int(len(work_df))
    missed = int((work_df['Shadow结果'] == '错过机会').sum())
    avoided = int((work_df['Shadow结果'] == '避免亏损').sum())
    rows.append({
        'scope': 'overall',
        'sample_count': total,
        'missed_count': missed,
        'avoided_count': avoided,
        'missed_rate': round(missed / total * 100, 2) if total else 0.0,
        'avg_opportunity_return_pct': round(float(work_df['Shadow机会收益(%)'].dropna().mean()) if not work_df['Shadow机会收益(%)'].dropna().empty else 0.0, 2),
    })
    return rows



def _summarize_shadow_opportunities_by_bucket(df):
    if df is None or df.empty or 'Shadow结果' not in df.columns or '策略桶' not in df.columns:
        return []
    work_df = df.copy()
    work_df['Shadow机会收益(%)'] = pd.to_numeric(work_df['Shadow机会收益(%)'], errors='coerce') if 'Shadow机会收益(%)' in work_df.columns else np.nan
    work_df = work_df[work_df['Shadow结果'].notna() & (work_df['Shadow结果'].astype(str).str.strip() != '')]
    if work_df.empty:
        return []
    rows = []
    for bucket, group in work_df.groupby('策略桶', dropna=True):
        bucket_name = str(bucket or '').strip()
        if not bucket_name:
            continue
        total = int(len(group))
        missed = int((group['Shadow结果'] == '错过机会').sum())
        avoided = int((group['Shadow结果'] == '避免亏损').sum())
        rows.append({
            'strategy_bucket': bucket_name,
            'sample_count': total,
            'missed_count': missed,
            'avoided_count': avoided,
            'missed_rate': round(missed / total * 100, 2) if total else 0.0,
            'avg_opportunity_return_pct': round(float(group['Shadow机会收益(%)'].dropna().mean()) if not group['Shadow机会收益(%)'].dropna().empty else 0.0, 2),
        })
    return sorted(rows, key=lambda row: (-row['sample_count'], row['strategy_bucket']))



def _evaluate_shadow_outcome(algo_side, change_pct):
    if pd.isna(change_pct):
        return '', None
    opportunity_return = float(change_pct) if algo_side == 'LONG' else -float(change_pct)
    return ('错过机会' if opportunity_return > 0 else '避免亏损'), round(opportunity_return, 2)



def _prepare_execution_quality_frame(df):
    if df is None or df.empty:
        return None
    work_df = df.copy()
    work_df['预估成本(U)'] = pd.to_numeric(work_df['预估成本(U)'], errors='coerce') if '预估成本(U)' in work_df.columns else np.nan
    work_df['实际成本(U)'] = pd.to_numeric(work_df['实际成本(U)'], errors='coerce') if '实际成本(U)' in work_df.columns else np.nan
    work_df['成本校准偏差(U)'] = pd.to_numeric(work_df['成本校准偏差(U)'], errors='coerce') if '成本校准偏差(U)' in work_df.columns else np.nan
    work_df['开平成本偏差(U)'] = pd.to_numeric(work_df['开平成本偏差(U)'], errors='coerce') if '开平成本偏差(U)' in work_df.columns else np.nan
    has_execution = work_df['预估成本(U)'].notna() | work_df['实际成本(U)'].notna()
    work_df = work_df[has_execution].copy()
    if work_df.empty:
        return None
    return work_df



def _build_open_trade_execution_quality_frame(accounts):
    rows = []
    for acc in accounts or []:
        for symbol, meta in (getattr(acc, 'real_trade_meta', {}) or {}).items():
            estimated_cost = meta.get('estimated_cost_u')
            if estimated_cost is None:
                estimated_cost = (meta.get('rr_costs') or {}).get('estimated_cost_u')
            if estimated_cost is None:
                continue
            rows.append({
                '交易对': symbol,
                '市场状态': meta.get('market_regime', ''),
                '策略桶': meta.get('strategy_bucket', ''),
                '预估成本(U)': estimated_cost,
                '实际成本(U)': np.nan,
                '成本校准偏差(U)': np.nan,
                '开平成本偏差(U)': np.nan,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)



def _summarize_execution_quality(df):
    if df is None or df.empty:
        return []
    work_df = _prepare_execution_quality_frame(df)
    if work_df is None or work_df.empty:
        return []
    rows = []
    trade_count = int(len(work_df))
    estimated_avg = float(work_df['预估成本(U)'].dropna().mean()) if not work_df['预估成本(U)'].dropna().empty else 0.0
    actual_avg = float(work_df['实际成本(U)'].dropna().mean()) if not work_df['实际成本(U)'].dropna().empty else 0.0
    delta_avg = float(work_df['成本校准偏差(U)'].dropna().mean()) if not work_df['成本校准偏差(U)'].dropna().empty else 0.0
    slippage_avg = float(work_df['开平成本偏差(U)'].dropna().mean()) if not work_df['开平成本偏差(U)'].dropna().empty else 0.0
    rows.append({
        'scope': 'overall',
        'trade_count': trade_count,
        'avg_estimated_cost_u': round(estimated_avg, 4),
        'avg_actual_cost_u': round(actual_avg, 4),
        'avg_cost_delta_u': round(delta_avg, 4),
        'avg_open_close_cost_delta_u': round(slippage_avg, 4),
    })
    return rows



def _summarize_execution_quality_by_bucket(df):
    if df is None or df.empty or '策略桶' not in df.columns:
        return []
    work_df = _prepare_execution_quality_frame(df)
    if work_df is None or work_df.empty:
        return []
    rows = []
    for bucket, group in work_df.groupby('策略桶', dropna=True):
        bucket_name = str(bucket or '').strip()
        if not bucket_name:
            continue
        rows.append({
            'strategy_bucket': bucket_name,
            'trade_count': int(len(group)),
            'avg_estimated_cost_u': round(float(group['预估成本(U)'].dropna().mean()) if not group['预估成本(U)'].dropna().empty else 0.0, 4),
            'avg_actual_cost_u': round(float(group['实际成本(U)'].dropna().mean()) if not group['实际成本(U)'].dropna().empty else 0.0, 4),
            'avg_cost_delta_u': round(float(group['成本校准偏差(U)'].dropna().mean()) if not group['成本校准偏差(U)'].dropna().empty else 0.0, 4),
        })
    return sorted(rows, key=lambda row: (-row['trade_count'], row['strategy_bucket']))



def _summarize_execution_quality_by_regime(df):
    if df is None or df.empty or '市场状态' not in df.columns:
        return []
    work_df = _prepare_execution_quality_frame(df)
    if work_df is None or work_df.empty:
        return []
    rows = []
    for regime, group in work_df.groupby('市场状态', dropna=True):
        regime_name = str(regime or '').strip()
        if not regime_name:
            continue
        rows.append({
            'market_regime': regime_name,
            'trade_count': int(len(group)),
            'avg_estimated_cost_u': round(float(group['预估成本(U)'].dropna().mean()) if not group['预估成本(U)'].dropna().empty else 0.0, 4),
            'avg_actual_cost_u': round(float(group['实际成本(U)'].dropna().mean()) if not group['实际成本(U)'].dropna().empty else 0.0, 4),
            'avg_cost_delta_u': round(float(group['成本校准偏差(U)'].dropna().mean()) if not group['成本校准偏差(U)'].dropna().empty else 0.0, 4),
        })
    return sorted(rows, key=lambda row: (-row['trade_count'], row['market_regime']))



def _compute_strategy_penalties(summary_payload):
    penalties = {'bucket': {}, 'regime_bucket': {}}
    for row in summary_payload.get('bucket_summary', []) or []:
        bucket = str(row.get('strategy_bucket') or '')
        trades = int(row.get('trade_count') or 0)
        win_rate = float(row.get('win_rate') or 0.0)
        pnl = float(row.get('realized_pnl') or 0.0)
        if bucket and trades >= 2 and (pnl < 0 or win_rate < 35):
            penalties['bucket'][bucket] = 0.6
    for row in summary_payload.get('regime_bucket_matrix', []) or []:
        regime = str(row.get('market_regime') or '')
        bucket = str(row.get('strategy_bucket') or '')
        trades = int(row.get('trade_count') or 0)
        win_rate = float(row.get('win_rate') or 0.0)
        pnl = float(row.get('realized_pnl') or 0.0)
        if regime and bucket and trades >= 2 and (pnl < 0 or win_rate < 35):
            penalties['regime_bucket'][f'{regime}|{bucket}'] = 0.5
    return penalties



def _normalize_open_rejection_stage(reason):
    text = str(reason or '')
    if 'AI' in text:
        return 'AI'
    if '净RR' in text or 'RR' in text:
        return '净RR'
    if '15M' in text or '1H收盘' in text or '收盘确认' in text:
        return '确认层'
    if '敞口' in text:
        return '敞口'
    if '规模过小' in text:
        return '最小规模'
    if '冷却' in text or '重复信号' in text:
        return '冷却'
    if '否定词' in text:
        return '信号过滤'
    if 'conflict' in text:
        return '冲突覆盖'
    return '其他'



def _summarize_open_rejections(entries, since_hours=24):
    if not entries:
        return []
    cutoff = datetime.now() - pd.Timedelta(hours=since_hours)
    counts = {}
    for item in entries:
        ts = item.get('time')
        if ts is None or ts < cutoff:
            continue
        reason = str(item.get('reason') or '').strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    rows = [{'reason': reason, 'count': count} for reason, count in counts.items()]
    return sorted(rows, key=lambda row: (-row['count'], row['reason']))



def _summarize_open_rejection_stages(entries, since_hours=24):
    if not entries:
        return []
    cutoff = datetime.now() - pd.Timedelta(hours=since_hours)
    counts = {}
    for item in entries:
        ts = item.get('time')
        if ts is None or ts < cutoff:
            continue
        stage = str(item.get('stage') or _normalize_open_rejection_stage(item.get('reason')))
        if not stage:
            continue
        counts[stage] = counts.get(stage, 0) + 1
    rows = [{'stage': stage, 'count': count} for stage, count in counts.items()]
    return sorted(rows, key=lambda row: (-row['count'], row['stage']))



def _get_symbol_risk_cluster(symbol):
    symbol = str(symbol or '').upper()
    clusters = getattr(config, 'PORTFOLIO_RISK_CLUSTERS', {}) or {}
    for cluster, symbols in clusters.items():
        if symbol in {str(item).upper() for item in (symbols or [])}:
            return cluster
    return 'other'



def _compute_portfolio_exposure_snapshot(trade_meta):
    snapshot = {'direction': {}, 'cluster': {}, 'symbol': {}}
    for symbol, meta in (trade_meta or {}).items():
        notional = float(meta.get('notional', 0.0) or 0.0)
        if notional <= 0:
            continue
        side = str(meta.get('side') or '').upper() or 'UNKNOWN'
        cluster = str(meta.get('risk_cluster') or _get_symbol_risk_cluster(symbol))
        snapshot['direction'][side] = snapshot['direction'].get(side, 0.0) + notional
        snapshot['cluster'][cluster] = snapshot['cluster'].get(cluster, 0.0) + notional
        snapshot['symbol'][str(symbol).upper()] = snapshot['symbol'].get(str(symbol).upper(), 0.0) + notional
    return snapshot



def _compute_dynamic_position_multiplier(account, symbol, data, spec, net_rr=0.0, ai_confidence=0.0):
    if account is None:
        return 1.0
    base = 1.0
    vol_ratio = float(data.get('vol_ratio_1h') or 0.0)
    regime = str(data.get('market_regime') or _classify_market_regime(data))
    recent_closed_pnls = list(getattr(account, 'recent_closed_pnls', []) or [])
    recent_losses = len([p for p in recent_closed_pnls if float(p or 0.0) < 0])
    loss_streak = 0
    for pnl in reversed(recent_closed_pnls):
        if float(pnl or 0.0) < 0:
            loss_streak += 1
        else:
            break
    current_balance = max(float(account.get_base_capital() or 0.0), 1e-9)
    if getattr(account, 'mode', '') in ['TESTNET', 'REAL']:
        if getattr(account, 'real_balance', 0.0) > 0:
            peak = max(getattr(account, 'real_balance', current_balance), current_balance)
            drawdown = max((peak - current_balance) / max(peak, 1e-9), 0.0)
        else:
            drawdown = 0.0
    else:
        virtual_balance = float(getattr(account, 'virtual_balance', current_balance) or current_balance)
        peak = max(virtual_balance, current_balance)
        drawdown = max((peak - current_balance) / max(peak, 1e-9), 0.0)

    if drawdown >= 0.15:
        base *= 0.5
    elif drawdown >= 0.08:
        base *= 0.75

    if loss_streak >= 3 or recent_losses >= 5:
        base *= 0.6
    elif loss_streak >= 2:
        base *= 0.8

    if vol_ratio >= 1.8 or regime == 'event_volatility':
        base *= 0.7
    elif regime == 'high_vol_range':
        base *= 0.6
    elif vol_ratio <= 0.8 and regime == 'strong_trend':
        base *= 1.1

    if regime == 'weak_trend':
        base *= 0.9
    if regime in {'low_vol_chop', 'low_liquidity'}:
        base *= 0.75

    if getattr(config, 'ENABLE_QUANT_SCORING', False):
        base *= max(float(data.get('suggested_scaler') or 1.0), 0.0)

    ai_entry_scaler = max(float(data.get('ai_entry_scaler') or 1.0), 0.0)
    base *= ai_entry_scaler
    if bool(data.get('ai_caution_mode', False)) and ai_entry_scaler >= 1.0:
        base *= 0.5

    base *= max(float(data.get('m15_entry_multiplier') or 1.0), 0.0)

    if float(net_rr or 0.0) >= 1.8 and float(ai_confidence or 0.0) >= 75:
        base *= 1.1
    elif float(net_rr or 0.0) < 1.0:
        base *= 0.85

    symbol_params = _get_symbol_trading_params(symbol)
    base *= float(symbol_params.get('position_scaler', 1.0) or 1.0)
    return max(0.2, min(base, 1.35))





def _build_calibration_recommendations(summary_payload):
    recommendations = []
    rejection_summary = summary_payload.get('open_rejection_summary', []) or []
    bucket_summary = summary_payload.get('bucket_summary', []) or []
    regime_summary = summary_payload.get('regime_summary', []) or []
    symbol_summary = summary_payload.get('symbol_summary', []) or []
    stage_summary = summary_payload.get('open_rejection_stage_summary', []) or []
    shadow_summary = summary_payload.get('shadow_opportunity_summary', []) or []
    shadow_bucket_summary = summary_payload.get('shadow_opportunity_bucket_summary', []) or []
    execution_summary = summary_payload.get('execution_quality_summary', []) or []
    execution_bucket_summary = summary_payload.get('execution_quality_bucket_summary', []) or []
    execution_regime_summary = summary_payload.get('execution_quality_regime_summary', []) or []

    if rejection_summary:
        top_reason = rejection_summary[0]
        if '净RR' in str(top_reason.get('reason', '')):
            recommendations.append('净RR 拦截最多，建议结合 Shadow 错过率判断是否小幅下调 min_net_rr_floor。')
        elif 'AI' in str(top_reason.get('reason', '')):
            recommendations.append('AI 拦截最多，建议按 Shadow 结果复查同向/反向置信阈值。')
        elif '15M' in str(top_reason.get('reason', '')) or '收盘' in str(top_reason.get('reason', '')):
            recommendations.append('确认层拦截偏多，建议按策略桶放宽 15M/1H 确认条件。')

    if stage_summary:
        top_stage = stage_summary[0]
        if top_stage.get('stage') == '敞口':
            recommendations.append('敞口层拦截偏多，建议复查 cluster cap、单币 cap 和当前 notional 统计。')
        elif top_stage.get('stage') == '净RR':
            recommendations.append('净RR 阶段拦截偏多，优先检查 obstacle 距离和成本参数是否过度保守。')

    for row in shadow_summary:
        samples = int(row.get('sample_count') or 0)
        missed_rate = float(row.get('missed_rate') or 0.0)
        avg_return = float(row.get('avg_opportunity_return_pct') or 0.0)
        if samples >= 5 and missed_rate >= 55 and avg_return > 0:
            recommendations.append('Shadow 显示错过机会偏多，建议优先小幅降低 AI 置信阈值或净RR门槛。')
            break
        if samples >= 5 and missed_rate <= 25 and avg_return <= 0:
            recommendations.append('Shadow 显示拦截多为避免亏损，当前 AI/RR 拦截层不宜继续放宽。')
            break

    for row in shadow_bucket_summary:
        samples = int(row.get('sample_count') or 0)
        missed_rate = float(row.get('missed_rate') or 0.0)
        avg_return = float(row.get('avg_opportunity_return_pct') or 0.0)
        if samples >= 3 and missed_rate >= 60 and avg_return > 0:
            recommendations.append(f"策略桶 {row.get('strategy_bucket')} Shadow 错过率偏高，建议小幅放宽该桶 AI/RR 门槛。")
            break

    for row in bucket_summary:
        if row.get('trade_count', 0) >= 3 and float(row.get('win_rate') or 0.0) < 35:
            recommendations.append(f"策略桶 {row.get('strategy_bucket')} 胜率偏低，建议下调 max_scaler 或提高 min_net_rr。")
            break

    for row in regime_summary:
        if row.get('signal_count', 0) >= 3 and float(row.get('win_rate') or 0.0) < 35:
            recommendations.append(f"市场状态 {row.get('market_regime')} 表现偏弱，建议降低该 regime 的仓位系数。")
            break

    for row in symbol_summary:
        trades = int(row.get('trade_count') or 0)
        win_rate = float(row.get('win_rate') or 0.0)
        pnl = float(row.get('realized_pnl') or 0.0)
        if trades >= 3 and (win_rate < 35 or pnl < 0):
            recommendations.append(f"币种 {row.get('symbol')} 近期表现偏弱，建议降低 position_scaler 或提高该币净RR门槛。")
            break

    for row in execution_summary:
        trades = int(row.get('trade_count') or 0)
        delta = float(row.get('avg_cost_delta_u') or 0.0)
        if trades >= 3 and delta > 0:
            recommendations.append('整体实际成本高于预估，建议提高 slippage/funding 成本假设或减少市价成交。')
            break

    for row in execution_bucket_summary:
        trades = int(row.get('trade_count') or 0)
        delta = float(row.get('avg_cost_delta_u') or 0.0)
        if trades >= 2 and delta > 0:
            recommendations.append(f"策略桶 {row.get('strategy_bucket')} 执行成本偏高，建议降低该桶仓位或改用更保守入场。")
            break

    for row in execution_regime_summary:
        trades = int(row.get('trade_count') or 0)
        delta = float(row.get('avg_cost_delta_u') or 0.0)
        if trades >= 2 and delta > 0:
            recommendations.append(f"市场状态 {row.get('market_regime')} 执行成本偏高，建议降低该 regime 的下单 aggressiveness。")
            break

    return recommendations[:8]




def _compute_realistic_rr_preview(direction, data, current_price, obstacle_resolver):
    try:
        direction = str(direction).upper()
        current_price = float(current_price or 0.0)
        defense_low = float(data.get('defense_low') or current_price)
        defense_high = float(data.get('defense_high') or current_price)
        sl_price = defense_low * 0.995 if direction == 'LONG' else defense_high * 1.005
        risk_space = abs(current_price - sl_price)
        if risk_space <= 0:
            return None

        closest_obstacle = obstacle_resolver(direction, data, current_price)
        if closest_obstacle is None:
            return None

        realistic_reward = abs(closest_obstacle - current_price)
        gross_rr = realistic_reward / risk_space if risk_space > 0 else 0.0
        costs = _compute_rr_costs_bps(data)
        cost_ratio = float(costs.get('total_cost_bps', 0.0) or 0.0) / 10000.0
        net_reward = realistic_reward - (current_price * cost_ratio)
        net_rr = max(net_reward / risk_space, 0.0) if risk_space > 0 else 0.0
        return {
            'rr': gross_rr,
            'gross_rr': gross_rr,
            'net_rr': net_rr,
            'fee_bps': costs['fee_bps'],
            'slippage_bps': costs['slippage_bps'],
            'funding_bps': costs['funding_bps'],
            'total_cost_bps': costs['total_cost_bps'],
            'closest_obstacle': closest_obstacle,
            'sl_price': sl_price,
            'risk_space': risk_space,
            'realistic_reward': realistic_reward,
        }
    except Exception:
        return None


# ==========================================
# ==========================================
def _is_robot_trade_history_row(row):
    source = row.get('来源', '')
    if pd.notna(source) and str(source).strip():
        return str(source).strip() == '机器人'
    order_id = str(row.get('订单ID', '') or '').strip().upper()
    return order_id.startswith(('REAL_', 'SIM_', 'RISK_'))


class SubAccount:
    """MAM 子账户执行单元"""
    def __init__(self, name, api_key, api_secret, api_passphrase, monitor, prompt_mode="aggressive"):
        self.name = name
        self.monitor = monitor 
        self.mode = monitor.mode
        self.funds_mode = monitor.funds_mode
        self.enable_ai = monitor.enable_ai
        self.enable_ai_close = monitor.enable_ai_close
        
        self.prompt_mode = prompt_mode
        self.api_passphrase = api_passphrase
        
        self.analyzer = monitor.analyzer
        self.risk_controller = DynamicRiskController(self)
        
        self.total_trades = 0                
        self.win_trades = 0 
        self.recent_closed_pnls = deque(maxlen=getattr(config, 'LOSS_TRACK_WINDOW', 10))
        self.cooldown_until = 0.0
        self.bucket_cooldowns = {}
        self.bucket_loss_streaks = {}
        self.symbol_reentry_cooldowns = {}

        if self.mode in ['TESTNET', 'REAL']:
            if not HAS_ENGINE: raise ImportError("找不到 okx_engine.py，无法开启实盘模式。")
            self.engine = OKXExecutionEngine(api_key, api_secret, api_passphrase, mode=self.mode, proxies=config.OKX_PROXIES)
            self.real_positions = {}
            self.real_balance = 0.0
            
            self.real_trade_meta = ToolKit.load_active_trades(account_name=self.name)
            if self.real_trade_meta:
                logger.info(f"[{self.name}] 检测到未完成的实盘交易，已挂载 {len(self.real_trade_meta)} 个币种。")
        else:
            self.engine = None
            self.virtual_balance = config.INITIAL_BALANCE
            self.open_positions = {}

        self._load_recent_trade_stats()

    def _execute_position_ai_action(self, symbol, trade_id, decision, state, position_data):
        confirmed_action = str(state.get('confirmed_action', '') or '')
        advisory_action = confirmed_action.replace('确认', '', 1) if confirmed_action.startswith('确认') else confirmed_action
        if advisory_action in {'推保本线', '减仓防守', '全部退出'}:
            return f'仅建议：{advisory_action}'
        return '仅建议：无需执行'

    def get_base_capital(self):
        if self.funds_mode == 'ISOLATED': return config.INITIAL_BALANCE
        return self.real_balance if self.mode in ['TESTNET', 'REAL'] else self.virtual_balance

    def _load_recent_trade_stats(self):
        filename = f"trades_history_{self.name}.csv"
        if not os.path.exists(filename):
            return
        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            pnl_col = '净盈亏(U)'
            if pnl_col not in df.columns:
                return
            for value in df[pnl_col].tail(getattr(config, 'LOSS_TRACK_WINDOW', 10)):
                try:
                    self.recent_closed_pnls.append(float(value))
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"[{self.name}] 加载历史亏损统计失败: {e}")

    def _record_closed_trade(self, net_pnl, strategy_bucket=''):
        try:
            pnl_value = float(net_pnl)
        except Exception:
            return

        if bool(getattr(config, 'ENABLE_LOSS_COOLDOWN', True)):
            self.recent_closed_pnls.append(pnl_value)
            trigger = max(1, int(getattr(config, 'LOSS_COOLDOWN_TRIGGER', 2)))
            losses = 0
            for pnl in reversed(self.recent_closed_pnls):
                if pnl < 0:
                    losses += 1
                else:
                    break
            if losses >= trigger:
                cooldown_hours = float(getattr(config, 'LOSS_COOLDOWN_HOURS', 4))
                self.cooldown_until = time.time() + cooldown_hours * 3600
                logger.warning(f"[{self.name}] 连续亏损 {losses} 笔，进入冷却模式 {cooldown_hours} 小时。")

        bucket = str(strategy_bucket or '').strip()
        if not bucket:
            return
        streaks = getattr(self, 'bucket_loss_streaks', None)
        if streaks is None:
            self.bucket_loss_streaks = {}
            streaks = self.bucket_loss_streaks
        streaks[bucket] = int(streaks.get(bucket, 0) or 0) + 1 if pnl_value < 0 else 0
        if pnl_value >= 0:
            self.bucket_cooldowns.pop(bucket, None)
            monitor = getattr(self, 'monitor', None)
            if monitor is not None:
                try:
                    ToolKit.save_runtime_state(monitor._runtime_state_file(), monitor, getattr(monitor, 'accounts', []))
                except Exception as e:
                    logger.warning(f"保存策略桶状态失败: {e}")
            return
        trigger = max(1, int(getattr(config, 'BUCKET_LOSS_COOLDOWN_TRIGGER', 2)))
        if streaks[bucket] >= trigger:
            cooldown_hours = float(getattr(config, 'BUCKET_LOSS_COOLDOWN_HOURS', 2))
            self.bucket_cooldowns[bucket] = time.time() + cooldown_hours * 3600
            logger.warning(f"[{self.name}] 策略桶 {bucket} 连续亏损 {streaks[bucket]} 笔，暂停 {cooldown_hours} 小时。")
        monitor = getattr(self, 'monitor', None)
        if monitor is not None:
            try:
                ToolKit.save_runtime_state(monitor._runtime_state_file(), monitor, getattr(monitor, 'accounts', []))
            except Exception as e:
                logger.warning(f"保存策略桶状态失败: {e}")

    def _cooldown_active(self):
        if not bool(getattr(config, 'ENABLE_LOSS_COOLDOWN', True)):
            self.cooldown_until = 0.0
            self.recent_closed_pnls.clear()
            return False
        if self.cooldown_until and time.time() >= self.cooldown_until:
            self.cooldown_until = 0.0
            self.recent_closed_pnls.clear()
            logger.info(f"[{self.name}] 连续亏损冷却已结束，恢复正常开仓。")
            return False
        return self.cooldown_until > time.time()

    def _bucket_cooldown_active(self, strategy_bucket, now_ts=None):
        bucket = str(strategy_bucket or '')
        if not bucket:
            return False
        current_ts = float(now_ts if now_ts is not None else time.time())
        expires_at = float(self.bucket_cooldowns.get(bucket, 0.0) or 0.0)
        if expires_at and current_ts >= expires_at:
            self.bucket_cooldowns.pop(bucket, None)
            return False
        return expires_at > current_ts

    def _pending_confirmation_key(self, symbol):
        return f"{self.name}:{symbol}"

    def _m15_confirmation_status(self, data, side):
        price_15m = float(data.get('price_15m') or data.get('price') or 0.0)
        ema20_15m = float(data.get('ema20_15m') or 0.0)
        vwap_15m = float(data.get('vwap_15m') or 0.0)
        macd_15m = float(data.get('macd_15m') or 0.0)
        macd_signal_15m = float(data.get('macd_signal_15m') or 0.0)
        prev_macd_15m = float(data.get('prev_macd_15m') or 0.0)
        prev_macd_signal_15m = float(data.get('prev_macd_signal_15m') or 0.0)
        cci_15m = float(data.get('cci_15m') or 0.0)
        prev_cci_15m = float(data.get('prev_cci_15m') or 0.0)
        closed_cci_15m = float(data.get('closed_cci_15m') or prev_cci_15m or cci_15m or 0.0)
        prev_closed_cci_15m = float(data.get('prev_closed_cci_15m') or closed_cci_15m or 0.0)
        closed_above_ema20 = int(data.get('closed_above_ema20_count_15m') or 0)
        closed_below_ema20 = int(data.get('closed_below_ema20_count_15m') or 0)
        trading_params = _get_symbol_trading_params(data.get('symbol'))
        reduced_scaler = float(trading_params.get('m15_reduced_scaler', getattr(config, 'M15_CONFIRM_REDUCED_SCALER', 0.7)) or 0.7)
        vwap_ready = vwap_15m > 0
        atr_1h = float(data.get('atr') or 0.0)
        pullback_buffer = atr_1h * 0.35 if atr_1h > 0 else max(price_15m * 0.003, 1e-9)

        if side == "LONG":
            price_ok = price_15m > ema20_15m
            macd_ok = macd_15m > macd_signal_15m and prev_macd_15m <= prev_macd_signal_15m
            cci_ok = closed_cci_15m > -100 and closed_cci_15m > prev_closed_cci_15m
            vwap_ok = vwap_ready and price_15m > vwap_15m
            hold_ok = closed_above_ema20 >= 2
            near_ema_ok = price_ok and abs(price_15m - ema20_15m) <= pullback_buffer
            near_vwap_ok = (not vwap_ready) or (vwap_ok and abs(price_15m - vwap_15m) <= pullback_buffer)
        else:
            price_ok = price_15m < ema20_15m
            macd_ok = macd_15m < macd_signal_15m and prev_macd_15m >= prev_macd_signal_15m
            cci_ok = closed_cci_15m < 100 and closed_cci_15m < prev_closed_cci_15m
            vwap_ok = vwap_ready and price_15m < vwap_15m
            hold_ok = closed_below_ema20 >= 2
            near_ema_ok = price_ok and abs(price_15m - ema20_15m) <= pullback_buffer
            near_vwap_ok = (not vwap_ready) or (vwap_ok and abs(price_15m - vwap_15m) <= pullback_buffer)

        pullback_ok = near_ema_ok and near_vwap_ok

        score = 0
        if macd_ok:
            score += 1
        if cci_ok:
            score += 1
        if vwap_ok:
            score += 1
        if hold_ok:
            score += 1

        normal_entry = price_ok and pullback_ok and score >= 3
        reduced_entry = score == 2 and price_ok and hold_ok and pullback_ok
        entry_multiplier = 1.0 if normal_entry else (reduced_scaler if reduced_entry else None)

        return {
            'passed': bool(normal_entry or reduced_entry),
            'price_ok': price_ok,
            'macd_ok': macd_ok,
            'cci_ok': cci_ok,
            'vwap_ok': vwap_ok,
            'hold_ok': hold_ok,
            'pullback_ok': pullback_ok,
            'score': score,
            'price_15m': price_15m,
            'ema20_15m': ema20_15m,
            'vwap_15m': vwap_15m,
            'cci_15m': cci_15m,
            'closed_cci_15m': closed_cci_15m,
            'prev_closed_cci_15m': prev_closed_cci_15m,
            'vwap_ready': vwap_ready,
            'entry_multiplier': entry_multiplier,
            'entry_mode': 'normal' if normal_entry else ('reduced' if reduced_entry else 'wait'),
        }

    def _trend_direction_value(self, ema_status):
        text = str(ema_status or "")
        if any(k in text for k in ["多头排列", "金叉"]):
            return 1
        if any(k in text for k in ["空头排列", "死叉"]):
            return -1
        return 0

    def _detect_reversal_pattern(self, candles, direction):
        try:
            if not candles or len(candles) < 2:
                return None

            parsed = []
            for candle in candles[-3:]:
                parsed.append({
                    "open": float(candle.get("open", 0.0)),
                    "high": float(candle.get("high", 0.0)),
                    "low": float(candle.get("low", 0.0)),
                    "close": float(candle.get("close", 0.0)),
                })

            prev = parsed[-2]
            curr = parsed[-1]
            prev_body_low = min(prev["open"], prev["close"])
            prev_body_high = max(prev["open"], prev["close"])
            curr_body_low = min(curr["open"], curr["close"])
            curr_body_high = max(curr["open"], curr["close"])
            curr_body = abs(curr["close"] - curr["open"])
            curr_range = max(curr["high"] - curr["low"], 1e-9)
            upper_shadow = curr["high"] - curr_body_high
            lower_shadow = curr_body_low - curr["low"]

            bullish_engulf = prev["close"] < prev["open"] and curr["close"] > curr["open"] and curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
            bearish_engulf = prev["close"] > prev["open"] and curr["close"] < curr["open"] and curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
            hammer = curr["close"] >= curr["open"] and lower_shadow >= curr_body * 2 and upper_shadow <= max(curr_body, curr_range * 0.15)
            shooting_star = curr["close"] <= curr["open"] and upper_shadow >= curr_body * 2 and lower_shadow <= max(curr_body, curr_range * 0.15)

            morning_star = False
            evening_star = False
            if len(parsed) >= 3:
                first, second, third = parsed[-3], parsed[-2], parsed[-1]
                first_body = abs(first["close"] - first["open"])
                second_body = abs(second["close"] - second["open"])
                third_body = abs(third["close"] - third["open"])
                first_mid = (first["open"] + first["close"]) / 2.0
                morning_star = first["close"] < first["open"] and second_body <= max(first_body * 0.5, 1e-9) and third["close"] > third["open"] and third["close"] >= first_mid and third_body >= second_body
                evening_star = first["close"] > first["open"] and second_body <= max(first_body * 0.5, 1e-9) and third["close"] < third["open"] and third["close"] <= first_mid and third_body >= second_body

            if direction == "LONG":
                if bullish_engulf:
                    return "吞没形态"
                if hammer:
                    return "锤子线"
                if morning_star:
                    return "启明星"
            else:
                if bearish_engulf:
                    return "吞没形态"
                if shooting_star:
                    return "射击之星"
                if evening_star:
                    return "黄昏星"
            return None
        except Exception:
            return None

    def _nearest_key_level(self, direction, data, current_price):
        if direction == "LONG":
            levels = [
                data.get("support_1h"),
                data.get("support_4h"),
                data.get("support_1d"),
                data.get("ema200_support_4h"),
                data.get("ema200_support_1d"),
            ]
            valid_levels = [float(v) for v in levels if v and float(v) < current_price]
            return max(valid_levels) if valid_levels else None

        levels = [
            data.get("resistance_1h"),
            data.get("resistance_4h"),
            data.get("resistance_1d"),
            data.get("ema200_resistance_4h"),
            data.get("ema200_resistance_1d"),
        ]
        valid_levels = [float(v) for v in levels if v and float(v) > current_price]
        return min(valid_levels) if valid_levels else None

    def _rr_obstacle_levels(self, direction, data, current_price):
        direction = str(direction).upper()
        current_price = float(current_price or 0.0)
        if direction == 'LONG':
            levels = [
                data.get('resistance_1h'),
                data.get('resistance_4h'),
                data.get('resistance_1d'),
                data.get('ema200_resistance_4h'),
                data.get('ema200_resistance_1d'),
            ]
            valid_levels = [float(v) for v in levels if v and float(v) > current_price]
            return min(valid_levels) if valid_levels else None

        levels = [
            data.get('support_1h'),
            data.get('support_4h'),
            data.get('support_1d'),
            data.get('ema200_support_4h'),
            data.get('ema200_support_1d'),
        ]
        valid_levels = [float(v) for v in levels if v and float(v) < current_price]
        return max(valid_levels) if valid_levels else None

    def _rr_obstacle_levels_by_scope(self, direction, data, current_price, scope='nearest'):
        direction = str(direction).upper()
        current_price = float(current_price or 0.0)
        if direction == 'LONG':
            if scope == 'near':
                levels = [data.get('resistance_1h')]
            elif scope == 'extended':
                levels = [
                    data.get('resistance_4h'),
                    data.get('resistance_1d'),
                    data.get('ema200_resistance_4h'),
                    data.get('ema200_resistance_1d'),
                ]
            else:
                levels = [
                    data.get('resistance_1h'),
                    data.get('resistance_4h'),
                    data.get('resistance_1d'),
                    data.get('ema200_resistance_4h'),
                    data.get('ema200_resistance_1d'),
                ]
            valid_levels = []
            for value in levels:
                try:
                    level = float(value)
                    if level > current_price:
                        valid_levels.append(level)
                except (TypeError, ValueError):
                    continue
            return min(valid_levels) if valid_levels else None

        if scope == 'near':
            levels = [data.get('support_1h')]
        elif scope == 'extended':
            levels = [
                data.get('support_4h'),
                data.get('support_1d'),
                data.get('ema200_support_4h'),
                data.get('ema200_support_1d'),
            ]
        else:
            levels = [
                data.get('support_1h'),
                data.get('support_4h'),
                data.get('support_1d'),
                data.get('ema200_support_4h'),
                data.get('ema200_support_1d'),
            ]
        valid_levels = []
        for value in levels:
            try:
                level = float(value)
                if level < current_price:
                    valid_levels.append(level)
            except (TypeError, ValueError):
                continue
        return max(valid_levels) if valid_levels else None

    def _compute_btc_near_extended_rr_preview(self, direction, data, current_price):
        near_preview = _compute_realistic_rr_preview(
            direction,
            data,
            current_price,
            lambda d, payload, price: self._rr_obstacle_levels_by_scope(d, payload, price, scope='near'),
        )
        extended_preview = _compute_realistic_rr_preview(
            direction,
            data,
            current_price,
            lambda d, payload, price: self._rr_obstacle_levels_by_scope(d, payload, price, scope='extended'),
        )
        return near_preview, extended_preview

    def _preview_realistic_rr(self, direction, data, current_price):
        return _compute_realistic_rr_preview(direction, data, current_price, self._rr_obstacle_levels)

    def should_override_conflict(self, direction, data):
        try:
            direction = str(direction).upper()
            current_price = float(data.get("price") or 0.0)
            current_score = float(data.get("long_score", 0.0) if direction == "LONG" else data.get("short_score", 0.0))
            expected_dir = 1 if direction == "LONG" else -1
            trend_dir_4h = self._trend_direction_value(data.get("ema_status_4h"))
            trend_dir_1d = self._trend_direction_value(data.get("ema_status_1d"))

            conflict_frames = []
            if trend_dir_4h == -expected_dir:
                conflict_frames.append("4H")
            if trend_dir_1d == -expected_dir:
                conflict_frames.append("1D")
            if not conflict_frames:
                return None

            defense_low = float(data.get("defense_low") or current_price)
            defense_high = float(data.get("defense_high") or current_price)
            sl_price = defense_low * 0.995 if direction == "LONG" else defense_high * 1.005
            risk = abs(current_price - sl_price)
            if risk <= 0:
                return {
                    "direction": direction,
                    "conflict_frames": conflict_frames,
                    "score": 0,
                    "passed": False,
                    "pass_mode": "reject",
                    "scaler": 0.0,
                    "required_ai_confidence": 0.0,
                    "rr": 0.0,
                    "pattern": "无",
                    "near_key_level": False,
                    "key_level": None,
                    "current_score": current_score,
                    "cci_4h": float(data.get("cci_4h") or 0.0),
                    "prev_cci_4h": float(data.get("prev_cci_4h") or 0.0),
                    "score_items": [],
                }

            closest_obstacle = self._rr_obstacle_levels(direction, data, current_price)
            reward = abs(closest_obstacle - current_price) if closest_obstacle is not None else 0.0
            rr = reward / risk if reward > 0 else 0.0

            pattern_name = self._detect_reversal_pattern(data.get("recent_candles_4h") or data.get("recent_candles_1h") or [], direction)
            cci_4h = float(data.get("cci_4h") or 0.0)
            prev_cci_4h = float(data.get("prev_cci_4h") or 0.0)
            if direction == "LONG":
                cci_revert_ok = prev_cci_4h <= -150.0 and cci_4h >= -100.0
            else:
                cci_revert_ok = prev_cci_4h >= 150.0 and cci_4h <= 100.0

            key_level = self._nearest_key_level(direction, data, current_price)
            near_key_level = bool(key_level and abs(current_price - key_level) / max(current_price, 1e-9) < 0.01)

            score = 0
            score_items = []
            if pattern_name:
                score += 2
                score_items.append(f"反转K线+2({pattern_name})")
            if cci_revert_ok:
                score += 1
                score_items.append("4H CCI回归+1")
            if rr >= 2.5:
                score += 1
                score_items.append(f"RR+1({rr:.2f})")
            if current_score >= 6.5:
                score += 1
                score_items.append(f"1H评分+1({current_score:.1f})")
            if near_key_level:
                score += 1
                score_items.append(f"关键位邻近+1({key_level:.4f})")

            if score >= 3:
                pass_mode = "half_size"
                passed = True
                scaler = 0.5
                required_ai_confidence = 0.0
            elif score == 2:
                pass_mode = "micro_with_ai"
                passed = True
                scaler = 0.2
                required_ai_confidence = 70.0
            else:
                pass_mode = "reject"
                passed = False
                scaler = 0.0
                required_ai_confidence = 0.0

            return {
                "direction": direction,
                "conflict_frames": conflict_frames,
                "score": score,
                "passed": passed,
                "pass_mode": pass_mode,
                "scaler": scaler,
                "required_ai_confidence": required_ai_confidence,
                "rr": rr,
                "pattern": pattern_name or "无",
                "near_key_level": near_key_level,
                "key_level": key_level,
                "current_score": current_score,
                "cci_4h": cci_4h,
                "prev_cci_4h": prev_cci_4h,
                "score_items": score_items,
            }
        except Exception:
            return None

    def _dynamic_rr_profile(self, direction, data, current_price):
        direction = str(direction).upper()
        adx = float(data.get('adx_1h') or 0.0)
        prev_adx = float(data.get('prev_adx_1h') or 0.0)
        ema20 = float(data.get('ema20_1h') or 0.0)
        rsi = float(data.get('rsi_1h') or 0.0)
        boll_high = data.get('boll_high_1h')
        boll_low = data.get('boll_low_1h')
        atr = float(data.get('atr') or 0.0)

        near_ema20 = False
        if ema20 > 0 and atr > 0:
            near_ema20 = abs(current_price - ema20) <= max(atr * 0.6, current_price * 0.003)

        if direction == 'LONG':
            initial_stage = adx < 25 and current_price >= ema20 and near_ema20
            terminal_stage = boll_high is not None and current_price >= float(boll_high) and rsi > 70
        else:
            initial_stage = adx < 25 and current_price <= ema20 and near_ema20
            terminal_stage = boll_low is not None and current_price <= float(boll_low) and rsi < 30

        continuation_stage = adx >= 25 and adx > prev_adx

        # Live RR execution currently uses `min_rr` and `label` only.
        # The returned stage profile is descriptive here; it is not an additional sizing authority in this pass.
        if terminal_stage:
            return {'stage': 'trend_end', 'label': 'trend_end', 'min_rr': 1.5, 'scaler': 0.5}
        if continuation_stage:
            return {'stage': 'trend_continuation', 'label': 'trend_continuation', 'min_rr': 1.2, 'scaler': 1.0}
        if initial_stage:
            return {'stage': 'trend_initial', 'label': 'trend_initial', 'min_rr': 1.0, 'scaler': 0.7}
        return {'stage': 'trend_continuation', 'label': 'trend_continuation', 'min_rr': 1.2, 'scaler': 1.0}


    def _apply_low_score_ai_override(self, data, ai_decision, score_threshold):
        try:
            if not ai_decision:
                return None
            if str(data.get('symbol') or '').upper() == 'BTCUSDT':
                return None
            ai_action = str(ai_decision.get('action', 'PASS')).strip().upper()
            confidence = float(ai_decision.get('confidence', 0.0) or 0.0)
            if ai_action not in ['LONG', 'SHORT'] or confidence < 80:
                return None

            score_key = 'long_score' if ai_action == 'LONG' else 'short_score'
            original_score = float(data.get(score_key, 0.0) or 0.0)
            if original_score >= float(score_threshold or 0.0):
                return None

            pattern = self._detect_reversal_pattern(data.get('recent_candles_4h') or [], ai_action)
            strong_patterns = {'吞没形态', '启明星', '黄昏星', '锤子线', '射击之星'}
            if pattern not in strong_patterns:
                return None

            boosted_score = max(float(score_threshold or 0.0), 6.5)
            data[score_key] = max(original_score, boosted_score)
            price = float(data.get('price') or 0.0)
            atr = float(data.get('atr') or 0.0)
            if price > 0 and atr > 0:
                boosted_scaler = self.analyzer.calculate_position_scaler(data[score_key], atr, price, score_threshold)
                data['suggested_scaler'] = max(float(data.get('suggested_scaler', 0.0) or 0.0), float(boosted_scaler))
            data['ai_score_override'] = {
                'action': ai_action,
                'confidence': confidence,
                'pattern': pattern,
                'original_score': original_score,
                'boosted_score': boosted_score,
            }
            return data['ai_score_override']
        except Exception:
            return None

    def _derive_ai_entry_scaler(self, ai_action, algo_side, confidence, ai_reason):
        same_side = str(ai_action).upper() == str(algo_side).upper()
        reason_text = str(ai_reason or '')
        tentative_keywords = ["试多", "试空", "轻仓试", "尝试", "偏向", "博弈", "反弹概率较高"]
        cautious = any(k in reason_text for k in tentative_keywords)
        if same_side and confidence >= 72 and not cautious:
            return 1.0, False, 'AI强通过'
        if same_side and confidence >= 65:
            return (0.7 if not cautious else 0.5), cautious, 'AI谨慎通过'
        if same_side and confidence >= 60:
            return 0.3, True, 'AI边缘试仓'
        return 0.0, cautious, 'AI拒绝'

    def _normalize_ai_leverage_factor(self, prompt_mode, leverage_factor):
        if str(prompt_mode or '').lower() != 'aggressive':
            return 1.0
        allowed_factors = {0.5, 0.7, 0.85, 1.0}
        try:
            factor = float(leverage_factor)
        except (TypeError, ValueError):
            return 1.0
        return factor if factor in allowed_factors else 1.0

    def _resolve_final_leverage(self, symbol, base_leverage, ai_leverage_factor):
        try:
            base = float(base_leverage)
        except (TypeError, ValueError):
            base = 1.0
        factor = self._normalize_ai_leverage_factor(getattr(self, 'prompt_mode', 'aggressive'), ai_leverage_factor)
        raw_leverage = int(round(base * factor))
        if symbol in getattr(config, 'MAINSTREAM', []):
            bounds = (8, 12) if symbol in ('BTCUSDT', 'ETHUSDT') else (5, 8)
        else:
            bounds = (2, 4)
        return max(bounds[0], min(bounds[1], raw_leverage))

    def _save_to_trade_history(self, is_simulate, symbol, trade_id, open_time, side, leverage, entry, close_price, close_reason, net_pnl, balance, meta=None, data=None):
        filename = f"trades_history_{self.name}.csv"
        mode_str = "模拟盘" if is_simulate else "实盘"
        meta = meta or {}
        data = data or {}
        try:
            trade_rec = {
                "账户": self.name,
                "平仓时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "模式": mode_str,
                "开仓时间": open_time,
                "订单ID": trade_id,
                "交易对": symbol,
                "方向": side,
                "杠杆": f"{leverage}x",
                "开仓均价": entry,
                "平仓均价": close_price,
                "平仓原因": close_reason,
                "来源": "机器人",
                "机器人交易ID": trade_id,
                "净盈亏(U)": round(net_pnl, 4),
                "当前余额(U)": round(balance, 2)
            }
            pd.DataFrame([trade_rec]).to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"[{self.name}] 记录交易历史 CSV 失败: {e}")

        try:
            estimated_cost_u = data.get('estimated_cost_u')
            if estimated_cost_u is None:
                estimated_cost_u = (data.get('rr_costs') or {}).get('estimated_cost_u')
            actual_cost_u = None
            try:
                estimated_value = float(estimated_cost_u) if estimated_cost_u is not None else None
                gross_pnl = float(meta.get('gross_pnl', 0.0) or 0.0)
                realized_pnl = float(net_pnl)
                if estimated_value is not None and gross_pnl:
                    actual_cost_u = max(0.0, round(gross_pnl - realized_pnl, 4))
            except Exception:
                actual_cost_u = None
            ToolKit.log_trade_close(
                symbol=symbol,
                trade_id=trade_id,
                open_time=open_time,
                side=side,
                notional=float(meta.get('notional', 0.0) or 0.0),
                entry_price=entry,
                close_price=close_price,
                close_reason=close_reason,
                net_pnl=net_pnl,
                current_balance=balance,
                enable_ai=self.enable_ai,
                leverage=leverage,
                market_regime=meta.get('market_regime', data.get('market_regime', '')),
                strategy_bucket=meta.get('strategy_bucket', data.get('strategy_bucket', '')),
                estimated_cost_u=estimated_cost_u,
                actual_cost_u=actual_cost_u,
            )
        except Exception as e:
            logger.error(f"[{self.name}] 回写 AI 决策日志失败: {e}")

        self._record_closed_trade(net_pnl, strategy_bucket=meta.get('strategy_bucket', data.get('strategy_bucket', '')))
        self._mark_symbol_reentry_cooldown(symbol, close_reason)

    def _clear_signal_cooldown(self, symbol, side, close_reason):
        cache_key = f"{self.name}_{symbol}_{side}"
        if cache_key in self.monitor.algo_cache:
            self.monitor.algo_cache.pop(cache_key, None)
            logger.info(f"[{self.name}] {symbol} {side} {close_reason}已清空同向信号冷却，等待重新确认后入场。")

    def _mark_symbol_reentry_cooldown(self, symbol, close_reason, now_ts=None):
        if '反转' not in str(close_reason or ''):
            return
        cooldown_minutes = float(getattr(config, 'SYMBOL_REENTRY_COOLDOWN_MINUTES', 45))
        expires_at = float(now_ts if now_ts is not None else time.time()) + cooldown_minutes * 60
        self.symbol_reentry_cooldowns[str(symbol)] = expires_at
        monitor = getattr(self, 'monitor', None)
        if monitor is not None:
            try:
                ToolKit.save_runtime_state(monitor._runtime_state_file(), monitor, getattr(monitor, 'accounts', []))
            except Exception as e:
                logger.warning(f"保存同币种重入冷却失败: {e}")
        logger.info(f"[{self.name}] {symbol} 主动反转平仓后进入 {cooldown_minutes:.0f} 分钟重入冷却。")

    def _symbol_reentry_cooldown_allows_override(self, symbol, ai_confidence=0.0, net_rr=0.0, m15_confirmation_passed=False, now_ts=None):
        cooldowns = getattr(self, 'symbol_reentry_cooldowns', None)
        if cooldowns is None:
            self.symbol_reentry_cooldowns = {}
            cooldowns = self.symbol_reentry_cooldowns
        current_ts = float(now_ts if now_ts is not None else time.time())
        expires_at = float(cooldowns.get(str(symbol), 0.0) or 0.0)
        if not expires_at:
            return True
        if current_ts >= expires_at:
            cooldowns.pop(str(symbol), None)
            return True
        min_confidence = float(getattr(config, 'SYMBOL_REENTRY_OVERRIDE_AI_CONFIDENCE', 75))
        min_net_rr = float(getattr(config, 'SYMBOL_REENTRY_OVERRIDE_NET_RR', 1.5))
        return bool(m15_confirmation_passed) and float(ai_confidence or 0.0) >= min_confidence and float(net_rr or 0.0) >= min_net_rr

    def process_simulate_trading(self, data, spec):
        symbol, current_price, signal_text = data['symbol'], data['price'], data['signal']
        FEE = getattr(config, 'FEE_RATE', 0.0004)
        RR_RATIO = getattr(config, 'TAKE_PROFIT_RR', 2.0)
        # ==========================================
        # ==========================================
        if symbol in self.open_positions:
            pos = self.open_positions[symbol]
            side, entry, sl, tp, notional = pos['side'], pos['entry'], pos['sl'], pos.get('tp', 0), pos['notional']

            pnl_pct = (current_price - entry) / entry if side == "LONG" else (entry - current_price) / entry
            gross_floating = notional * pnl_pct
            estimated_close_fee = notional * 0.0004
            pos['unrealized_pnl'] = gross_floating - estimated_close_fee

            close_reason = None
            if side == "LONG":
                if current_price <= sl:
                    close_reason = "跌破防守止损"
                elif current_price >= tp:
                    close_reason = "跌破防守止损"
                elif any(k in signal_text for k in ["抄底", "看多"]):
                    if self.enable_ai_close:
                        ai_dec = self.monitor._evaluate_ai_with_cache(symbol, "SHORT", data, signal_text)
                        if ai_dec and ai_dec.get('action') == "SHORT":
                            close_reason = "AI确认反转平多"
            elif side == "SHORT":
                if current_price >= sl:
                    close_reason = "跌破防守止损"
                elif current_price <= tp:
                    close_reason = "跌破防守止损"
                elif any(k in signal_text for k in ["抄底", "看多"]):
                    if self.enable_ai_close:
                        ai_dec = self.monitor._evaluate_ai_with_cache(symbol, "LONG", data, signal_text)
                        if ai_dec and ai_dec.get('action') == "LONG":
                            close_reason = "AI确认反转平多"

            if close_reason:
                net_pnl = gross_floating - (notional * FEE)

                if self.funds_mode == 'SHARED':
                    self.virtual_balance += (pos['margin'] + net_pnl)

                self.total_trades += 1
                if net_pnl > 0:
                    self.win_trades += 1

                current_balance = self.get_base_capital()
                self._save_to_trade_history(True, symbol, pos['trade_id'], pos['open_time'], side, spec['leverage'], entry, current_price, close_reason, net_pnl, current_balance, meta=pos, data=data)
                self._clear_signal_cooldown(symbol, side, close_reason)
                del self.open_positions[symbol]

        # ==========================================
        # ==========================================
        if symbol not in self.open_positions:
            max_pos = getattr(config, 'MAX_OPEN_POSITIONS', 3)
            if len(self.open_positions) >= max_pos:
                return
            long_score = float(data.get('long_score', 0.0))
            short_score = float(data.get('short_score', 0.0))
            threshold = float(data.get('score_threshold', getattr(config, 'MIN_OPEN_SCORE', 4.5)))
            negative_words = ["不建议", "拒绝", "放弃", "谨慎", "切勿", "暂停"]
            if any(n in signal_text for n in negative_words):
                logger.info(f"[{self.name}] {symbol} 本轮未触发AI | LONG {long_score:.1f} / SHORT {short_score:.1f} / 阈值 {threshold:.1f} | 原因: 信号命中否定词，跳过开仓。")
                self.monitor._record_open_rejection(self.name, symbol, '信号命中否定词')
                return
            algo_side = "LONG" if any(k in signal_text for k in ["抄底", "看多", "反弹"]) else ("SHORT" if any(k in signal_text for k in ["逃顶", "看空"]) else None)
            if not algo_side:
                logger.info(f"[{self.name}] {symbol} 本轮未触发AI | LONG {long_score:.1f} / SHORT {short_score:.1f} / 阈值 {threshold:.1f} | 原因: 量化信号未形成明确方向。")
                return

            dynamic_margin_ratio = spec['margin_ratio']
            margin_needed = self.get_base_capital() * dynamic_margin_ratio

            open_fee = (margin_needed * spec['leverage']) * FEE
            if self.get_base_capital() < (margin_needed + open_fee):
                logger.warning(f"[{self.name}] 模拟盘余额不足，无法开仓 {symbol}")
                return

            if self.funds_mode == 'SHARED':
                self.virtual_balance -= (margin_needed + open_fee)

            # 记录持仓
            self.open_positions[symbol] = {
                "trade_id": f"SIM_{int(time.time())}",
                "side": algo_side,
                "entry": current_price,
                "sl": data.get('defense_low') if algo_side == "LONG" else data.get('defense_high'),
                "tp": current_price * 1.05 if algo_side == "LONG" else current_price * 0.95,
                "margin": margin_needed,
                "notional": margin_needed * spec['leverage'],
                "open_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            logger.info(f"[{self.name}] 模拟开仓成功，已锁定保证金: {margin_needed:.2f} U")

    def process_real_trading(self, data, spec):
        symbol, current_price, signal_text = data['symbol'], data['price'], data['signal']

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        if symbol in self.real_trade_meta and symbol not in self.real_positions:
            meta = self.real_trade_meta[symbol]

            if meta['side'] == 'LONG':
                close_reason = "触及止损(或被手动强平)" if current_price <= meta['sl'] * 1.01 else "触及止盈"
            else:
                close_reason = "触及止损(或被手动强平)" if current_price >= meta['sl'] * 0.99 else "触及止盈"

            net_pnl = _get_realized_net_pnl(self.engine, symbol, meta.get('qty', 0.0), meta['entry'], current_price, meta['side'])

            logger.warning(f"[{self.name}] 实盘被动结算: {symbol} {meta['side']} | {close_reason}")

            current_balance = self.get_base_capital()
            self._save_to_trade_history(False, symbol, meta.get('trade_id', '未知'), meta.get('open_time', '未知'), meta['side'], spec['leverage'], meta['entry'], current_price, close_reason, net_pnl, current_balance, meta=meta, data=data)
            self._clear_signal_cooldown(symbol, meta['side'], close_reason)

            pnl_color = "#28a745" if net_pnl > 0 else "#dc3545"
            close_msg = (
                f"### [{self.name}] 被动平仓: {symbol} {meta['side']}\n"
                f"- **平仓原因**: {close_reason}\n"
                f"- **开仓均价**: {meta['entry']:.5f}\n"
                f"- **当前均价**: {current_price:.5f}\n"
                f"- **预计盈亏**: <font color='{pnl_color}'>**{net_pnl:+.2f} U**</font>\n"
                f"- **当前余额**: {current_balance:.2f} U"
            )
            ToolKit.send_dingtalk_msg(close_msg, title=f"[{self.name}] {symbol.replace('USDT', '')} 被动平仓")

            del self.real_trade_meta[symbol]
            ToolKit.save_active_trades(self.real_trade_meta, account_name=self.name)

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        if symbol in self.real_positions:
            pos = self.real_positions[symbol]
            side, qty, entry = pos['side'], pos['qty'], pos['entry']

            close_reason = None
            meta = self.real_trade_meta.get(symbol, {})
            trade_id = meta.get('trade_id', f"REAL_{symbol}_{int(time.time())}")
            open_time_text = meta.get('open_time', '')
            holding_minutes = 0.0
            try:
                open_dt = datetime.strptime(open_time_text, '%Y-%m-%d %H:%M:%S')
                holding_minutes = max((datetime.now() - open_dt).total_seconds() / 60.0, 0.0)
            except Exception:
                holding_minutes = 0.0

            notional = float(meta.get('notional', 0.0) or 0.0)
            pnl_pct = ((current_price - entry) / entry * 100.0) if side == 'LONG' else ((entry - current_price) / entry * 100.0)
            unrealized_pnl = notional * (pnl_pct / 100.0) if notional > 0 else 0.0
            position_data = {
                'account_name': self.name,
                'symbol': symbol,
                'trade_id': trade_id,
                'side': side,
                'entry_price': round(float(entry), 6),
                'current_price': round(float(current_price), 6),
                'unrealized_pnl': round(unrealized_pnl, 4),
                'unrealized_pnl_pct': round(pnl_pct, 4),
                'holding_minutes': round(holding_minutes, 1),
                'stop_loss': meta.get('sl'),
                'take_profit': meta.get('tp'),
                'tp_phase': meta.get('tp_phase', 0),
                'trend_state': data.get('trend_state'),
                'ema_trend': data.get('ema_trend'),
                'macd_info': data.get('macd_info'),
                'cci_info': data.get('cci_info'),
                'support_1h': data.get('support_1h'),
                'resistance_1h': data.get('resistance_1h'),
                'support_4h': data.get('support_4h'),
                'resistance_4h': data.get('resistance_4h'),
                'price_15m': data.get('price_15m'),
                'ema20_15m': data.get('ema20_15m'),
                'vwap_15m': data.get('vwap_15m'),
                'cci_15m': data.get('cci_15m'),
            }

            if self.enable_ai:
                position_ai_decision, position_ai_state, position_ai_fresh = self.monitor._evaluate_position_ai_with_cache(
                    self.name,
                    symbol,
                    trade_id,
                    position_data,
                    prompt_mode=getattr(self, 'prompt_mode', 'aggressive'),
                )
                if position_ai_decision and position_ai_state and position_ai_fresh:
                    execution_state = self._execute_position_ai_action(symbol, trade_id, position_ai_decision, position_ai_state, position_data)
                    self.monitor._log_position_ai_execution(self.name, symbol, trade_id, execution_state)
                    position_key = self.monitor._position_ai_key(self.name, symbol, trade_id)
                    if self.monitor._should_notify_position_ai(position_key, position_ai_decision, position_ai_state, position_data):
                        card_msg = ToolKit.format_position_ai_card(position_data, position_ai_decision, position_ai_state)
                        ToolKit.send_dingtalk_msg(card_msg, title=f"[{self.name}] {symbol.replace('USDT', '')} 持仓后AI评估")
            strategy_bucket = str(meta.get('strategy_bucket', '') or '')
            long_close_keywords = ["逃顶", "看空", "减仓"]
            short_close_keywords = ["抄底", "看多"]
            if strategy_bucket == 'reversal_probe':
                long_close_keywords = ["逃顶", "看空", "减仓", "等待确认"]
                short_close_keywords = ["抄底", "看多", "等待确认"]

            if side == "LONG" and any(k in signal_text for k in long_close_keywords):
                if not self.enable_ai_close:
                    close_reason = "算法反转主动平多"
                else:
                    ai_dec = self.monitor._evaluate_ai_with_cache(symbol, "SHORT", data, signal_text)
                    if ai_dec and str(ai_dec.get('action')).upper() == "SHORT" and float(ai_dec.get('confidence', 0)) >= 60:
                        close_reason = f"AI确认反转平多: {ai_dec.get('reason', '')}"

            elif side == "SHORT" and any(k in signal_text for k in short_close_keywords):
                if not self.enable_ai_close:
                    close_reason = "算法反转主动平空"
                else:
                    ai_dec = self.monitor._evaluate_ai_with_cache(symbol, "LONG", data, signal_text)
                    if ai_dec and str(ai_dec.get('action')).upper() == "LONG" and float(ai_dec.get('confidence', 0)) >= 60:
                        close_reason = f"AI确认反转平空: {ai_dec.get('reason', '')}"

            if close_reason:
                logger.warning(f"[{self.name}] 实盘预警: {symbol} 触发 {close_reason}，请求市价强平。")
                self.engine.cancel_all_orders(symbol)

                close_order_id = self.engine.execute_market_close(symbol, side, qty)
                if close_order_id:
                    realized_pnl = get_realized_pnl_record(self.engine, symbol, qty, entry, current_price, side, order_id=close_order_id)
                    net_pnl = float(realized_pnl['net_pnl'])
                    if realized_pnl.get('avg_price'):
                        current_price = float(realized_pnl['avg_price'])

                    meta = self.real_trade_meta.get(symbol, {})
                    trade_id = meta.get('trade_id', f"UNKNOWN_{int(time.time())}")
                    open_time = meta.get('open_time', '未知')

                    current_balance = self.get_base_capital()
                    self._save_to_trade_history(False, symbol, trade_id, open_time, side, spec['leverage'], entry, current_price, close_reason, net_pnl, current_balance, meta=meta, data=data)
                    self._clear_signal_cooldown(symbol, side, close_reason)

                    if symbol in self.real_trade_meta:
                        del self.real_trade_meta[symbol]
                        ToolKit.save_active_trades(self.real_trade_meta, account_name=self.name)

                    del self.real_positions[symbol]

                    pnl_color = "#28a745" if net_pnl > 0 else "#dc3545"
                    close_msg = (
                        f"### [{self.name}] \u4e3b\u52a8\u5e73\u4ed3: {symbol} {side}\n"
                        f"- **\u5e73\u4ed3\u539f\u56e0**: {close_reason}\n"
                        f"- **\u5f00\u4ed3\u5747\u4ef7**: {entry:.5f}\n"
                        f"- **\u5f53\u524d\u5747\u4ef7**: {current_price:.5f}\n"
                        f"- **\u9884\u8ba1\u76c8\u4e8f**: <font color='{pnl_color}'>**{net_pnl:+.2f} U**</font>"
                    )
                    ToolKit.send_dingtalk_msg(close_msg, title=f"[{self.name}] {symbol.replace('USDT', '')} \u4e3b\u52a8\u5e73\u4ed3")

        # ---------------------------------------------------------
        # ---------------------------------------------------------
        if symbol not in self.real_positions:
            max_pos = getattr(config, 'MAX_OPEN_POSITIONS', 3)
            if len(self.real_positions) >= max_pos:
                return
            if self._cooldown_active():
                cooldown_left = max(0, int(self.cooldown_until - time.time()))
                logger.info(f"[{self.name}] \u8fde\u7eed\u4e8f\u635f\u51b7\u5374\u4e2d\uff0c{symbol} \u6682\u505c\u5f00\u65b0\u5355\uff0c\u5269\u4f59 {cooldown_left // 60} \u5206\u949f\u3002")
                self.monitor._record_open_rejection(self.name, symbol, '\u8d26\u6237\u51b7\u5374\u4e2d')
                return
            strategy_bucket = str(data.get('strategy_bucket', '') or '')
            if self._bucket_cooldown_active(strategy_bucket):
                logger.info(f"[{self.name}] {symbol} \u7b56\u7565\u6876 {strategy_bucket} \u51b7\u5374\u4e2d\uff0c\u6682\u4e0d\u65b0\u5f00\u4ed3\u3002")
                self.monitor._record_open_rejection(self.name, symbol, '\u7b56\u7565\u6876\u51b7\u5374\u4e2d')
                return
            long_score = float(data.get('long_score', 0.0))
            short_score = float(data.get('short_score', 0.0))

            long_score = float(data.get('long_score', 0.0))
            short_score = float(data.get('short_score', 0.0))
            base_threshold = float(getattr(config, 'MIN_OPEN_SCORE', 5.5))
            threshold = base_threshold + 1.0 if (float(data.get('atr', 0.0)) / max(float(current_price), 1e-9)) > 0.04 else base_threshold
            data['score_threshold'] = threshold
            negative_words = ["不建议", "拒绝", "放弃", "谨慎", "切勿", "暂停"]
            if any(n in signal_text for n in negative_words):
                logger.info(f"[{self.name}] {symbol} 本轮未触发AI | LONG {long_score:.1f} / SHORT {short_score:.1f} / 阈值 {threshold:.1f} | 原因: 信号命中否定词，跳过开仓。")
                self.monitor._record_open_rejection(self.name, symbol, '信号命中否定词')
                return

            algo_side = "LONG" if any(k in signal_text for k in ["抄底", "看多", "异动", "刺客", "反弹"]) else ("SHORT" if any(k in signal_text for k in ["逃顶", "看空", "减仓"]) else None)
            ai_prefetched_decision = None

            pending_key = self._pending_confirmation_key(symbol)
            if not algo_side and self.enable_ai:
                probe_side = "LONG" if long_score > short_score else ("SHORT" if short_score > long_score else None)
                probe_score = long_score if probe_side == "LONG" else (short_score if probe_side == "SHORT" else 0.0)
                override_floor = max(threshold - 0.5, 0.0)
                override_ceiling = threshold
                if probe_side and override_floor <= probe_score < override_ceiling:
                    p_mode = getattr(self, 'prompt_mode', 'aggressive')
                    ai_prefetched_decision = self.monitor._evaluate_ai_with_cache(symbol, probe_side, data, signal_text, prompt_mode=p_mode)
                    override_meta = self._apply_low_score_ai_override(data, ai_prefetched_decision, threshold)
                    if override_meta:
                        algo_side = str(override_meta.get('action', probe_side)).upper()
                        long_score = float(data.get('long_score', long_score))
                        short_score = float(data.get('short_score', short_score))
                        logger.info(f"[{self.name}] {symbol} AI high-confidence score override active | action={algo_side} | confidence={override_meta.get('confidence', 0):.1f} | pattern={override_meta.get('pattern', 'none')} | score {override_meta.get('original_score', 0):.1f}->6.5")
            if not algo_side:
                if pending_key in self.monitor.pending_m15_confirmations:
                    self.monitor.pending_m15_confirmations.pop(pending_key, None)
                    logger.info(f"[{self.name}] {symbol} 1H \u5019\u9009\u65b9\u5411\u5df2\u6d88\u5931\uff0c\u53d6\u6d88\u539f 15M \u7b49\u5f85\u4fe1\u53f7\u3002")
                if ai_prefetched_decision:
                    precheck_action = str(ai_prefetched_decision.get('action', 'PASS')).upper()
                    logger.info(
                        f"[{self.name}] {symbol} \u672c\u8f6e\u672a\u5f62\u6210\u5f00\u4ed3\u5019\u9009 | AI\u9884\u5ba1\u7ed3\u679c: {precheck_action} | "
                        f"LONG {long_score:.1f} / SHORT {short_score:.1f} / \u9608\u503c {threshold:.1f} | "
                        f"\u539f\u56e0: \u91cf\u5316\u4fe1\u53f7\u672a\u5f62\u6210\u660e\u786e\u65b9\u5411\u3002"
                    )
                else:
                    logger.info(
                        f"[{self.name}] {symbol} \u672c\u8f6e\u672a\u89e6\u53d1AI | LONG {long_score:.1f} / SHORT {short_score:.1f} / \u9608\u503c {threshold:.1f} | "
                        f"\u539f\u56e0: \u91cf\u5316\u4fe1\u53f7\u672a\u5f62\u6210\u660e\u786e\u65b9\u5411\u3002"
                    )
                self.monitor._record_open_rejection(self.name, symbol, '\u91cf\u5316\u4fe1\u53f7\u672a\u5f62\u6210\u660e\u786e\u65b9\u5411')
                return
            pending_key = self._pending_confirmation_key(symbol)
            pending_signal = self.monitor.pending_close_confirmations.get(pending_key)
            pending_m15_signal = self.monitor.pending_m15_confirmations.get(pending_key)
            bar_open_1h_ms = int(data.get('bar_open_1h_ms') or 0)
            bar_open_15m_ms = int(data.get('bar_open_15m_ms') or 0)
            current_score = float(data.get('long_score', 0.0) if algo_side == "LONG" else data.get('short_score', 0.0))
            conflict_override_active = False
            symbol_params = _get_symbol_trading_params(symbol)
            m15_strong_signal_buffer = float(symbol_params.get('m15_strong_signal_buffer', getattr(config, "M15_CONFIRM_STRONG_SIGNAL_BUFFER", 1.0)) or 1.0)
            skip_m15_confirmation_for_strong_signal = current_score >= (threshold + m15_strong_signal_buffer)
            close_confirmation_passed = False
            m15_confirmation_passed = False
            m15_entry_multiplier = 1.0
            ai_decision = ai_prefetched_decision
            ai_caution_mode = False

            if pending_signal:
                if bar_open_1h_ms <= int(pending_signal.get('bar_open_ms', 0)):
                    logger.info(f"[{self.name}] {symbol} 等待 1H 收盘确认，当前 K 线尚未收完。")
                    self.monitor._record_open_rejection(self.name, symbol, '等待1H收盘确认')
                    return
                if bar_open_1h_ms >= int(pending_signal.get('expire_at_ms', 0)):
                    logger.info(f"[{self.name}] {symbol} 收盘确认信号已过期，放弃本轮开仓。")
                    self.monitor.pending_close_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '1H收盘确认已过期')
                    return
                if algo_side != pending_signal.get('algo_side'):
                    logger.info(f"[{self.name}] {symbol} 收盘后方向已改变，放弃原等待信号。")
                    self.monitor.pending_close_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '1H收盘后方向已改变')
                    return
                if current_score < float(pending_signal.get('score_at_signal', current_score)) - 0.5:
                    logger.info(f"[{self.name}] {symbol} 收盘确认失败，评分回落超过 0.5 分，放弃开仓。")
                    self.monitor.pending_close_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '1H收盘确认失败')
                    return
                close_confirmation_passed = True
                ai_decision = pending_signal.get('ai_decision') or {}
                ai_caution_mode = bool(pending_signal.get('ai_caution_mode', False))
                data['ai_entry_scaler'] = float(pending_signal.get('ai_entry_scaler', 1.0) or 1.0)
                data['ai_entry_label'] = pending_signal.get('ai_entry_label', '')
                data['ai_leverage_factor'] = float(pending_signal.get('ai_leverage_factor', 1.0) or 1.0)
                algo_side = str(pending_signal.get('open_side', algo_side)).upper()
                logger.info(f"[{self.name}] {symbol} 收盘确认通过，继续处理待执行开仓。")
                self.monitor.pending_close_confirmations.pop(pending_key, None)
            elif pending_m15_signal:
                pending_open_side = str(pending_m15_signal.get('open_side', algo_side)).upper()
                if algo_side != pending_m15_signal.get('algo_side'):
                    logger.info(f"[{self.name}] {symbol} 15M 等待期间方向已改变，放弃原等待信号。")
                    self.monitor.pending_m15_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '15M等待期间方向已改变')
                    return
                if bar_open_1h_ms > int(pending_m15_signal.get('origin_1h_bar_ms', bar_open_1h_ms)):
                    logger.info(f"[{self.name}] {symbol} 1H 信号已切换，放弃原 15M 等待信号。")
                    self.monitor.pending_m15_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '1H信号已切换')
                    return
                if bar_open_15m_ms >= int(pending_m15_signal.get('expire_at_ms', 0)):
                    logger.info(f"[{self.name}] {symbol} 15M 确认超时，放弃本轮开仓。")
                    self.monitor.pending_m15_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '15M确认超时')
                    return
                if current_score < float(pending_m15_signal.get('score_at_signal', current_score)) - 0.5:
                    logger.info(f"[{self.name}] {symbol} 15M 确认失败，评分回落超过 0.5 分，放弃开仓。")
                    self.monitor.pending_m15_confirmations.pop(pending_key, None)
                    self.monitor._record_open_rejection(self.name, symbol, '15M确认失败')
                    return
                m15_status = self._m15_confirmation_status(data, pending_open_side)
                data['pullback_ok'] = bool(m15_status.get('pullback_ok'))
                data['m15_score'] = int(m15_status.get('score', 0) or 0)
                data['m15_entry_mode'] = m15_status.get('entry_mode', 'wait')
                data['m15_price_ok'] = bool(m15_status.get('price_ok'))
                data['m15_hold_ok'] = bool(m15_status.get('hold_ok'))
                if not m15_status.get('passed'):
                    logger.info(f"[{self.name}] {symbol} 15M 评分未达标，继续等待 | score={m15_status.get('score', 0)}/4 | price_vs_ema20={'OK' if m15_status.get('price_ok') else 'NO'} | macd={'OK' if m15_status.get('macd_ok') else 'NO'} | cci_closed={'OK' if m15_status.get('cci_ok') else 'NO'} | vwap={'OK' if m15_status.get('vwap_ok') else ('MISS' if not m15_status.get('vwap_ready') else 'NO')} | hold2={'OK' if m15_status.get('hold_ok') else 'NO'} | pullback={'OK' if m15_status.get('pullback_ok') else 'NO'}")
                    self.monitor._record_open_rejection(self.name, symbol, '15M评分未达标')
                    return
                algo_side = pending_open_side
                m15_confirmation_passed = True
                m15_entry_multiplier = float(m15_status.get('entry_multiplier') or 1.0)
                entry_desc = "0.7x缩仓" if m15_entry_multiplier < 1.0 else "正常仓位"
                logger.info(f"[{self.name}] {symbol} 15M 评分确认通过 | score={m15_status.get('score', 0)}/4 | 入场方式: {entry_desc}")
                self.monitor.pending_m15_confirmations.pop(pending_key, None)
            elif self.monitor._is_duplicate_signal(symbol, algo_side, data, signal_text, account_name=self.name):
                self.monitor.pending_m15_confirmations.pop(pending_key, None)
                logger.info(f"[{self.name}] {symbol} 本轮未触发AI | LONG {long_score:.1f} / SHORT {short_score:.1f} / 阈值 {threshold:.1f} | 原因: 重复信号冷却中。")
                self.monitor._record_open_rejection(self.name, symbol, '重复信号冷却中')
                return
            if bool(getattr(config, "REQUIRE_15M_CONFIRMATION", True)) and not close_confirmation_passed and not m15_confirmation_passed:
                if skip_m15_confirmation_for_strong_signal:
                    data['pullback_ok'] = data.get('pullback_ok')
                    data['m15_score'] = data.get('m15_score', 0)
                    data['m15_entry_mode'] = 'skipped_strong_signal'
                    data['m15_price_ok'] = data.get('m15_price_ok', False)
                    data['m15_hold_ok'] = data.get('m15_hold_ok', False)
                    logger.info(f"[{self.name}] {symbol} 15M 确认已放宽，当前分数 {current_score:.1f} >= 阈值 {threshold:.1f} + 缓冲 {m15_strong_signal_buffer:.1f}，直接进入AI审核。")
                else:
                    m15_status = self._m15_confirmation_status(data, algo_side)
                    data['pullback_ok'] = bool(m15_status.get('pullback_ok'))
                    data['m15_score'] = int(m15_status.get('score', 0) or 0)
                    data['m15_entry_mode'] = m15_status.get('entry_mode', 'wait')
                    data['m15_price_ok'] = bool(m15_status.get('price_ok'))
                    data['m15_hold_ok'] = bool(m15_status.get('hold_ok'))
                    if not m15_status.get('passed'):
                        expire_bars = max(1, int(getattr(config, "M15_CONFIRM_EXPIRE_BARS", 5)))
                        if bar_open_15m_ms > 0:
                            self.monitor.pending_m15_confirmations[pending_key] = {
                                "algo_side": algo_side,
                                "open_side": algo_side,
                                "score_at_signal": current_score,
                                "bar_open_ms": bar_open_15m_ms,
                                "expire_at_ms": bar_open_15m_ms + expire_bars * 15 * 60 * 1000,
                                "origin_1h_bar_ms": bar_open_1h_ms,
                            }
                        logger.info(f"[{self.name}] {symbol} 15M 评分未达标，已暂存信号等待 | score={m15_status.get('score', 0)}/4 | price_vs_ema20={'OK' if m15_status.get('price_ok') else 'NO'} | macd={'OK' if m15_status.get('macd_ok') else 'NO'} | cci_closed={'OK' if m15_status.get('cci_ok') else 'NO'} | vwap={'OK' if m15_status.get('vwap_ok') else ('MISS' if not m15_status.get('vwap_ready') else 'NO')} | hold2={'OK' if m15_status.get('hold_ok') else 'NO'} | pullback={'OK' if m15_status.get('pullback_ok') else 'NO'}")
                        self.monitor._record_open_rejection(self.name, symbol, '15M评分未达标待确认')
                        return
                    m15_confirmation_passed = True
                    m15_entry_multiplier = float(m15_status.get('entry_multiplier') or 1.0)
                    entry_desc = "0.7x缩仓" if m15_entry_multiplier < 1.0 else "正常仓位"
                    logger.info(f"[{self.name}] {symbol} 15M 评分确认通过 | score={m15_status.get('score', 0)}/4 | 入场方式: {entry_desc}")
            current_mode = str(getattr(self, 'prompt_mode', 'aggressive')).lower()
            is_aggressive = current_mode == 'aggressive'
            if is_aggressive:
                conflict_override_active = self.should_override_conflict(algo_side, data)
                conflict_override_active = self.monitor._maybe_enable_relaxed_conflict_ai_review(
                    conflict_override_active,
                    skip_m15_confirmation_for_strong_signal,
                )
                if conflict_override_active:
                    conflict_frames = "/".join(conflict_override_active.get("conflict_frames", [])) or "HTF"
                    score_items = " | ".join(conflict_override_active.get("score_items") or ["no_bonus"])
                    if not conflict_override_active.get("passed"):
                        logger.info(f"[{self.name}] {symbol} conflict exception rejected | frames: {conflict_frames} | score: {conflict_override_active.get('score', 0)} | details: {score_items}")
                        self.monitor._record_open_rejection(self.name, symbol, 'conflict exception rejected')
                        return
                    logger.info(f"[{self.name}] {symbol} conflict exception active | frames: {conflict_frames} | score: {conflict_override_active.get('score', 0)} | RR: {conflict_override_active.get('rr', 0.0):.2f} | 4H CCI: {conflict_override_active.get('cci_4h', 0.0):.2f} (prev {conflict_override_active.get('prev_cci_4h', 0.0):.2f}) | pattern: {conflict_override_active.get('pattern', 'none')} | scaler: {conflict_override_active.get('scaler', 0.0):.2f}x | details: {score_items}")
                    if conflict_override_active.get("pass_mode") == "micro_with_ai" and not self.enable_ai:
                        logger.info(f"[{self.name}] {symbol} conflict micro entry requires AI confidence >= {conflict_override_active.get('required_ai_confidence', 70.0):.0f}, but AI is disabled. Skip opening.")
                        return
            open_side, final_decision_desc = algo_side, "算法顺势开仓"
            net_rr = self.monitor._initial_open_net_rr(data)
            in_symbol_reentry_cooldown = not self._symbol_reentry_cooldown_allows_override(symbol, now_ts=time.time())
            base_leverage = float(spec.get('leverage', 1.0) or 1.0)
            ai_leverage_factor = float(data.get('ai_leverage_factor', 1.0) or 1.0)
            final_leverage = self._resolve_final_leverage(symbol, base_leverage, ai_leverage_factor)
            fallback_sl = data.get('defense_low', current_price) if open_side == "LONG" else data.get('defense_high', current_price)
            data['ai_entry_scaler'] = 1.0
            data['ai_entry_label'] = ''
            data['ai_leverage_factor'] = ai_leverage_factor
            new_trade_id = f"REAL_{symbol}_{int(time.time())}"

            # ==========================================
            # ==========================================
            if self.enable_ai and not close_confirmation_passed:
                p_mode = getattr(self, 'prompt_mode', 'aggressive')
                if not ai_decision:
                    ai_decision = self.monitor._evaluate_ai_with_cache(symbol, algo_side, data, signal_text, prompt_mode=p_mode)

                if ai_decision:
                    override_meta = self._apply_low_score_ai_override(data, ai_decision, threshold)
                    if override_meta and str(override_meta.get('action', algo_side)).upper() == algo_side:
                        current_score = float(data.get('long_score', 0.0) if algo_side == "LONG" else data.get('short_score', 0.0))
                        logger.info(f"[{self.name}] {symbol} AI score override confirmed | action={algo_side} | pattern={override_meta.get('pattern', 'none')} | score {override_meta.get('original_score', 0):.1f}->{override_meta.get('boosted_score', 0):.1f}")

                    ai_action = str(ai_decision.get('action', 'PASS')).strip().upper()
                    confidence = float(ai_decision.get('confidence', 0))
                    ai_reason = str(ai_decision.get('reason', '')).strip()
                    same_side_threshold = float(symbol_params.get('same_side_ai_confidence', 65 if str(p_mode).lower() == 'aggressive' else 60))
                    reverse_threshold = float(symbol_params.get('reverse_ai_confidence', 75 if str(p_mode).lower() == 'aggressive' else 70))
                    conflict_ai_gate = float(conflict_override_active.get('required_ai_confidence', 0.0)) if conflict_override_active and open_side == algo_side else 0.0
                    if conflict_ai_gate > 0:
                        same_side_threshold = max(same_side_threshold, conflict_ai_gate)

                    ai_entry_scaler, ai_caution_mode, ai_entry_label = self._derive_ai_entry_scaler(
                        ai_action, algo_side, confidence, ai_reason
                    )
                    ai_leverage_factor = self._normalize_ai_leverage_factor(
                        p_mode, ai_decision.get('leverage_factor', 1.0)
                    )
                    data['ai_entry_scaler'] = ai_entry_scaler
                    data['ai_entry_label'] = ai_entry_label
                    data['ai_action'] = ai_action
                    data['ai_confidence'] = confidence
                    data['ai_leverage_factor'] = ai_leverage_factor
                    if in_symbol_reentry_cooldown and not self._symbol_reentry_cooldown_allows_override(
                        symbol,
                        ai_confidence=confidence,
                        net_rr=net_rr,
                        m15_confirmation_passed=m15_confirmation_passed,
                    ):
                        logger.info(f"[{self.name}] {symbol} 刚主动平仓后仍在重入冷却中，需 AI>=75、netRR>=1.5 且 15M 确认通过才允许重开。")
                        self.monitor._record_open_rejection(self.name, symbol, '同币种重入冷却中')
                        return
                    final_leverage = self._resolve_final_leverage(symbol, base_leverage, ai_leverage_factor)

                    wait_for_close = bool(ai_decision.get('wait_for_close', False)) and bool(getattr(config, 'REQUIRE_CLOSE_CONFIRMATION', False))

                    if ai_action == "PASS":
                        logger.info(f"[{self.name}] {symbol} AI 返回 PASS，本轮不执行开仓。")
                        self.monitor._record_open_rejection(self.name, symbol, 'AI PASS')
                        open_side = None
                    elif ai_action == algo_side:
                        if confidence >= same_side_threshold and ai_entry_scaler > 0:
                            open_side = algo_side
                            final_decision_desc = f"{ai_entry_label} (置信度:{confidence})"
                            if wait_for_close:
                                expire_bars = max(1, int(getattr(config, 'CLOSE_CONFIRMATION_EXPIRE_BARS', 2)))
                                self.monitor.pending_close_confirmations[pending_key] = {
                                    'algo_side': algo_side,
                                    'open_side': algo_side,
                                    'score_at_signal': current_score,
                                    'bar_open_ms': bar_open_1h_ms,
                                    'expire_at_ms': bar_open_1h_ms + expire_bars * 60 * 60 * 1000,
                                    'ai_decision': ai_decision,
                                    'ai_caution_mode': ai_caution_mode,
                                    'ai_entry_scaler': data.get('ai_entry_scaler', 1.0),
                                    'ai_entry_label': data.get('ai_entry_label', ''),
                                    'ai_leverage_factor': data.get('ai_leverage_factor', 1.0),
                                }
                                logger.info(f"[{self.name}] {symbol} AI 要求等待 1H 收盘确认，已暂存信号。")
                                return
                        else:
                            logger.info(f"[{self.name}] {symbol} AI 顺势置信度 {confidence} 不足 {same_side_threshold}，或仅建议 PASS/拒绝，本轮跳过开仓。")
                            self.monitor._record_open_rejection(self.name, symbol, 'AI顺势置信度不足')
                            open_side = None
                    elif ai_action in ["LONG", "SHORT"]:
                        if confidence >= reverse_threshold:
                            open_side = ai_action
                            reverse_scaler, reverse_caution, reverse_label = self._derive_ai_entry_scaler(ai_action, ai_action, confidence, ai_reason)
                            ai_caution_mode = reverse_caution
                            data['ai_entry_scaler'] = min(reverse_scaler or 0.5, 0.5)
                            data['ai_entry_label'] = reverse_label or 'AI逆转通过'
                            data['ai_leverage_factor'] = ai_leverage_factor
                            final_decision_desc = f"AI 强制逆转为 {ai_action} (置信度:{confidence})"
                            logger.warning(f"[{self.name}] {symbol} AI 强制逆转，算法方向 {algo_side}，AI 改为 {ai_action}，原因: {ai_reason}")
                            if wait_for_close:
                                expire_bars = max(1, int(getattr(config, 'CLOSE_CONFIRMATION_EXPIRE_BARS', 2)))
                                pending_score = float(data.get('long_score', 0.0) if ai_action == "LONG" else data.get('short_score', 0.0))
                                self.monitor.pending_close_confirmations[pending_key] = {
                                    'algo_side': algo_side,
                                    'open_side': ai_action,
                                    'score_at_signal': pending_score,
                                    'bar_open_ms': bar_open_1h_ms,
                                    'expire_at_ms': bar_open_1h_ms + expire_bars * 60 * 60 * 1000,
                                    'ai_decision': ai_decision,
                                    'ai_caution_mode': ai_caution_mode,
                                    'ai_entry_scaler': data.get('ai_entry_scaler', 1.0),
                                    'ai_entry_label': data.get('ai_entry_label', ''),
                                    'ai_leverage_factor': data.get('ai_leverage_factor', 1.0),
                                }
                                logger.info(f"[{self.name}] {symbol} AI 逆转后要求等待 1H 收盘确认，已暂存信号。")
                                return
                        else:
                            logger.info(f"[{self.name}] {symbol} AI 想逆转为 {ai_action}，但置信度 {confidence} 不足 {reverse_threshold}，按 PASS 处理。")
                            open_side = None
                else:
                    logger.info(f"[{self.name}] {symbol} AI 未返回有效结果，本轮降级跳过开仓。")
                    self.monitor._record_open_rejection(self.name, symbol, 'AI未返回有效结果')
                    open_side = None
            elif close_confirmation_passed:
                ai_leverage_factor = float(data.get('ai_leverage_factor', 1.0) or 1.0)
                if in_symbol_reentry_cooldown and not self._symbol_reentry_cooldown_allows_override(
                    symbol,
                    ai_confidence=float((ai_decision or {}).get('confidence', 0.0) or 0.0),
                    net_rr=net_rr,
                    m15_confirmation_passed=m15_confirmation_passed,
                ):
                    logger.info(f"[{self.name}] {symbol} 刚主动平仓后仍在重入冷却中，需 AI>=75、netRR>=1.5 且 15M 确认通过才允许重开。")
                    self.monitor._record_open_rejection(self.name, symbol, '同币种重入冷却中')
                    return
                final_leverage = self._resolve_final_leverage(symbol, base_leverage, ai_leverage_factor)
                final_decision_desc = data.get('ai_entry_label') or "收盘确认通过"
                if ai_caution_mode:
                    final_decision_desc += " (谨慎缩仓)"
            # ==========================================
            # ==========================================
            # ==========================================
            if open_side:
                sl_price = data.get('defense_low', current_price) * 0.995 if open_side == "LONG" else data.get('defense_high', current_price) * 1.005
                base_scaler = data.get('suggested_scaler', 1.0) if getattr(config, 'ENABLE_QUANT_SCORING', False) else 1.0
                data['suggested_scaler'] = float(base_scaler or 1.0)
                data['ai_entry_scaler'] = float(data.get('ai_entry_scaler', 1.0) or 1.0)
                data['m15_entry_multiplier'] = float(m15_entry_multiplier or 1.0)
                data['ai_caution_mode'] = bool(ai_caution_mode)
                scaler = _compute_dynamic_position_multiplier(
                    self,
                    symbol,
                    data,
                    spec,
                    net_rr=net_rr,
                    ai_confidence=float(ai_decision.get('confidence', 0) if ai_decision else 0),
                )
                if conflict_override_active and open_side == conflict_override_active.get('direction'):
                    conflict_scaler = float(conflict_override_active.get('scaler', 1.0) or 1.0)
                    scaler = float(scaler) * conflict_scaler
                    if conflict_override_active.get('pass_mode') == 'half_size':
                        scaler = max(float(scaler), 0.5)
                    elif conflict_override_active.get('pass_mode') == 'micro_with_ai':
                        scaler = max(float(scaler), 0.2)
                    final_decision_desc += f" (conflict_exception {conflict_scaler:.2f}x)"
                if data.get('ai_entry_label') and data.get('ai_entry_label') not in final_decision_desc:
                    final_decision_desc = f"{final_decision_desc} [{data.get('ai_entry_label')}]"
                if m15_entry_multiplier < 1.0:
                    final_decision_desc += f" (15M缩仓{m15_entry_multiplier:.2f}x)"
                allow_micro_entry = bool(
                    (conflict_override_active and conflict_override_active.get('pass_mode') == 'micro_with_ai')
                    or float(data.get('ai_entry_scaler', 1.0) or 1.0) <= 0.3
                )
                if getattr(config, 'ENABLE_QUANT_SCORING', False) and scaler <= 0.2 and not ai_caution_mode and not allow_micro_entry:
                    return

                risk_space = abs(current_price - sl_price)
                if risk_space > 0:
                    closest_obstacle = self._rr_obstacle_levels(open_side, data, current_price)

                    if closest_obstacle:
                        if symbol == 'BTCUSDT':
                            near_rr_preview, extended_rr_preview = self._compute_btc_near_extended_rr_preview(open_side, data, current_price)
                            rr_preview = near_rr_preview or extended_rr_preview
                            if near_rr_preview:
                                data['btc_near_gross_rr'] = near_rr_preview.get('gross_rr')
                                data['btc_near_net_rr'] = near_rr_preview.get('net_rr')
                                data['btc_near_obstacle'] = near_rr_preview.get('closest_obstacle')
                            if extended_rr_preview:
                                data['btc_extended_gross_rr'] = extended_rr_preview.get('gross_rr')
                                data['btc_extended_net_rr'] = extended_rr_preview.get('net_rr')
                                data['btc_extended_obstacle'] = extended_rr_preview.get('closest_obstacle')
                        else:
                            rr_preview = _compute_realistic_rr_preview(open_side, data, current_price, self._rr_obstacle_levels)
                        if rr_preview:
                            realistic_rr = float(rr_preview.get('gross_rr', rr_preview.get('rr', 0.0)) or 0.0)
                            net_rr = float(rr_preview.get('net_rr', realistic_rr) or 0.0)
                            data['realistic_rr'] = realistic_rr
                            data['gross_rr'] = realistic_rr
                            data['net_expected_rr'] = net_rr
                            data['net_rr'] = net_rr
                            data['rr_costs'] = {
                                'fee_bps': rr_preview.get('fee_bps'),
                                'slippage_bps': rr_preview.get('slippage_bps'),
                                'funding_bps': rr_preview.get('funding_bps'),
                                'total_cost_bps': rr_preview.get('total_cost_bps'),
                            }
                        else:
                            realistic_rr = 0.0
                            net_rr = 0.0
                        strategy_plan = _build_strategy_plan(open_side, data, conflict_override_active=conflict_override_active if open_side == algo_side else None)
                        data['market_regime'] = strategy_plan.get('market_regime')
                        data['strategy_bucket'] = strategy_plan.get('strategy_bucket')
                        data['strategy_plan'] = strategy_plan

                        rr_profile = self._dynamic_rr_profile(open_side, data, current_price)
                        min_rr_required = float(strategy_plan.get('min_net_rr', rr_profile.get('min_rr', 1.2)) or 1.2)
                        rr_label = strategy_plan.get('rr_label', rr_profile.get('label', 'trend_continuation'))
                        quant_score = float(data.get('long_score', 0.0) if open_side == "LONG" else data.get('short_score', 0.0))
                        ai_conf = float(ai_decision.get('confidence', 0) if ai_decision else 0)
                        min_net_rr_floor = float(symbol_params.get('min_net_rr_floor', getattr(config, 'MIN_NET_RR_HARD_FLOOR', 0.7)) or 0.7)

                        rr_scaler = 1.0
                        if net_rr < min_net_rr_floor or net_rr <= 0:
                            rr_scaler = 0.0
                        elif net_rr < min_rr_required:
                            rr_scaler = 0.5

                        if (
                            symbol != 'BTCUSDT'
                            and rr_scaler == 0.0
                            and net_rr < 0.5
                            and net_rr > 0
                            and (quant_score >= 7.0 or ai_conf >= 80)
                        ):
                            rr_scaler = 0.5
                            logger.info(f"[{self.name}] {symbol} net RR below 0.5 but high score override | score={quant_score:.1f} | confidence={ai_conf:.1f} | rr_scaler locked at 0.5")

                        if symbol == 'BTCUSDT':
                            near_net_rr = float(data.get('btc_near_net_rr') or net_rr or 0.0)
                            extended_net_rr = float(data.get('btc_extended_net_rr') or 0.0)
                            btc_rr_mode = 'BTC近端RR过低'
                            btc_rr_cap = 0.0

                            if near_net_rr < 0.5:
                                rr_scaler = 0.0
                            elif near_net_rr < 0.8:
                                if extended_net_rr >= 1.8 and ai_conf >= 68:
                                    rr_scaler = min(max(rr_scaler, 0.3), 0.2)
                                    btc_rr_mode = 'BTC扩展空间极小试仓'
                                    btc_rr_cap = 0.2
                                else:
                                    rr_scaler = 0.0
                                    btc_rr_mode = 'BTC近端偏低且扩展确认不足'
                            elif near_net_rr < 1.0:
                                if extended_net_rr >= 1.4 and ai_conf >= 60:
                                    rr_scaler = min(max(rr_scaler, 0.5), 0.35)
                                    btc_rr_mode = 'BTC扩展空间小仓试仓'
                                    btc_rr_cap = 0.35
                                else:
                                    rr_scaler = 0.0
                                    btc_rr_mode = 'BTC近端不足且扩展空间不够'
                            elif ai_conf >= 65:
                                btc_rr_mode = 'BTC近端RR达标谨慎评估'
                                btc_rr_cap = 0.7
                                rr_scaler = min(rr_scaler, btc_rr_cap)
                            else:
                                rr_scaler = 0.0
                                btc_rr_mode = 'BTC近端RR达标但AI不足'

                            data['btc_rr_mode'] = btc_rr_mode
                            data['btc_rr_cap'] = btc_rr_cap
                            logger.info(
                                f"[{self.name}] BTC RR layered gate | near={near_net_rr:.2f} | extended={extended_net_rr:.2f} | "
                                f"ai_conf={ai_conf:.1f} | mode={btc_rr_mode} | cap={btc_rr_cap:.2f} | rr_scaler={rr_scaler:.2f}"
                            )

                        if rr_scaler == 0.0:
                            logger.warning(f"[{self.name}] {symbol} strategy rejected | bucket={data.get('strategy_bucket')} | regime={data.get('market_regime')} | gross_rr={realistic_rr:.2f} | net_rr={net_rr:.2f} < floor={min_net_rr_floor:.2f} | obstacle={closest_obstacle}")
                            self.monitor._record_open_rejection(self.name, symbol, '净RR不足')
                            return

                        max_strategy_scaler = float(strategy_plan.get('max_scaler', 1.0) or 1.0)
                        penalty_payload = self.monitor.get_dashboard_overview_payload() if hasattr(self.monitor, 'get_dashboard_overview_payload') else {}
                        penalties = _compute_strategy_penalties(penalty_payload)
                        bucket_penalty = float(penalties.get('bucket', {}).get(data.get('strategy_bucket', ''), 1.0) or 1.0)
                        regime_bucket_key = f"{data.get('market_regime', '')}|{data.get('strategy_bucket', '')}"
                        regime_bucket_penalty = float(penalties.get('regime_bucket', {}).get(regime_bucket_key, 1.0) or 1.0)
                        rr_scaler = min(rr_scaler, max_strategy_scaler) * bucket_penalty * regime_bucket_penalty
                        if rr_scaler < 1.0:
                            scaler *= rr_scaler
                            final_decision_desc += f" ({data.get('strategy_bucket')} {rr_scaler:.2f}x)"
                            logger.info(f"[{self.name}] {symbol} strategy sizing active | bucket={data.get('strategy_bucket')} | regime={data.get('market_regime')} | gross_rr={realistic_rr:.2f} | net_rr={net_rr:.2f} | min_rr={min_rr_required:.1f} | rr_scaler={rr_scaler:.2f}")
                        else:
                            logger.info(f"[{self.name}] {symbol} strategy sizing active | bucket={data.get('strategy_bucket')} | regime={data.get('market_regime')} | gross_rr={realistic_rr:.2f} | net_rr={net_rr:.2f} | scaler=1.00x")

                base_capital = max(float(self.get_base_capital() or 0.0), 1e-9)
                portfolio_snapshot = _compute_portfolio_exposure_snapshot(getattr(self.monitor, 'all_trade_meta', None) or self.real_trade_meta)
                total_notional_exposure = sum(float(meta.get('notional', 0) or 0.0) for meta in self.real_trade_meta.values())
                symbol_notional_exposure = sum(float(meta.get('notional', 0) or 0.0) for sym, meta in self.real_trade_meta.items() if sym == symbol)
                direction_notional_exposure = portfolio_snapshot.get('direction', {}).get(open_side, 0.0)
                cluster_name = _get_symbol_risk_cluster(symbol)
                cluster_notional_exposure = portfolio_snapshot.get('cluster', {}).get(cluster_name, 0.0)
                max_account_exposure_ratio = float(getattr(config, 'MAX_ACCOUNT_EXPOSURE_RATIO', 0.8) or 0.8)
                max_symbol_exposure_ratio = float(getattr(config, 'MAX_SYMBOL_EXPOSURE_RATIO', 0.35) or 0.35)
                max_direction_exposure_ratio = float(getattr(config, 'MAX_DIRECTION_EXPOSURE_RATIO', 0.65) or 0.65)
                max_cluster_exposure_ratio = float(getattr(config, 'MAX_CLUSTER_EXPOSURE_RATIO', 0.55) or 0.55)
                current_account_exposure_ratio = total_notional_exposure / base_capital
                current_symbol_exposure_ratio = symbol_notional_exposure / base_capital
                current_direction_exposure_ratio = direction_notional_exposure / base_capital
                current_cluster_exposure_ratio = cluster_notional_exposure / base_capital
                available_account_ratio = max(max_account_exposure_ratio - current_account_exposure_ratio, 0.0)
                available_symbol_ratio = max(max_symbol_exposure_ratio - current_symbol_exposure_ratio, 0.0)
                available_direction_ratio = max(max_direction_exposure_ratio - current_direction_exposure_ratio, 0.0)
                available_cluster_ratio = max(max_cluster_exposure_ratio - current_cluster_exposure_ratio, 0.0)
                if available_account_ratio <= 0 or available_symbol_ratio <= 0 or available_direction_ratio <= 0 or available_cluster_ratio <= 0:
                    logger.warning(f"[{self.name}] 账户/单币/方向/组合敞口已满，拒绝开仓 {symbol}。")
                    self.monitor._record_open_rejection(self.name, symbol, '账户敞口达到上限')
                    return
                dynamic_margin_ratio = min(
                    spec['margin_ratio'] * scaler,
                    0.10,
                    available_account_ratio / max(final_leverage, 1e-9),
                    available_symbol_ratio / max(final_leverage, 1e-9),
                    available_direction_ratio / max(final_leverage, 1e-9),
                    available_cluster_ratio / max(final_leverage, 1e-9),
                )
                target_margin_used = self.get_base_capital() * dynamic_margin_ratio
                target_notional = target_margin_used * final_leverage
                safe_qty = self.engine.contracts_from_notional(symbol, target_notional, current_price)
                actual_notional = self.engine.estimate_notional(symbol, safe_qty, current_price)
                margin_used = actual_notional / final_leverage if final_leverage > 0 else 0.0
                min_effective_notional = float(getattr(config, 'MIN_OPEN_NOTIONAL_USDT', 30.0) or 30.0)
                min_margin_ratio = float(getattr(config, 'MIN_OPEN_MARGIN_RATIO', 0.01) or 0.01)
                min_floor_net_rr = float(getattr(config, 'MIN_OPEN_FLOOR_NET_RR', 1.2) or 1.2)
                min_target_notional = base_capital * min_margin_ratio * final_leverage
                strategy_bucket = str(data.get('strategy_bucket', '') or '')
                ai_entry_label = str(data.get('ai_entry_label', '') or '')
                ai_action_text = str(data.get('ai_action', '') or '').upper()
                try:
                    ai_confidence_value = float(data.get('ai_confidence', 0.0) or 0.0)
                except (TypeError, ValueError):
                    ai_confidence_value = 0.0
                ema_status_4h = str(data.get('ema_status_4h', '') or '')
                ema_status_1d = str(data.get('ema_status_1d', '') or '')
                supportive_trend = (
                    open_side == 'LONG' and ('多头' in ema_status_4h or '多头' in ema_status_1d)
                ) or (
                    open_side == 'SHORT' and ('空头' in ema_status_4h or '空头' in ema_status_1d)
                )
                quality_allows_floor = bool(
                    (
                        net_rr >= min_floor_net_rr
                        and strategy_bucket == 'trend_continuation'
                        and (
                            m15_confirmation_passed
                            or data.get('m15_entry_mode') == 'skipped_strong_signal'
                        )
                        and '谨慎' not in ai_entry_label
                    )
                    or (
                        symbol in {'ETHUSDT', 'SOLUSDT', 'XRPUSDT'}
                        and ai_action_text == open_side
                        and ai_confidence_value >= 60
                        and net_rr >= 0.9
                        and supportive_trend
                    )
                )
                if actual_notional < min_effective_notional and quality_allows_floor:
                    target_notional = max(target_notional, min_effective_notional, min_target_notional)
                    safe_qty = self.engine.contracts_from_notional(symbol, target_notional, current_price)
                    actual_notional = self.engine.estimate_notional(symbol, safe_qty, current_price)
                    margin_used = actual_notional / final_leverage if final_leverage > 0 else 0.0
                min_qty = float(self.engine.get_symbol_info(symbol).get('minQty', 0) or 0)
                if safe_qty <= 0 or actual_notional < min_effective_notional or (min_qty > 0 and safe_qty < min_qty):
                    logger.info(
                        f"[{self.name}] {symbol} 跳过开仓：有效下单规模过小 | qty={safe_qty} | actual_notional={actual_notional:.2f} U | min_notional={min_effective_notional:.2f} U | min_qty={min_qty}"
                    )
                    self.monitor._record_open_rejection(self.name, symbol, '下单规模过小')
                    return
                safe_sl = float(self.engine.format_price(symbol, sl_price if sl_price > 0 else fallback_sl))

                risk_space = abs(current_price - safe_sl)
                safe_tp = float(self.engine.format_price(symbol, current_price + (risk_space * getattr(config, 'TAKE_PROFIT_RR', 2.0)) if open_side == "LONG" else current_price - (risk_space * getattr(config, 'TAKE_PROFIT_RR', 2.0))))

                if safe_qty > 0 and actual_notional > 0:
                    logger.info(f"[{self.name}] 开仓指令: {symbol} {open_side} | 合约张数: {safe_qty} | 目标名义金额: {target_notional:.2f} U | 实际名义金额: {actual_notional:.2f} U | 杠杆: base={base_leverage:.0f}x | ai_factor={ai_leverage_factor:.2f} | final={final_leverage}x")

                    if self.engine.execute_open_with_sltp(symbol, open_side, safe_qty, safe_sl, safe_tp, final_leverage):
                        self.real_trade_meta[symbol] = {
                            'trade_id': new_trade_id, 'side': open_side, 'entry': current_price, 'sl': safe_sl, 'tp': safe_tp,
                            'notional': actual_notional, 'tp_phase': 0, 'open_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'qty': safe_qty,
                            'market_regime': data.get('market_regime', ''), 'strategy_bucket': data.get('strategy_bucket', '')
                        }
                        ToolKit.save_active_trades(self.real_trade_meta, account_name=self.name)

                        self.real_positions[symbol] = {
                            'side': open_side, 'qty': safe_qty, 'entry': current_price
                        }

                        if not data.get('card_sent'):
                            if self.enable_ai and 'ai_decision' in locals() and ai_decision:
                                data['ai_action'] = ai_decision.get('action', 'PASS')
                                data['ai_reason'] = ai_decision.get('reason', '未知')
                            else:
                                data['ai_action'] = '未启用'
                                data['ai_reason'] = '纯算法执行'
                            data['is_trade_triggered'] = True
                            data['trade_side'] = open_side
                            data['trade_tp'] = safe_tp
                            data['trade_sl'] = safe_sl
                            data['trade_margin'] = margin_used
                            data['current_balance'] = self.get_base_capital()
                            data['actual_leverage'] = final_leverage
                            data['net_rr'] = net_rr
                            data['net_expected_rr'] = net_rr
                            data['m15_entry_multiplier'] = float(data.get('m15_entry_multiplier') or m15_entry_multiplier or 1.0)

                            card_msg = ToolKit.format_dingtalk_card(data, mode=self.mode, enable_ai=self.enable_ai)
                            ToolKit.send_dingtalk_msg(card_msg, title=f"{symbol.replace('USDT', '')} 交易决策分析")
                            data['card_sent'] = True

                        if not self.enable_ai:
                            ai_style_tag = "AI已关闭(纯算法)"
                        else:
                            current_mode = getattr(self, 'prompt_mode', 'aggressive')
                            ai_style_tag = "激进(抄底/逃顶)" if current_mode == 'aggressive' else "保守(右侧趋势)"
                        market_regime_label = ToolKit._display_market_regime(data.get('market_regime'))
                        strategy_bucket_label = ToolKit._display_strategy_bucket(data.get('strategy_bucket'))
                        ai_entry_label = str(data.get('ai_entry_label') or final_decision_desc or '未标记')
                        rr_quality = ToolKit._rr_quality(net_rr, symbol=symbol)
                        notional_used = margin_used * final_leverage
                        position_reasons = []
                        if '谨慎' in ai_entry_label or '边缘' in ai_entry_label:
                            position_reasons.append('AI谨慎')
                        if float(net_rr or 0.0) < 1.0:
                            position_reasons.append('RR偏低')
                        if str(data.get('market_regime') or '') in {'high_vol_range', 'event_volatility', 'low_liquidity'}:
                            position_reasons.append('波动/流动性风险较高')
                        if float(data.get('m15_entry_multiplier') or 1.0) < 1.0:
                            position_reasons.append('15M确认缩仓')
                        position_reason = ' + '.join(position_reasons) if position_reasons else '信号质量与风控约束正常'
                        open_msg = (
                            f"### [{self.name}] 交易执行: {symbol} {open_side}\n"
                            f"- **交易类型**: {market_regime_label} / {strategy_bucket_label}\n"
                            f"- **开仓性质**: {ai_entry_label}\n"
                            f"- **AI 模式**: {ai_style_tag}\n"
                            f"- **开仓动作**: {final_decision_desc}\n"
                            f"- **RR评价**: 净RR {net_rr:.2f}，{rr_quality}\n"
                            f"- **仓位原因**: {position_reason}\n\n"
                            f"**仓位与杠杆**\n"
                            f"- **分配杠杆**: {final_leverage}x (base {base_leverage:.0f}x / AI {ai_leverage_factor:.2f})\n"
                            f"- **占用本金**: {margin_used:.2f} U\n"
                            f"- **名义仓位**: {notional_used:.2f} U\n"
                            f"- **最新余额**: {self.get_base_capital():.2f} U\n\n"
                            f"**风控价位**\n"
                            f"- **入场价格**: {current_price:.5f}\n"
                            f"- **预计防守**: {safe_sl}\n"
                            f"- **预计止盈**: {safe_tp}"
                        )
                        ToolKit.send_dingtalk_msg(open_msg, title=f"[{self.name}] {symbol.replace('USDT','')} 开仓")
                else:
                    logger.info(f"[{self.name}] {symbol} 计算下单数量为 0，跳过开仓。")

    def panic_close_all(self):
        logger.warning(f"\n[{self.name}] 收到紧急指令：开始一键市价全平。")

        if self.mode == 'SIMULATE':
            for sym, pos in list(self.open_positions.items()):
                spec = self.monitor.get_symbol_spec(sym)
                current_price = pos['entry']
                pnl_pct = (current_price - pos['entry']) / pos['entry'] * (1 if pos['side'] == "LONG" else -1)
                net_pnl = (pos['notional'] * pnl_pct) - (pos['notional'] * 0.0004)
                self._save_to_trade_history(True, sym, pos.get('trade_id', '未知'), pos.get('open_time', '未知'), pos['side'], spec['leverage'], pos['entry'], current_price, "紧急一键全平", net_pnl, self.get_base_capital())
            self.open_positions.clear()
        else:
            for sym, pos in list(self.real_positions.items()):
                spec = self.monitor.get_symbol_spec(sym)
                try:
                    self.engine.cancel_all_orders(sym)
                    close_order_id = self.engine.execute_market_close(sym, pos['side'], pos['qty'])
                    current_price = pos['entry']
                    realized_pnl = get_realized_pnl_record(self.engine, sym, pos['qty'], pos['entry'], current_price, pos['side'], order_id=close_order_id)
                    net_pnl = float(realized_pnl['net_pnl'])
                    if realized_pnl.get('avg_price'):
                        current_price = float(realized_pnl['avg_price'])

                    meta = self.real_trade_meta.get(sym, {})
                    self._save_to_trade_history(False, sym, meta.get('trade_id', '未知'), meta.get('open_time', '未知'), pos['side'], spec['leverage'], pos['entry'], current_price, "紧急全平", net_pnl, self.get_base_capital())

                    if sym in self.real_trade_meta:
                        del self.real_trade_meta[sym]
                except Exception as e:
                    logger.error(f"[{self.name}] 强平失败 {sym}: {e}")
            ToolKit.save_active_trades(self.real_trade_meta, account_name=self.name)

        ToolKit.send_dingtalk_msg(f"### [{self.name}] 紧急避险已启动\n该账户所有持仓已被强制市价清空！", title=f"[{self.name}] 紧急全平通知")

# ========================================== 
# ==========================================
class OKXAlphaMonitor:
    def _runtime_state_file(self):
        return 'runtime_state.json'

    def _apply_runtime_state(self, state, accounts=None):
        accounts = accounts if accounts is not None else getattr(self, 'accounts', [])
        self.recent_open_rejections = deque(maxlen=2000)
        for item in list((state or {}).get('recent_open_rejections', []) or []):
            ts_value = item.get('time')
            parsed_time = None
            if ts_value:
                try:
                    parsed_time = datetime.strptime(str(ts_value), '%Y-%m-%d %H:%M:%S')
                except Exception:
                    parsed_time = None
            self.recent_open_rejections.append({
                'time': parsed_time,
                'account_name': item.get('account_name'),
                'symbol': item.get('symbol'),
                'reason': item.get('reason'),
            })
        account_state = (state or {}).get('accounts', {}) or {}
        for acc in accounts or []:
            snapshot = account_state.get(acc.name, {}) or {}
            acc.bucket_cooldowns = dict(snapshot.get('bucket_cooldowns', {}) or {})
            acc.bucket_loss_streaks = dict(snapshot.get('bucket_loss_streaks', {}) or {})
            acc.symbol_reentry_cooldowns = dict(snapshot.get('symbol_reentry_cooldowns', {}) or {})

    def __init__(self, enable_ai=True, enable_ai_close=True):
        self.mode = getattr(config, 'TRADING_MODE', 'SIMULATE').upper()
        self.symbols = config.SYMBOLS
        self.funds_mode = getattr(config, 'FUNDS_MODE', 'SHARED').upper()

        self.enable_ai = enable_ai
        self.enable_ai_close = enable_ai_close
        self.ai_cache = {}
        self.ai_cache_ttl = getattr(config, 'AI_CACHE_TTL', 3600)
        self.position_ai_cache = {}
        self.position_ai_state = {}
        self.position_ai_peak_pnl_pct = {}
        self.position_ai_executed_actions = {}
        self.algo_cache = {}
        self.pending_close_confirmations = {}
        self.pending_m15_confirmations = {}
        self.recent_open_rejections = deque(maxlen=2000)
        self.latest_symbol_snapshots = {}
        self.last_dashboard_refresh = None

        self.loop_count = 0
        self.last_mihomo_restart = 0
        self.fast_scan_until = 0.0

        self.ai_advisor = AITradingAdvisor()
        self.analyzer = MarketAnalyzer(proxies=config.OKX_PROXIES)

        self.accounts = []
        if self.mode in ['TESTNET', 'REAL']:
            for acc_cfg in getattr(config, 'ACCOUNTS', []):
                if acc_cfg.get('api_key'):
                    p_mode = acc_cfg.get('prompt_mode', 'aggressive')

                    self.accounts.append(SubAccount(
                        acc_cfg['name'],
                        acc_cfg['api_key'],
                        acc_cfg['api_secret'],
                        acc_cfg.get('passphrase', ''),
                        self,
                        prompt_mode=p_mode
                    ))
            if not self.accounts:
                logger.error("实盘模式下未在 config.py 中配置任何有效的 ACCOUNTS，系统退出。")
                sys.exit(1)
        else:
            self.accounts.append(SubAccount("模拟体验号", "", "", "", self, prompt_mode="aggressive"))

        runtime_state = ToolKit.load_runtime_state(self._runtime_state_file())
        self._apply_runtime_state(runtime_state, self.accounts)

        signal.signal(signal.SIGTERM, self._graceful_shutdown)
        signal.signal(signal.SIGINT, self._graceful_shutdown)

        logger.info(f"==========================================")
        strategy_name = "量化打分+波动率" if getattr(config, 'ENABLE_QUANT_SCORING', False) else "经典指标流"
        logger.info(f"资管监控主引擎启动 | 模式: {self.mode} | 策略: {strategy_name}")
        logger.info(f"已挂载子账户数量: {len(self.accounts)}")
        logger.info(f"AI 风控状态 | 开仓控制: {'开启' if self.enable_ai else '关闭'} | 平仓反转确认: {'开启' if self.enable_ai_close else '关闭'} | 持仓AI模式: 仅建议，不自动下单")
        logger.info(f"==========================================")

    def _preview_realistic_rr(self, direction, data, current_price):
        obstacle_resolver = self.accounts[0]._rr_obstacle_levels if self.accounts else (lambda *_: None)
        return _compute_realistic_rr_preview(direction, data, current_price, obstacle_resolver)

    @staticmethod
    def _maybe_enable_relaxed_conflict_ai_review(conflict_override, skip_m15_confirmation_for_strong_signal):
        if not conflict_override:
            return conflict_override
        if not skip_m15_confirmation_for_strong_signal:
            return conflict_override
        if int(conflict_override.get('score', 0) or 0) != 1:
            return conflict_override
        if conflict_override.get('passed'):
            return conflict_override

        conflict_override['passed'] = True
        conflict_override['pass_mode'] = 'micro_with_ai'
        conflict_override['scaler'] = 0.2
        conflict_override['required_ai_confidence'] = 75.0
        score_items = list(conflict_override.get('score_items') or [])
        if '15M强信号AI复核+0' not in score_items:
            score_items.append('15M强信号AI复核+0')
        conflict_override['score_items'] = score_items
        return conflict_override

    @staticmethod
    def _initial_open_net_rr(data):
        try:
            return float(data.get('net_expected_rr', data.get('net_rr', 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0

        # ==========================================
    # ==========================================
    def _get_future_price_for_audit(self, symbol, timestamp_ms):
        """辅助函数：获取信号发出后 4 小时内的行情走势"""
        inst_id = f"{symbol.replace('USDT', '')}-USDT-SWAP"
        url = "https://www.okx.com/api/v5/market/history-candles"
        params = {"instId": inst_id, "bar": "1H", "before": timestamp_ms, "limit": 5}
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json().get("data", [])
            if not data or len(data) < 2: return None
            ordered = list(reversed(data))
            return {"entry": float(ordered[0][1]), "last": float(ordered[-1][4])}
        except: 
            return None

    def _load_balance_snapshots(self, account_name):
        filename = f"balance_snapshots_{account_name}.csv"
        if not os.path.exists(filename):
            return []
        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            if '时间' not in df.columns or '余额(U)' not in df.columns:
                return []
            snapshots = []
            for _, row in df.iterrows():
                try:
                    snapshots.append({'ts': str(row['时间']), 'balance': float(row['余额(U)'])})
                except Exception:
                    continue
            return snapshots
        except Exception as e:
            logger.warning(f"加载余额快照失败 [{account_name}]: {e}")
            return []

    def _append_balance_snapshot(self, account_name, balance, now=None):
        now = now or datetime.now()
        filename = f"balance_snapshots_{account_name}.csv"
        snapshot_ts = now.strftime('%Y-%m-%d %H:%M:%S')
        try:
            snapshots = self._load_balance_snapshots(account_name)
            if snapshots:
                last_ts = pd.to_datetime(snapshots[-1]['ts'])
                if (now - last_ts).total_seconds() < 3600:
                    return
            entry = pd.DataFrame([{'时间': snapshot_ts, '余额(U)': round(float(balance), 4)}])
            entry.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.warning(f"写入余额快照失败 [{account_name}]: {e}")

    def _calculate_true_24h_pnl(self, current_balance, snapshots, now=None):
        now = now or datetime.now()
        cutoff = now - timedelta(days=1)
        baseline = None
        for snap in snapshots:
            try:
                snap_ts = pd.to_datetime(snap.get('ts'))
                snap_balance = float(snap.get('balance'))
            except Exception:
                continue
            if snap_ts <= cutoff:
                baseline = snap_balance
            else:
                break
        if baseline is None:
            return None
        return round(float(current_balance) - baseline, 2)

    def _snapshot_now_str(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _update_dashboard_symbol_snapshot(self, symbol, data):
        if not hasattr(self, 'latest_symbol_snapshots'):
            self.latest_symbol_snapshots = {}

        funding_snapshot = data.get('funding_snapshot') or {}
        snapshot = {
            'symbol': symbol,
            'price': float(data.get('price') or 0.0),
            'realtime_price': float(data.get('realtime_price') or data.get('price') or 0.0),
            'trend_state': str(data.get('trend_state') or ''),
            'long_score': float(data.get('long_score') or 0.0),
            'short_score': float(data.get('short_score') or 0.0),
            'realistic_rr': float(data.get('realistic_rr')) if data.get('realistic_rr') is not None else None,
            'funding_sentiment': str(funding_snapshot.get('sentiment') or data.get('funding_sentiment') or '未知'),
            'money_flow': str(data.get('money_flow') or ''),
            'has_position': bool(data.get('has_position')),
            'position_side': data.get('position_side'),
            'latest_ai_action': str(data.get('ai_action') or 'N/A'),
            'latest_ai_confidence': float(data.get('ai_confidence') or 0.0),
            'updated_at': self._snapshot_now_str(),
        }
        self.latest_symbol_snapshots[symbol] = snapshot

    def get_dashboard_symbols_payload(self):
        rows = []
        symbol_order = list(getattr(self, 'symbols', []) or [])
        for symbol in symbol_order:
            snap = dict((getattr(self, 'latest_symbol_snapshots', {}) or {}).get(symbol, {}))
            if not snap:
                continue
            funding_sentiment = snap.get('funding_sentiment')
            if not funding_sentiment:
                funding_sentiment = str((snap.get('funding_snapshot') or {}).get('sentiment') or '未知')
            rows.append({
                'symbol': symbol,
                'latest_price': snap.get('realtime_price') or snap.get('price'),
                'trend_state': snap.get('trend_state'),
                'long_score': snap.get('long_score'),
                'short_score': snap.get('short_score'),
                'realistic_rr': snap.get('realistic_rr'),
                'funding_sentiment': funding_sentiment,
                'money_flow': snap.get('money_flow'),
                'has_position': bool(snap.get('has_position')),
                'position_side': snap.get('position_side'),
                'latest_ai_action': snap.get('latest_ai_action'),
                'latest_ai_confidence': snap.get('latest_ai_confidence'),
                'last_update_time': snap.get('updated_at'),
            })
        return sorted(rows, key=lambda row: symbol_order.index(row['symbol']) if row['symbol'] in symbol_order else 9999)

    def get_dashboard_positions_payload(self):
        rows = []
        for acc in getattr(self, 'accounts', []) or []:
            positions = getattr(acc, 'real_positions', {}) or {}
            meta_map = getattr(acc, 'real_trade_meta', {}) or {}
            for symbol, pos in positions.items():
                meta = meta_map.get(symbol, {})
                qty = float(pos.get('qty') or 0.0)
                entry = float(pos.get('entry') or 0.0)
                current_price = float(pos.get('current_price') or pos.get('mark_price') or entry or 0.0)
                if meta.get('notional'):
                    notional = float(meta.get('notional'))
                elif getattr(acc, 'engine', None):
                    notional = float(acc.engine.estimate_notional(symbol, qty, current_price))
                else:
                    notional = qty * current_price
                unrealized = float(pos.get('unrealized_pnl') or 0.0)
                unrealized_pct = (unrealized / max(notional, 1e-9)) * 100 if notional > 0 else 0.0
                rows.append({
                    'account_name': acc.name,
                    'symbol': symbol,
                    'side': pos.get('side'),
                    'entry_price': entry,
                    'current_price': current_price,
                    'quantity': qty,
                    'notional': round(notional, 2),
                    'leverage': meta.get('leverage'),
                    'stop_loss': meta.get('sl'),
                    'take_profit': meta.get('tp'),
                    'tp_phase': meta.get('tp_phase'),
                    'open_time': meta.get('open_time'),
                    'unrealized_pnl': round(unrealized, 2),
                    'unrealized_pnl_percent': round(unrealized_pct, 2),
                })
        return sorted(rows, key=lambda row: (row['account_name'], row['symbol']))

    def get_dashboard_account_stats_payload(self):
        rows = []
        for acc in getattr(self, 'accounts', []) or []:
            current_balance = float(acc.get_base_capital())
            trade_24h_pnl = 0.0
            total_trade_pnl = 0.0
            open_position_count = len(getattr(acc, 'real_positions', {}) or {})
            history_file = f"trades_history_{acc.name}.csv"
            if os.path.exists(history_file):
                df = pd.read_csv(history_file, encoding='utf-8-sig')
                if '平仓时间' in df.columns and '净盈亏(U)' in df.columns:
                    df = df[df.apply(_is_robot_trade_history_row, axis=1)].copy()
                    df['平仓时间_dt'] = pd.to_datetime(df['平仓时间'])
                    df['净盈亏(U)'] = pd.to_numeric(df['净盈亏(U)'], errors='coerce').fillna(0.0)
                    last_24h = df[df['平仓时间_dt'] > (datetime.now() - pd.Timedelta(days=1))]
                    trade_24h_pnl = float(last_24h['净盈亏(U)'].sum()) if not last_24h.empty else 0.0
                    total_trade_pnl = float(df['净盈亏(U)'].sum()) if not df.empty else 0.0
            exposure_notional = 0.0
            for symbol, meta in (getattr(acc, 'real_trade_meta', {}) or {}).items():
                exposure_notional += float(meta.get('notional') or 0.0)
            exposure_ratio = (exposure_notional / max(current_balance, 1e-9)) if current_balance > 0 else 0.0
            cooldown_active = bool(getattr(acc, 'cooldown_until', 0.0) and time.time() < acc.cooldown_until)
            recent_losses = 0
            for pnl in reversed(list(getattr(acc, 'recent_closed_pnls', []) or [])):
                if float(pnl) < 0:
                    recent_losses += 1
                else:
                    break
            rows.append({
                'account_name': acc.name,
                'current_balance': round(current_balance, 2),
                'trade_24h_pnl': round(trade_24h_pnl, 2),
                'total_trade_pnl': round(total_trade_pnl, 2),
                'open_position_count': open_position_count,
                'exposure_ratio': round(exposure_ratio, 4),
                'cooldown_state': cooldown_active,
                'loss_streak_context': recent_losses,
            })
        return rows

    def _record_open_rejection(self, account_name, symbol, reason):
        self.recent_open_rejections.append({
            'time': datetime.now(),
            'account_name': account_name,
            'symbol': symbol,
            'reason': reason,
            'stage': _normalize_open_rejection_stage(reason),
        })
        try:
            ToolKit.save_runtime_state(self._runtime_state_file(), self, getattr(self, 'accounts', []))
        except Exception as e:
            logger.warning(f"保存未开仓原因状态失败: {e}")

    def get_dashboard_overview_payload(self):
        account_stats = self.get_dashboard_account_stats_payload()
        total_balance = sum(float(row.get('current_balance') or 0.0) for row in account_stats)
        total_trade_24h = sum(float(row.get('trade_24h_pnl') or 0.0) for row in account_stats)
        total_trade_pnl = sum(float(row.get('total_trade_pnl') or 0.0) for row in account_stats)
        open_positions = self.get_dashboard_positions_payload()
        bucket_summary = []
        regime_summary = []
        regime_bucket_matrix = []
        symbol_summary = []
        execution_quality_summary = []
        execution_quality_bucket_summary = []
        execution_quality_regime_summary = []
        shadow_opportunity_summary = []
        shadow_opportunity_bucket_summary = []
        backtest_parameter_recommendations = []
        ai_shadow_quality_summary = []
        ai_shadow_quality_recommendations = []
        open_rejection_summary = _summarize_open_rejections(getattr(self, 'recent_open_rejections', []), since_hours=24)
        open_rejection_stage_summary = _summarize_open_rejection_stages(getattr(self, 'recent_open_rejections', []), since_hours=24)
        calibration_recommendations = []
        log_file = 'ai_decision_logs.csv'
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file, encoding='utf-8-sig')
            except Exception:
                df = pd.read_csv(log_file)
            bucket_summary = _summarize_strategy_buckets(df)
            regime_summary = _summarize_market_regimes(df)
            regime_bucket_matrix = _summarize_regime_bucket_matrix(df)
            symbol_summary = _summarize_symbols(df)
            execution_quality_summary = _summarize_execution_quality(df)
            execution_quality_bucket_summary = _summarize_execution_quality_by_bucket(df)
            execution_quality_regime_summary = _summarize_execution_quality_by_regime(df)
            open_trade_execution_df = _build_open_trade_execution_quality_frame(getattr(self, 'accounts', []))
            if open_trade_execution_df is not None:
                if not execution_quality_summary:
                    execution_quality_summary = _summarize_execution_quality(open_trade_execution_df)
                if not execution_quality_bucket_summary:
                    execution_quality_bucket_summary = _summarize_execution_quality_by_bucket(open_trade_execution_df)
                if not execution_quality_regime_summary:
                    execution_quality_regime_summary = _summarize_execution_quality_by_regime(open_trade_execution_df)
            shadow_opportunity_summary = _summarize_shadow_opportunities(df)
            shadow_opportunity_bucket_summary = _summarize_shadow_opportunities_by_bucket(df)
            backtest_parameter_recommendations = build_backtest_parameter_advice(df)
            ai_shadow_quality_summary = summarize_ai_shadow_quality(df)
            ai_shadow_quality_recommendations = build_ai_shadow_recommendations(ai_shadow_quality_summary)
        elif not execution_quality_summary:
            open_trade_execution_df = _build_open_trade_execution_quality_frame(getattr(self, 'accounts', []))
            if open_trade_execution_df is not None:
                execution_quality_summary = _summarize_execution_quality(open_trade_execution_df)
                execution_quality_bucket_summary = _summarize_execution_quality_by_bucket(open_trade_execution_df)
                execution_quality_regime_summary = _summarize_execution_quality_by_regime(open_trade_execution_df)
        calibration_recommendations = _build_calibration_recommendations({
            'open_rejection_summary': open_rejection_summary,
            'open_rejection_stage_summary': open_rejection_stage_summary,
            'bucket_summary': bucket_summary,
            'regime_summary': regime_summary,
            'symbol_summary': symbol_summary,
            'shadow_opportunity_summary': shadow_opportunity_summary,
            'shadow_opportunity_bucket_summary': shadow_opportunity_bucket_summary,
            'execution_quality_summary': execution_quality_summary,
            'execution_quality_bucket_summary': execution_quality_bucket_summary,
            'execution_quality_regime_summary': execution_quality_regime_summary,
        })
        return {
            'mode': self.mode,
            'ai_enabled': bool(self.enable_ai),
            'ai_close_enabled': bool(self.enable_ai_close),
            'position_ai_execution_enabled': False,
            'position_ai_execution_mode': '仅建议，不自动下单',
            'monitored_symbol_count': len(getattr(self, 'symbols', []) or []),
            'account_count': len(getattr(self, 'accounts', []) or []),
            'total_balance': round(total_balance, 2),
            'trade_24h_pnl': round(total_trade_24h, 2),
            'total_trade_pnl': round(total_trade_pnl, 2),
            'open_position_count': len(open_positions),
            'bucket_summary': bucket_summary,
            'regime_summary': regime_summary,
            'regime_bucket_matrix': regime_bucket_matrix,
            'symbol_summary': symbol_summary,
            'execution_quality_summary': execution_quality_summary,
            'execution_quality_bucket_summary': execution_quality_bucket_summary,
            'execution_quality_regime_summary': execution_quality_regime_summary,
            'shadow_opportunity_summary': shadow_opportunity_summary,
            'shadow_opportunity_bucket_summary': shadow_opportunity_bucket_summary,
            'backtest_parameter_recommendations': backtest_parameter_recommendations,
            'ai_shadow_quality_summary': ai_shadow_quality_summary,
            'ai_shadow_quality_recommendations': ai_shadow_quality_recommendations,
            'open_rejection_summary': open_rejection_summary,
            'open_rejection_stage_summary': open_rejection_stage_summary,
            'calibration_recommendations': calibration_recommendations,
            'last_refresh_time': getattr(self, 'last_dashboard_refresh', None),
        }

    def get_dashboard_events_payload(self, limit=10):
        focus_symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT']
        empty_payload = {
            'latest_by_symbol': [],
            'grouped_events': {symbol: [] for symbol in focus_symbols},
            'global_latest': [],
        }
        log_file = 'ai_decision_logs.csv'
        if not os.path.exists(log_file):
            return empty_payload
        try:
            df = pd.read_csv(log_file, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(log_file)

        def _clean_event_value(value):
            return None if pd.isna(value) else value

        rows = []
        for _, row in df.iloc[::-1].iterrows():
            rows.append({
                'time': _clean_event_value(row.get('触发时间')),
                'symbol': _clean_event_value(row.get('交易对')),
                'type': 'AI_DECISION',
                'direction': _clean_event_value(row.get('策略触发方向')),
                'ai_action': _clean_event_value(row.get('AI最终判定')),
                'ai_confidence': float(row.get('AI置信度', 0) or 0),
                'reason_summary': str(row.get('AI决策原因') or ''),
                'account_name': _clean_event_value(row.get('账户')) if '账户' in row.index else None,
                'market_regime': _clean_event_value(row.get('市场状态')) if '市场状态' in row.index else None,
                'strategy_bucket': _clean_event_value(row.get('策略桶')) if '策略桶' in row.index else None,
                'holding_minutes': float(row.get('持仓时长(分钟)')) if pd.notna(row.get('持仓时长(分钟)')) else None,
                'realized_side': _clean_event_value(row.get('实际持仓方向')) if '实际持仓方向' in row.index else None,
                'cost_delta_u': float(row.get('开平成本偏差(U)')) if pd.notna(row.get('开平成本偏差(U)')) else None,
                'net_rr': float(row.get('净RR')) if pd.notna(row.get('净RR')) else None,
                'gross_rr': float(row.get('粗RR')) if pd.notna(row.get('粗RR')) else None,
                'total_cost_bps': float(row.get('预估总成本bps')) if pd.notna(row.get('预估总成本bps')) else None,
                'estimated_cost_u': float(row.get('预估成本(U)')) if pd.notna(row.get('预估成本(U)')) else None,
                'btc_near_gross_rr': float(row.get('BTC近端粗RR')) if 'BTC近端粗RR' in row.index and pd.notna(row.get('BTC近端粗RR')) else None,
                'btc_near_net_rr': float(row.get('BTC近端净RR')) if 'BTC近端净RR' in row.index and pd.notna(row.get('BTC近端净RR')) else None,
                'btc_near_obstacle': float(row.get('BTC近端目标')) if 'BTC近端目标' in row.index and pd.notna(row.get('BTC近端目标')) else None,
                'btc_extended_gross_rr': float(row.get('BTC扩展粗RR')) if 'BTC扩展粗RR' in row.index and pd.notna(row.get('BTC扩展粗RR')) else None,
                'btc_extended_net_rr': float(row.get('BTC扩展净RR')) if 'BTC扩展净RR' in row.index and pd.notna(row.get('BTC扩展净RR')) else None,
                'btc_extended_obstacle': float(row.get('BTC扩展目标')) if 'BTC扩展目标' in row.index and pd.notna(row.get('BTC扩展目标')) else None,
                'btc_rr_mode': _clean_event_value(row.get('BTC RR模式')) if 'BTC RR模式' in row.index else None,
                'btc_rr_cap': float(row.get('BTC RR仓位上限')) if 'BTC RR仓位上限' in row.index and pd.notna(row.get('BTC RR仓位上限')) else None,
                'ai_entry_label': _clean_event_value(row.get('AI仓位档位')) if 'AI仓位档位' in row.index else None,
                'ai_leverage_factor': float(row.get('AI杠杆系数')) if pd.notna(row.get('AI杠杆系数')) else None,
            })

        latest_by_symbol = []
        grouped_events = {}
        for symbol in focus_symbols:
            symbol_rows = [item for item in rows if item.get('symbol') == symbol]
            grouped_events[symbol] = symbol_rows[:5]
            if symbol_rows:
                latest_by_symbol.append(symbol_rows[0])
            else:
                latest_by_symbol.append({
                    'time': None,
                    'symbol': symbol,
                    'type': 'AI_DECISION',
                    'direction': None,
                    'ai_action': None,
                    'ai_confidence': None,
                    'reason_summary': None,
                    'account_name': None,
                })

        return {
            'latest_by_symbol': latest_by_symbol,
            'grouped_events': grouped_events,
            'global_latest': rows[:limit],
        }

    def get_dashboard_btc_rr_payload(self):
        empty_payload = {
            'latest': None,
            'mode_distribution': [],
            'recent': [],
        }
        log_file = 'ai_decision_logs.csv'
        if not os.path.exists(log_file):
            return empty_payload
        try:
            df = pd.read_csv(log_file, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(log_file)
        if df.empty or '交易对' not in df.columns:
            return empty_payload
        btc_df = df[df['交易对'] == 'BTCUSDT'].copy()
        if btc_df.empty:
            return empty_payload

        def _num(row, column):
            return float(row.get(column)) if column in row.index and pd.notna(row.get(column)) else None

        def _row_payload(row):
            return {
                'time': None if pd.isna(row.get('触发时间')) else row.get('触发时间'),
                'symbol': row.get('交易对'),
                'direction': row.get('策略触发方向'),
                'ai_action': row.get('AI最终判定'),
                'ai_confidence': _num(row, 'AI置信度'),
                'net_rr': _num(row, '净RR'),
                'btc_near_net_rr': _num(row, 'BTC近端净RR'),
                'btc_extended_net_rr': _num(row, 'BTC扩展净RR'),
                'btc_near_obstacle': _num(row, 'BTC近端目标'),
                'btc_extended_obstacle': _num(row, 'BTC扩展目标'),
                'btc_rr_mode': None if pd.isna(row.get('BTC RR模式')) else row.get('BTC RR模式'),
                'btc_rr_cap': _num(row, 'BTC RR仓位上限'),
            }

        recent_rows = [_row_payload(row) for _, row in btc_df.iloc[::-1].head(5).iterrows()]
        mode_distribution = []
        if 'BTC RR模式' in btc_df.columns:
            mode_counts = btc_df['BTC RR模式'].dropna().astype(str).value_counts().head(10)
            mode_distribution = [
                {'mode': mode, 'count': int(count)}
                for mode, count in mode_counts.items()
            ]
        return {
            'latest': recent_rows[0] if recent_rows else None,
            'mode_distribution': mode_distribution,
            'recent': recent_rows,
        }

    def get_dashboard_position_ai_payload(self, limit=80):
        empty_payload = {
            'summary': {
                'monitored_positions': 0,
                'normal_holding': 0,
                'risk_watching': 0,
                'confirm_reduce': 0,
                'confirm_breakeven': 0,
                'confirm_exit': 0,
                'high_risk': 0,
                'avg_confidence': None,
            },
            'latest_by_trade': [],
            'recent': [],
        }
        log_file = 'position_ai_decision_logs.csv'
        if not os.path.exists(log_file):
            return empty_payload
        try:
            df = pd.read_csv(log_file, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(log_file)
        if df.empty:
            return empty_payload

        def _clean(value):
            return None if pd.isna(value) else value

        def _num(row, column):
            return float(row.get(column)) if column in row.index and pd.notna(row.get(column)) else None

        rows = []
        for _, row in df.iloc[::-1].iterrows():
            rows.append({
                'time': _clean(row.get('时间')),
                'account_name': _clean(row.get('账户')),
                'symbol': _clean(row.get('交易对')),
                'trade_id': _clean(row.get('订单ID')),
                'side': _clean(row.get('方向')),
                'entry_price': _num(row, '开仓价'),
                'current_price': _num(row, '当前价'),
                'unrealized_pnl': _num(row, '浮动盈亏(U)'),
                'unrealized_pnl_pct': _num(row, '浮动盈亏比例(%)'),
                'holding_minutes': _num(row, '持仓分钟'),
                'ai_action': _clean(row.get('AI原始建议')),
                'ai_confidence': _num(row, 'AI置信度'),
                'risk_level': _clean(row.get('AI风险等级')),
                'ai_reason': _clean(row.get('AI理由')),
                'confirmed_action': _clean(row.get('系统确认状态')),
                'risk_streak': _num(row, '连续风险次数'),
                'hold_streak': _num(row, '连续持有次数'),
                'execution_state': _clean(row.get('是否执行')),
            })

        def _active_position_keys():
            keys = set()
            for acc in getattr(self, 'accounts', []) or []:
                account_name = getattr(acc, 'name', None)
                positions = getattr(acc, 'real_positions', {}) or {}
                meta_map = getattr(acc, 'real_trade_meta', {}) or {}
                for symbol in positions.keys():
                    trade_id = (meta_map.get(symbol, {}) or {}).get('trade_id')
                    if trade_id:
                        keys.add((account_name, symbol, trade_id))
            return keys

        active_position_keys = _active_position_keys()
        latest_by_trade_map = {}
        for item in rows:
            key = (item.get('account_name'), item.get('symbol'), item.get('trade_id'))
            if key in active_position_keys and key not in latest_by_trade_map:
                latest_by_trade_map[key] = item
        latest_by_trade = list(latest_by_trade_map.values())
        statuses = [str(item.get('confirmed_action') or '') for item in latest_by_trade]
        risk_levels = [str(item.get('risk_level') or '').lower() for item in latest_by_trade]
        confidences = [float(item.get('ai_confidence')) for item in latest_by_trade if item.get('ai_confidence') is not None]
        summary = {
            'monitored_positions': len(latest_by_trade),
            'normal_holding': statuses.count('正常持有'),
            'risk_watching': statuses.count('风险观察中'),
            'confirm_reduce': statuses.count('确认减仓防守'),
            'confirm_breakeven': statuses.count('确认推保本线'),
            'confirm_exit': statuses.count('确认全部退出'),
            'high_risk': sum(1 for level in risk_levels if level == 'high'),
            'avg_confidence': round(sum(confidences) / len(confidences), 1) if confidences else None,
        }
        return {
            'summary': summary,
            'latest_by_trade': latest_by_trade[:limit],
            'recent': rows[:limit],
        }

    def start_dashboard_server(self, host='127.0.0.1', port=5000):
        from dashboard_web import create_dashboard_app
        app = create_dashboard_app(self)
        app.run(host=host, port=port, debug=False, use_reloader=False)

    def _send_daily_report(self):
        """发送每日报告与 AI 审核统计"""
        logger.info("开始生成每日报告...")
        report_content = [f"**AstraQuant 每日报告**\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]

        for acc in self.accounts:
            history_file = f"trades_history_{acc.name}.csv"
            try:
                trade_24h_pnl = 0.0
                total_trade_pnl = 0.0
                win_rate = 0.0
                trade_count = 0
                if os.path.exists(history_file):
                    df = pd.read_csv(history_file, encoding='utf-8-sig')
                    if '平仓时间' in df.columns and '净盈亏(U)' in df.columns:
                        df = df[df.apply(_is_robot_trade_history_row, axis=1)].copy()
                        df['平仓时间_dt'] = pd.to_datetime(df['平仓时间'])
                        df['净盈亏(U)'] = pd.to_numeric(df['净盈亏(U)'], errors='coerce').fillna(0.0)
                        last_24h = df[df['平仓时间_dt'] > (datetime.now() - pd.Timedelta(days=1))]
                        trade_24h_pnl = float(last_24h['净盈亏(U)'].sum()) if not last_24h.empty else 0.0
                        total_trade_pnl = float(df['净盈亏(U)'].sum()) if not df.empty else 0.0
                        trade_count = len(last_24h)
                        win_rate = (last_24h['净盈亏(U)'] > 0).mean() * 100 if not last_24h.empty else 0
                report_content.append(f"\n**账户: {acc.name}**\n- 交易24H盈亏: `{trade_24h_pnl:+.2f} USDT`\n- 总盈利: `{total_trade_pnl:+.2f} USDT`\n- 24H 胜率: `{win_rate:.1f}%` (共 {trade_count} 笔)")
            except Exception as e:
                logger.error(f"生成账户日报失败 [{acc.name}]: {e}")

        log_file = "ai_decision_logs.csv"
        bucket_summary = []
        regime_summary = []
        regime_bucket_matrix = []
        symbol_summary = []
        execution_quality_summary = []
        execution_quality_bucket_summary = []
        execution_quality_regime_summary = []
        shadow_opportunity_summary = []
        shadow_opportunity_bucket_summary = []
        backtest_parameter_recommendations = []
        ai_shadow_quality_summary = []
        ai_shadow_quality_recommendations = []
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file, encoding='utf-8-sig')
                if "后续4H涨跌(%)" not in df.columns:
                    df["后续4H涨跌(%)"] = np.nan
                if "拦截评估结果" not in df.columns:
                    df["拦截评估结果"] = ""
                else:
                    df["拦截评估结果"] = df["拦截评估结果"].fillna('').astype(str)
                if "Shadow结果" not in df.columns:
                    df["Shadow结果"] = ""
                else:
                    df["Shadow结果"] = df["Shadow结果"].fillna('').astype(str)
                if "Shadow机会收益(%)" not in df.columns:
                    df["Shadow机会收益(%)"] = np.nan
                if "Shadow机会方向" not in df.columns:
                    df["Shadow机会方向"] = ""
                else:
                    df["Shadow机会方向"] = df["Shadow机会方向"].fillna('').astype(str)

                mask = (df["AI最终判定"] == "PASS") & (df["后续4H涨跌(%)"].isna())
                pending_indices = df[mask].index
                if len(pending_indices) > 0:
                    logger.info(f"发现 {len(pending_indices)} 条待补全 AI 审核记录，开始回填 CSV...")
                    for idx in pending_indices:
                        row = df.loc[idx]
                        try:
                            dt = datetime.strptime(row["触发时间"], '%Y-%m-%d %H:%M:%S')
                            ts_ms = int(dt.timestamp() * 1000)
                            price_move = self._get_future_price_for_audit(row["交易对"], ts_ms)
                            if price_move:
                                change_pct = (price_move['last'] - price_move['entry']) / price_move['entry'] * 100
                                is_hero = (row["策略触发方向"] == "LONG" and change_pct < 0) or (row["策略触发方向"] == "SHORT" and change_pct > 0)
                                status = "成功拦截" if is_hero else "错失机会"
                                shadow_status, opportunity_return = _evaluate_shadow_outcome(row["策略触发方向"], change_pct)
                                df.at[idx, "后续4H涨跌(%)"] = round(change_pct, 2)
                                df.at[idx, "拦截评估结果"] = status
                                df.at[idx, "Shadow结果"] = shadow_status
                                df.at[idx, "Shadow机会收益(%)"] = opportunity_return
                                df.at[idx, "Shadow机会方向"] = row["策略触发方向"]
                        except Exception:
                            pass
                        time.sleep(0.2)
                    df.to_csv(log_file, index=False, encoding='utf-8-sig')

                completed_mask = (df["AI最终判定"] == "PASS") & (df["拦截评估结果"].isin(["成功拦截", "错失机会"]))
                total_completed = int(completed_mask.sum())
                total_success = int((df["拦截评估结果"] == "成功拦截").sum())

                df['触发时间_dt'] = pd.to_datetime(df['触发时间'])
                last_24h_ai = df[(df['触发时间_dt'] > (datetime.now() - pd.Timedelta(days=1))) & completed_mask]
                if not last_24h_ai.empty:
                    h24_success = int((last_24h_ai["拦截评估结果"] == "成功拦截").sum())
                    h24_acc = (h24_success / len(last_24h_ai)) * 100
                    report_content.append(f"\n**AI 拦截表现(近24H):**\n- 拦截准确率: `{h24_acc:.1f}%` ({h24_success}/{len(last_24h_ai)})")
                    pass_details = []
                    for _, row in last_24h_ai.tail(5).iterrows():
                        mark = '✅' if row['拦截评估结果'] == '成功拦截' else '❌'
                        pass_details.append(f"  - {row['交易对']} | {row['策略触发方向']} | {row['后续4H涨跌(%)']:+.2f}% {mark}")
                    if pass_details:
                        report_content.append("\n".join(pass_details))

                if total_completed > 0:
                    report_content.append(f"\n**AI 历史总拦截准确率:** `{(total_success / total_completed) * 100:.1f}%` ({total_success}/{total_completed})")

                bucket_summary = _summarize_strategy_buckets(df)
                regime_summary = _summarize_market_regimes(df)
                regime_bucket_matrix = _summarize_regime_bucket_matrix(df)
                symbol_summary = _summarize_symbols(df)
                execution_quality_summary = _summarize_execution_quality(df)
                execution_quality_bucket_summary = _summarize_execution_quality_by_bucket(df)
                execution_quality_regime_summary = _summarize_execution_quality_by_regime(df)
                shadow_opportunity_summary = _summarize_shadow_opportunities(df)
                shadow_opportunity_bucket_summary = _summarize_shadow_opportunities_by_bucket(df)
                backtest_parameter_recommendations = build_backtest_parameter_advice(df)
                ai_shadow_quality_summary = summarize_ai_shadow_quality(df)
                ai_shadow_quality_recommendations = build_ai_shadow_recommendations(ai_shadow_quality_summary)
                live_ai_review_rows = build_live_ai_review_rows(df)
                live_ai_review_recommendations = build_live_ai_review_recommendations(live_ai_review_rows)
                if bucket_summary:
                    report_content.append("\n**策略桶表现摘要**")
                    for row in bucket_summary[:4]:
                        report_content.append(
                            f"- {row['strategy_bucket']} | 信号 {row['signal_count']} | 开仓 {row['trade_count']} | 胜率 {row['win_rate']:.1f}% | PnL {row['realized_pnl']:+.2f}U"
                        )
                if regime_summary:
                    report_content.append("\n**市场状态摘要**")
                    for row in regime_summary[:4]:
                        report_content.append(
                            f"- {row['market_regime']} | 信号 {row['signal_count']} | 开仓 {row['trade_count']} | 胜率 {row['win_rate']:.1f}% | PnL {row['realized_pnl']:+.2f}U"
                        )
                if regime_bucket_matrix:
                    report_content.append("\n**状态×策略组合**")
                    for row in regime_bucket_matrix[:6]:
                        report_content.append(
                            f"- {row['market_regime']} × {row['strategy_bucket']} | 信号 {row['signal_count']} | 开仓 {row['trade_count']} | 胜率 {row['win_rate']:.1f}% | PnL {row['realized_pnl']:+.2f}U"
                        )
                if execution_quality_summary:
                    report_content.append("\n**执行质量摘要**")
                    for row in execution_quality_summary[:3]:
                        report_content.append(
                            f"- {row['scope']} | 成交 {row['trade_count']} 笔 | 预估 {row['avg_estimated_cost_u']:.4f}U | 实际 {row['avg_actual_cost_u']:.4f}U | 偏差 {row['avg_cost_delta_u']:+.4f}U"
                        )
                if execution_quality_bucket_summary:
                    report_content.append("\n**执行质量×策略桶**")
                    for row in execution_quality_bucket_summary[:4]:
                        report_content.append(
                            f"- {row['strategy_bucket']} | 成交 {row['trade_count']} 笔 | 预估 {row['avg_estimated_cost_u']:.4f}U | 实际 {row['avg_actual_cost_u']:.4f}U | 偏差 {row['avg_cost_delta_u']:+.4f}U"
                        )
                if execution_quality_regime_summary:
                    report_content.append("\n**执行质量×市场状态**")
                    for row in execution_quality_regime_summary[:4]:
                        report_content.append(
                            f"- {row['market_regime']} | 成交 {row['trade_count']} 笔 | 预估 {row['avg_estimated_cost_u']:.4f}U | 实际 {row['avg_actual_cost_u']:.4f}U | 偏差 {row['avg_cost_delta_u']:+.4f}U"
                        )
                if shadow_opportunity_summary:
                    report_content.append("\n**Shadow 未开仓机会追踪**")
                    for row in shadow_opportunity_summary[:3]:
                        report_content.append(
                            f"- {row['scope']} | 样本 {row['sample_count']} | 错过 {row['missed_count']} | 避免亏损 {row['avoided_count']} | 错过率 {row['missed_rate']:.1f}% | 平均机会 {row['avg_opportunity_return_pct']:+.2f}%"
                        )
                if shadow_opportunity_bucket_summary:
                    report_content.append("\n**Shadow 机会×策略桶**")
                    for row in shadow_opportunity_bucket_summary[:4]:
                        report_content.append(
                            f"- {row['strategy_bucket']} | 样本 {row['sample_count']} | 错过 {row['missed_count']} | 避免亏损 {row['avoided_count']} | 错过率 {row['missed_rate']:.1f}% | 平均机会 {row['avg_opportunity_return_pct']:+.2f}%"
                        )
                if backtest_parameter_recommendations:
                    report_content.append("\n**回测/复盘参数建议**")
                    for row in backtest_parameter_recommendations[:6]:
                        report_content.append(f"- [{row['severity']}] {row['recommendation']}")
                if ai_shadow_quality_summary:
                    report_content.append("\n**AI质量×Shadow摘要**")
                    for row in ai_shadow_quality_summary[:5]:
                        report_content.append(
                            f"- 置信度 {row['confidence_band']} | 样本 {row['sample_count']} | 放行 {row['approved_count']} | PASS {row['pass_count']} | 放行胜率 {row['approved_win_rate']:.1f}% | Shadow错过 {row['shadow_missed_rate']:.1f}%"
                        )
                if ai_shadow_quality_recommendations:
                    report_content.append("\n**AI质量×Shadow建议**")
                    for item in ai_shadow_quality_recommendations[:5]:
                        report_content.append(f"- {item}")
                if live_ai_review_rows:
                    report_content.append("\n**AI实盘复盘摘要**")
                    for row in live_ai_review_rows[:8]:
                        report_content.append(
                            f"- {row['scope']}:{row['name']} | 样本 {row['sample_count']} | 放行 {row['approved_count']} | PASS {row['pass_count']} | 胜率 {row['win_rate']:.1f}% | PnL {row['realized_pnl']:+.2f}U | Shadow错过 {row['shadow_missed_rate']:.1f}%"
                        )
                if live_ai_review_recommendations:
                    report_content.append("\n**AI实盘复盘建议**")
                    for item in live_ai_review_recommendations[:6]:
                        report_content.append(f"- {item}")
            except Exception as e:
                logger.error(f"AI 日报统计失败: {e}")

        open_rejection_summary = _summarize_open_rejections(getattr(self, 'recent_open_rejections', []), since_hours=24)
        open_rejection_stage_summary = _summarize_open_rejection_stages(getattr(self, 'recent_open_rejections', []), since_hours=24)
        if open_rejection_summary:
            report_content.append("\n**未开仓原因摘要**")
            for row in open_rejection_summary[:6]:
                report_content.append(f"- {row['reason']} | {row['count']} 次")
        if open_rejection_stage_summary:
            report_content.append("\n**未开仓阶段摘要**")
            for row in open_rejection_stage_summary[:6]:
                report_content.append(f"- {row['stage']} | {row['count']} 次")

        calibration_recommendations = _build_calibration_recommendations({
            'open_rejection_summary': open_rejection_summary,
            'open_rejection_stage_summary': open_rejection_stage_summary,
            'bucket_summary': bucket_summary,
            'regime_summary': regime_summary,
            'symbol_summary': symbol_summary,
            'shadow_opportunity_summary': shadow_opportunity_summary,
            'shadow_opportunity_bucket_summary': shadow_opportunity_bucket_summary,
            'execution_quality_summary': execution_quality_summary,
            'execution_quality_bucket_summary': execution_quality_bucket_summary,
            'execution_quality_regime_summary': execution_quality_regime_summary,
        })
        if calibration_recommendations:
            report_content.append("\n**参数校准建议**")
            for item in calibration_recommendations:
                report_content.append(f"- {item}")

        report_text = "\n".join(report_content)
        ToolKit.send_dingtalk_msg(report_text, title="AstraQuant 每日报告")

    def _graceful_shutdown(self, signum, frame):
        logger.warning("收到停止信号，正在保存所有账户状态并安全退出...")
        if self.mode in ['TESTNET', 'REAL']:
            for acc in self.accounts:
                ToolKit.save_active_trades(acc.real_trade_meta, account_name=acc.name)
        sys.exit(0)

    def get_symbol_spec(self, symbol):
        if symbol in getattr(config, 'MAINSTREAM', []): return config.SYMBOL_CONFIG["MAINSTREAM"]
        elif symbol in getattr(config, 'MID_TIER', []): return config.SYMBOL_CONFIG["MID_TIER"]
        else: return config.SYMBOL_CONFIG.get("MEME_SHIT", {"leverage": 3, "margin_ratio": 0.03, "risk_level": 7})

    def _is_duplicate_signal(self, symbol, algo_side, data, signal_text, account_name=""):
        cache_key = f"{account_name}_{symbol}_{algo_side}"
        current_time = time.time()
        current_price = float(data.get('price') or 0.0)
        atr_pct = data.get('atr', 0) / data.get('price', 1)

        if cache_key in self.algo_cache:
            cache_meta = self.algo_cache[cache_key]
            last_time = float(cache_meta.get('time', 0.0) or 0.0)
            last_price = float(cache_meta.get('price', 0.0) or 0.0)
            cooldown = 120 if atr_pct > 0.03 else 300
            effective_cooldown = cooldown
            if last_price > 0:
                if algo_side == 'LONG':
                    favorable_move = (current_price - last_price) / last_price
                else:
                    favorable_move = (last_price - current_price) / last_price
                if favorable_move > 0.003:
                    effective_cooldown = 60
            if current_time - last_time < effective_cooldown:
                return True

        self.algo_cache[cache_key] = {'time': current_time, 'price': current_price}
        return False

    def _position_ai_key(self, account_name, symbol, trade_id):
        return f"{account_name}:{symbol}:{trade_id}"

    def _update_position_ai_state(self, position_key, decision):
        action = str(decision.get('action', '继续持有') or '继续持有').strip()
        confidence = float(decision.get('confidence', 0.0) or 0.0)
        risk_level = str(decision.get('risk_level', 'normal') or 'normal').strip().lower()
        previous = dict(self.position_ai_state.get(position_key, {}) or {})
        risk_actions = {'减仓防守', '推保本线', '全部退出'}

        risk_streak = int(previous.get('risk_streak', 0) or 0)
        hold_streak = int(previous.get('hold_streak', 0) or 0)
        previous_status = str(previous.get('confirmed_action', '正常持有') or '正常持有')
        previous_action = str(previous.get('last_action', '') or '')

        if action == '继续持有':
            hold_streak += 1
            risk_streak = 0
            if previous_status != '正常持有' or previous_action in risk_actions:
                confirmed_action = '正常持有' if hold_streak >= 2 else '风险缓和观察中'
            else:
                confirmed_action = '正常持有'
        else:
            risk_streak += 1
            hold_streak = 0
            if action == '减仓防守' and (risk_streak >= 2 or (confidence >= 80 and risk_level == 'high')):
                confirmed_action = '确认减仓防守'
            elif action == '推保本线' and (risk_streak >= 2 or confidence >= 75):
                confirmed_action = '确认推保本线'
            elif action == '全部退出' and ((previous_action == '全部退出' and risk_streak >= 2) or (confidence >= 85 and risk_level == 'high')):
                confirmed_action = '确认全部退出'
            else:
                confirmed_action = '风险观察中'

        state = {
            'last_action': action,
            'last_confidence': confidence,
            'last_risk_level': risk_level,
            'last_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'risk_streak': risk_streak,
            'hold_streak': hold_streak,
            'confirmed_action': confirmed_action,
        }
        self.position_ai_state[position_key] = state
        return state

    def _position_ai_cache_ttl_for_state(self, position_key):
        status = str((self.position_ai_state.get(position_key) or {}).get('confirmed_action', '正常持有') or '正常持有')
        if status in {'确认减仓防守', '确认推保本线', '确认全部退出'}:
            return int(getattr(config, 'POSITION_AI_CONFIRMED_RISK_TTL', 300) or 300)
        if status in {'风险观察中', '风险缓和观察中'}:
            return int(getattr(config, 'POSITION_AI_RISK_TTL', 900) or 900)
        return int(getattr(config, 'POSITION_AI_NORMAL_TTL', 3600) or 3600)

    def _position_ai_update_peak_pnl_pct(self, position_key, position_data):
        try:
            pnl_pct = float(position_data.get('unrealized_pnl_pct', 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if not hasattr(self, 'position_ai_peak_pnl_pct'):
            self.position_ai_peak_pnl_pct = {}
        previous_peak = float(self.position_ai_peak_pnl_pct.get(position_key, pnl_pct) or 0.0)
        peak = max(previous_peak, pnl_pct)
        self.position_ai_peak_pnl_pct[position_key] = peak
        return peak

    def _position_ai_profit_drawdown_recheck_needed(self, position_key, position_data):
        try:
            pnl_pct = float(position_data.get('unrealized_pnl_pct', 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        peak = self._position_ai_update_peak_pnl_pct(position_key, position_data)
        if peak is None:
            return False
        min_peak = float(getattr(config, 'POSITION_AI_PROFIT_PROTECT_MIN_PCT', 0.5) or 0.5)
        drawdown_ratio = float(getattr(config, 'POSITION_AI_PROFIT_DRAWDOWN_RATIO', 0.5) or 0.5)
        if peak < min_peak or pnl_pct < 0:
            return False
        return pnl_pct <= peak * (1 - drawdown_ratio)

    def _position_ai_fast_recheck_needed(self, position_data, position_key=None):
        if position_key and self._position_ai_profit_drawdown_recheck_needed(position_key, position_data):
            return True
        try:
            pnl_pct = float(position_data.get('unrealized_pnl_pct', 0.0) or 0.0)
            adverse_threshold = float(getattr(config, 'POSITION_AI_ADVERSE_PNL_PCT', 3.0) or 3.0)
            if pnl_pct <= -adverse_threshold:
                return True

            stop_loss = position_data.get('stop_loss')
            current_price = float(position_data.get('current_price', 0.0) or 0.0)
            side = str(position_data.get('side', '') or '').upper()
            stop_buffer = float(getattr(config, 'POSITION_AI_STOP_BREACH_BUFFER', 0.002) or 0.002)
            if stop_loss is not None:
                stop_loss = float(stop_loss)
                if side == 'LONG' and current_price > 0 and current_price <= stop_loss * (1 + stop_buffer):
                    return True
                if side == 'SHORT' and current_price > 0 and current_price >= stop_loss * (1 - stop_buffer):
                    return True
        except (TypeError, ValueError):
            return False
        return False

    def _should_notify_position_ai(self, position_key, decision, state, position_data=None):
        previous_status = str((self.position_ai_state.get(position_key) or {}).get('confirmed_action', '正常持有') or '正常持有')
        current_status = str(state.get('confirmed_action', '正常持有') or '正常持有')
        confirmed_risk_statuses = {'确认减仓防守', '确认推保本线', '确认全部退出'}
        if current_status in confirmed_risk_statuses:
            return True
        if current_status != previous_status:
            return current_status in {'风险观察中', '风险缓和观察中', '正常持有'}
        if position_data:
            if self._position_ai_fast_recheck_needed(position_data, position_key):
                return True
        return False

    def _evaluate_position_ai_with_cache(self, account_name, symbol, trade_id, position_data, prompt_mode='aggressive'):
        position_key = self._position_ai_key(account_name, symbol, trade_id)
        current_time = time.time()
        cached = self.position_ai_cache.get(position_key)
        fast_recheck_needed = self._position_ai_fast_recheck_needed(position_data, position_key)
        if cached and not fast_recheck_needed and current_time - float(cached.get('time', 0.0) or 0.0) < self._position_ai_cache_ttl_for_state(position_key):
            return cached.get('decision'), cached.get('state'), False

        decision = self.ai_advisor.evaluate_position(position_data, prompt_mode=prompt_mode)
        if not decision:
            return None, None, False

        state = self._update_position_ai_state(position_key, decision)
        self.position_ai_cache[position_key] = {
            'decision': decision,
            'state': state,
            'time': current_time,
        }
        self._log_position_ai_decision(account_name, symbol, trade_id, position_data, decision, state)
        return decision, state, True

    def _log_position_ai_decision(self, account_name, symbol, trade_id, position_data, decision, state):
        log_file = 'position_ai_decision_logs.csv'
        row = {
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '账户': account_name,
            '交易对': symbol,
            '订单ID': trade_id,
            '方向': position_data.get('side', ''),
            '开仓价': position_data.get('entry_price', ''),
            '当前价': position_data.get('current_price', ''),
            '浮动盈亏(U)': position_data.get('unrealized_pnl', ''),
            '浮动盈亏比例(%)': position_data.get('unrealized_pnl_pct', ''),
            '持仓分钟': position_data.get('holding_minutes', ''),
            'AI原始建议': decision.get('action', ''),
            'AI置信度': decision.get('confidence', 0),
            'AI风险等级': decision.get('risk_level', ''),
            'AI理由': _safe_ai_reason(decision.get('reason', '')),
            '系统确认状态': state.get('confirmed_action', ''),
            '连续风险次数': state.get('risk_streak', 0),
            '连续持有次数': state.get('hold_streak', 0),
            '是否执行': '否，仅记录',
        }
        try:
            pd.DataFrame([row]).to_csv(log_file, mode='a', header=not os.path.isfile(log_file), index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f'写入持仓后 AI 决策日志失败: {e}')

    def _log_position_ai_execution(self, account_name, symbol, trade_id, execution_state):
        log_file = 'position_ai_decision_logs.csv'
        if not os.path.exists(log_file):
            return
        try:
            try:
                df = pd.read_csv(log_file, encoding='utf-8-sig')
            except Exception:
                df = pd.read_csv(log_file)
            if df.empty or '是否执行' not in df.columns:
                return
            mask = (df.get('账户') == account_name) & (df.get('交易对') == symbol) & (df.get('订单ID') == trade_id)
            if not mask.any():
                return
            df.loc[df[mask].index[-1], '是否执行'] = execution_state
            df.to_csv(log_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f'回写持仓后 AI 执行状态失败: {e}')

    def _evaluate_ai_with_cache(self, symbol, algo_side, data, signal_text, prompt_mode="aggressive"):
        atr = data.get('atr', data['price'] * 0.01)
        price_bucket = int(data['price'] / (atr * 0.2)) 
        
        try:
            last_kline = data.get('raw_klines_15m', [])[-1]
            stable_kline_id = hash(str(last_kline[:2])) if isinstance(last_kline, list) else hash(str(last_kline))
        except Exception:
            stable_kline_id = int(time.time() // 900) 

        cache_key = f"{symbol}_{algo_side}_{price_bucket}_{stable_kline_id}"
        current_time = time.time()

        if cache_key in self.ai_cache:
            if current_time - self.ai_cache[cache_key]['time'] < self.ai_cache_ttl:
                return self.ai_cache[cache_key]['decision']
                
            logger.info(f"盘面异动，调用 AI 独立评估 {symbol} (触发源: {algo_side} | 模式: {prompt_mode})...")
        
        ai_market_data = dict(data)
        ai_market_data['direction'] = algo_side
        rr_preview = self._preview_realistic_rr(algo_side, data, float(data.get('price') or 0.0))
        if rr_preview:
            rr_value = float(rr_preview.get('rr', 0.0) or 0.0)
            ai_market_data['realistic_rr'] = rr_value
            ai_market_data['gross_rr'] = rr_value
            ai_market_data['rr'] = rr_value
            ai_market_data['current_rr'] = rr_value
            ai_market_data['net_expected_rr'] = float(rr_preview.get('net_rr', rr_value) or 0.0)
            ai_market_data['net_rr'] = float(rr_preview.get('net_rr', rr_value) or 0.0)
            ai_market_data['rr_costs'] = {
                'fee_bps': rr_preview.get('fee_bps'),
                'slippage_bps': rr_preview.get('slippage_bps'),
                'funding_bps': rr_preview.get('funding_bps'),
                'total_cost_bps': rr_preview.get('total_cost_bps'),
            }
            ai_market_data['closest_obstacle'] = rr_preview.get('closest_obstacle')
        ai_decision = self.ai_advisor.evaluate_signal(ai_market_data, prompt_mode=prompt_mode)

        if ai_decision:
            self.ai_cache[cache_key] = {'decision': ai_decision, 'time': current_time}
            self._log_ai_decision(symbol, algo_side, ai_decision.get('action', 'PASS'), _safe_ai_reason(ai_decision.get('reason', '未知')), ai_market_data, signal_text, ai_decision.get('confidence', 0))
            
        return ai_decision

    def _log_ai_decision(self, symbol, algo_signal, final_action, ai_reason, data, signal_text, ai_confidence=0):
        """记录 AI 审核决策日志"""

        def clean_np_types(obj):
            if isinstance(obj, dict):
                return {k: clean_np_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_np_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        try:
            clean_data = clean_np_types(data)
            log_file = "ai_decision_logs.csv"
            dec_rec = _build_ai_decision_log_record(
                symbol=symbol,
                algo_signal=algo_signal,
                final_action=final_action,
                ai_reason=ai_reason,
                data=clean_data,
                signal_text=signal_text,
                ai_confidence=ai_confidence,
                account_name=clean_data.get('account_name', ''),
                order_id=clean_data.get('order_id', ''),
            )
            expected_columns = list(dec_rec.keys())
            if os.path.isfile(log_file):
                try:
                    df = pd.read_csv(log_file, encoding='utf-8-sig')
                except Exception:
                    df = pd.read_csv(log_file)
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = np.nan if col not in {'触发时间', '平仓时间', '账户', '订单ID', '交易对', '策略触发方向', 'AI最终判定', 'AI决策原因', '策略数据', '原始数据', 'AI仓位档位', '拦截评估结果'} else ''
                df = _append_ai_decision_log_row(df, expected_columns, dec_rec)
                df.to_csv(log_file, index=False, encoding='utf-8-sig')
            else:
                pd.DataFrame([dec_rec], columns=expected_columns).to_csv(log_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"记录 AI 审核日志失败: {e}")

    def start_monitoring(self):
        last_report_day = "" 
        tz_shanghai = timezone(timedelta(hours=8))
        try:
            while True:
                next_loop = self.loop_count + 1
                logger.info(f"开始第 {next_loop} 轮扫描 (监控 {len(self.symbols)} 个币种 x {len(self.accounts)} 个账户)")
                if os.path.exists("PANIC.flag"):
                    logger.error("发现紧急文件 PANIC.flag，所有账户立即避险。")
                    for acc in self.accounts: acc.panic_close_all()
                    os.remove("PANIC.flag")  
                    logger.info("系统已挂起。")
                    sys.exit(1) 
                    
                if self.mode in ['TESTNET', 'REAL']:
                    sync_success = True
                    for acc in self.accounts:
                        temp_positions = acc.engine.sync_real_positions(self.symbols)
                        temp_balance = acc.engine.get_real_usdt_balance()
                        
                        if temp_positions is None or temp_balance is None:
                            logger.error(f"[网络熔断] {acc.name} 状态同步失败，跳过本轮动作。")
                            sync_success = False
                            break
                        acc.real_positions = temp_positions
                        acc.real_balance = temp_balance
                        self._append_balance_snapshot(acc.name, temp_balance)
                    
                    if not sync_success:
                        self.last_mihomo_restart = ToolKit.check_and_restart_proxy(self.last_mihomo_restart)
                        time.sleep(15) 
                        continue

                fetch_success_count = 0
                price_snapshots = []
                market_rows = []
                ranked_candidates = []

                fast_scan_trigger_symbols = []

                for s in self.symbols:
                    try:
                        data = self.analyzer.get_market_data(s)
                        if data:
                            fetch_success_count += 1
                            rt_price = data.get('realtime_price')
                            rt_display = f"{rt_price:.4f}" if isinstance(rt_price, (int, float)) else 'N/A'
                            price_snapshots.append(f"{s}:策略{data['price']:.4f}/实时{rt_display}")
                            data['signal'] = self.analyzer.analyze_signals(data)
                            spec = self.get_symbol_spec(s)
                            market_rows.append((s, data, spec))
                            long_score = float(data.get('long_score') or 0.0)
                            short_score = float(data.get('short_score') or 0.0)
                            if data.get('market_regime') == 'strong_trend' or max(long_score, short_score) >= 7.0:
                                fast_scan_trigger_symbols.append(f"{s}:{max(long_score, short_score):.1f}/{data.get('market_regime', 'N/A')}")
                            candidate = _build_runtime_trade_candidate(s, data, spec, self.accounts)
                            if candidate:
                                ranked_candidates.append(candidate)
                    except Exception as e:
                        logger.error(f"[主循环容错] 处理 {s} 数据时发生异常，已跳过本轮: {e}")
                        continue
                    time.sleep(0.3)

                ranked_candidates = _select_trade_candidates_for_available_slots(ranked_candidates, self.accounts)
                market_rows = _sort_market_rows_by_candidates(market_rows, ranked_candidates, self.symbols)
                candidate_assignments = _allocate_trade_candidates_to_accounts(ranked_candidates, self.accounts)
                assigned_accounts_by_symbol = {}
                for item in candidate_assignments:
                    assigned_accounts_by_symbol.setdefault(item['symbol'], set()).add(item['account'].name)

                for s, data, spec in market_rows:
                    for acc in self.accounts:
                        if s in (getattr(acc, 'real_positions', {}) or {}):
                            acc.real_positions[s]['current_price'] = float(data.get('realtime_price') or data.get('price') or 0.0)
                        acc.risk_controller.manage_positions(s, data['price'], override_level=spec['risk_level'], leverage=spec['leverage'])
                        should_process_open = not assigned_accounts_by_symbol or acc.name in assigned_accounts_by_symbol.get(s, set())
                        if self.mode == 'SIMULATE':
                            if should_process_open:
                                acc.process_simulate_trading(data, spec)
                        else:
                            if should_process_open:
                                acc.process_real_trading(data, spec)

                    has_position = any(s in (getattr(acc, 'real_positions', {}) or {}) for acc in self.accounts)
                    position_side = None
                    latest_ai_action = data.get('ai_action', 'N/A')
                    latest_ai_confidence = float(data.get('ai_confidence') or 0.0)
                    for acc in self.accounts:
                        pos = (getattr(acc, 'real_positions', {}) or {}).get(s)
                        if pos:
                            position_side = pos.get('side')
                            break
                    data['has_position'] = has_position
                    data['position_side'] = position_side
                    data['ai_action'] = latest_ai_action
                    data['ai_confidence'] = latest_ai_confidence
                    self._update_dashboard_symbol_snapshot(s, data)
                    self.last_dashboard_refresh = self._snapshot_now_str()
                
                if fetch_success_count == 0 and len(self.symbols) > 0:
                    self.last_mihomo_restart = ToolKit.check_and_restart_proxy(self.last_mihomo_restart)

                self.loop_count += 1
                normal_scan_interval = int(getattr(config, 'SCAN_INTERVAL', 113) or 113)
                if fast_scan_trigger_symbols:
                    self.fast_scan_until = max(self.fast_scan_until, time.time() + 600)
                    logger.info(f"强信号触发加速扫描 10 分钟 | {' | '.join(fast_scan_trigger_symbols[:5])}")
                scan_interval = 60 if time.time() < self.fast_scan_until else normal_scan_interval
                scan_mode = '加速扫描' if scan_interval == 60 else '普通扫描'
                price_summary = ' | '.join(price_snapshots) if price_snapshots else '本轮无有效价格数据'
                logger.info(f"第 {self.loop_count} 轮扫描完成 | {scan_mode} | 成功获取 {fetch_success_count}/{len(self.symbols)} 个币种数据 | 当前价格: {price_summary} | {scan_interval} 秒后进入下一轮")

                now_sh = datetime.now(tz_shanghai)
                current_day = now_sh.strftime('%Y-%m-%d')
                
                if now_sh.hour == 10 and last_report_day != current_day:
                    logger.info(f"触发定时任务：当前上海时间 {now_sh.strftime('%Y-%m-%d %H:%M:%S')}")
                    self._send_daily_report()
                    last_report_day = current_day

                time.sleep(scan_interval)
                
        except KeyboardInterrupt:
            self._graceful_shutdown(None, None)

if __name__ == "__main__":
    _configure_stdio_utf8()
    monitor = OKXAlphaMonitor(enable_ai=True, enable_ai_close=True)
    if bool(getattr(config, 'ENABLE_WEB_DASHBOARD', False)):
        import threading
        threading.Thread(
            target=monitor.start_dashboard_server,
            kwargs={
                'host': getattr(config, 'WEB_DASHBOARD_HOST', '127.0.0.1'),
                'port': int(getattr(config, 'WEB_DASHBOARD_PORT', 5000)),
            },
            daemon=True,
        ).start()
    monitor.start_monitoring()








