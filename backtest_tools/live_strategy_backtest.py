import argparse
import concurrent.futures
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from main import (
    SubAccount,
    _allocate_trade_candidates_to_accounts,
    _build_runtime_trade_candidate,
    _compute_realistic_rr_preview,
    _extract_algo_side_from_signal,
    _sort_market_rows_by_candidates,
)
from market_analyzer import MarketAnalyzer

logging.getLogger().setLevel(logging.ERROR)

CACHE_DIR = ROOT / 'Cache_MarketData_LiveStrategy'
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
INTERVALS = {'15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D'}
INTERVAL_MS = {'15m': 15 * 60 * 1000, '1h': 60 * 60 * 1000, '4h': 4 * 60 * 60 * 1000, '1d': 24 * 60 * 60 * 1000}


class BacktestToolkit:
    @staticmethod
    def save_active_trades(*_args, **_kwargs):
        return None

    @staticmethod
    def save_runtime_state(*_args, **_kwargs):
        return None

    @staticmethod
    def send_dingtalk_msg(*_args, **_kwargs):
        return None

    @staticmethod
    def format_dingtalk_card(*_args, **_kwargs):
        return ''


class BacktestEngine:
    def __init__(self, simulator):
        self.simulator = simulator
        self.last_close_order = None

    def contracts_from_notional(self, _symbol, notional, price):
        if price <= 0:
            return 0.0
        return round(float(notional) / float(price), 6)

    def estimate_notional(self, _symbol, qty, price):
        return float(qty or 0.0) * float(price or 0.0)

    def get_symbol_info(self, _symbol):
        return {'minQty': 0.000001}

    def format_price(self, _symbol, price):
        return round(float(price or 0.0), 6)

    def execute_open_with_sltp(self, symbol, side, qty, sl, tp, leverage):
        self.simulator.pending_open = {
            'symbol': symbol,
            'side': side,
            'qty': float(qty or 0.0),
            'sl': float(sl or 0.0),
            'tp': float(tp or 0.0),
            'leverage': float(leverage or 1.0),
        }
        return True

    def cancel_all_orders(self, _symbol):
        return True

    def execute_market_close(self, symbol, current_position_side, qty):
        order_id = f'BT_CLOSE_{symbol}_{len(self.simulator.trades) + 1}'
        self.last_close_order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': current_position_side,
            'qty': float(qty or 0.0),
            'price': float(self.simulator.current_price or 0.0),
        }
        return order_id

    def get_order_realized_pnl(self, symbol, order_id):
        close = self.last_close_order or {}
        if close.get('order_id') != order_id:
            return None
        meta = self.simulator.account.real_trade_meta.get(symbol, {})
        if not meta:
            return None
        return self.simulator.realized_record(meta, close['price'], close_reason='主动反转')


class BacktestMonitor:
    def __init__(self, analyzer, simulator, initial_capital):
        self.mode = 'REAL'
        self.funds_mode = 'SHARED'
        self.enable_ai = False
        self.enable_ai_close = False
        self.analyzer = analyzer
        self.simulator = simulator
        self.accounts = []
        self.ai_cache = {}
        self.algo_cache = {}
        self.ai_cache_ttl = 0
        self.pending_close_confirmations = {}
        self.pending_m15_confirmations = {}
        self.recent_open_rejections = deque(maxlen=1000)
        self.all_trade_meta = {}
        self.initial_capital = initial_capital

    def _runtime_state_file(self):
        return str(ROOT / 'backtest_runtime_state_ignored.json')

    def _record_open_rejection(self, account_name, symbol, reason):
        self.recent_open_rejections.append({
            'time': self.simulator.current_time,
            'account_name': account_name,
            'symbol': symbol,
            'reason': reason,
        })

    def get_dashboard_overview_payload(self):
        return {}

    def _preview_realistic_rr(self, direction, data, current_price):
        account = self.accounts[0] if self.accounts else None
        resolver = account._rr_obstacle_levels if account else (lambda *_args: None)
        return _compute_realistic_rr_preview(direction, data, current_price, resolver)

    def _initial_open_net_rr(self, data):
        try:
            return float(data.get('net_expected_rr', data.get('net_rr', 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _is_duplicate_signal(self, symbol, algo_side, data, _signal_text, account_name=''):
        cache_key = f'{account_name}_{symbol}_{algo_side}'
        current_time = self.simulator.current_timestamp
        current_price = float(data.get('price') or 0.0)
        atr_pct = float(data.get('atr') or 0.0) / max(float(data.get('price') or 1.0), 1e-9)
        if cache_key in self.algo_cache:
            cache_meta = self.algo_cache[cache_key]
            last_time = float(cache_meta.get('time', 0.0) or 0.0)
            last_price = float(cache_meta.get('price', 0.0) or 0.0)
            cooldown = 120 if atr_pct > 0.03 else 300
            effective_cooldown = cooldown
            if last_price > 0:
                favorable_move = (current_price - last_price) / last_price if algo_side == 'LONG' else (last_price - current_price) / last_price
                if favorable_move > 0.003:
                    effective_cooldown = 60
            if current_time - last_time < effective_cooldown:
                return True
        self.algo_cache[cache_key] = {'time': current_time, 'price': current_price}
        return False

    def _maybe_enable_relaxed_conflict_ai_review(self, conflict_override, skip_m15_confirmation_for_strong_signal):
        if not conflict_override or not skip_m15_confirmation_for_strong_signal:
            return conflict_override
        if int(conflict_override.get('score', 0) or 0) == 1 and not conflict_override.get('passed'):
            conflict_override['passed'] = True
            conflict_override['pass_mode'] = 'micro_with_ai'
            conflict_override['scaler'] = 0.2
            conflict_override['required_ai_confidence'] = 75.0
        return conflict_override

    def _evaluate_ai_with_cache(self, _symbol, algo_side, _data, _signal_text, prompt_mode='aggressive'):
        return {'action': algo_side, 'confidence': 65.0, 'reason': 'backtest AI disabled: follow quant signal', 'leverage_factor': 1.0}


class HistoricalMarketAnalyzer(MarketAnalyzer):
    def __init__(self, history):
        super().__init__(proxies=None)
        self.history = history
        self.current_time = None
        self._indicator_cache = {}

    def get_realtime_price(self, symbol):
        frame = self._slice(symbol, '1h')
        return float(frame['close'].iloc[-1]) if frame is not None and not frame.empty else None

    def get_funding_rate(self, _symbol):
        return {'funding_rate': 0.0, 'next_funding_time': ''}

    def build_funding_sentiment_summary(self, _symbols):
        return 'backtest_no_funding'

    def _slice(self, symbol, interval):
        df = self.history.get(symbol, {}).get(interval)
        if df is None or df.empty or self.current_time is None:
            return None
        sliced = df[df['time'] <= self.current_time].tail(240).copy()
        return sliced if len(sliced) >= 60 else None

    def get_kline_indicators(self, symbol, interval):
        frame = self._slice(symbol, interval)
        if frame is None:
            return None
        cache_key = (symbol, interval, int(frame['time'].iloc[-1].value))
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        result = self._calculate_indicators_from_frame(symbol, interval, frame)
        self._indicator_cache[cache_key] = result
        return result

    def _calculate_indicators_from_frame(self, symbol, interval, df):
        try:
            df = df.copy().reset_index(drop=True)
            df[['open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote']] = df[['open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote']].astype(float)
            param_matrix = {
                '15m': {'rsi': 14, 'boll': 20, 'vol': 96},
                '1h': {'rsi': 14, 'boll': 20, 'vol': 24},
                '4h': {'rsi': 14, 'boll': 20, 'vol': 6},
                '1d': {'rsi': 10, 'boll': 14, 'vol': 5},
            }
            p = param_matrix.get(interval, {'rsi': 14, 'boll': 20, 'vol': 24})
            win_rsi, win_boll, win_vol = p['rsi'], p['boll'], p['vol']
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs(),
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(win_rsi).mean()
            plus_dm = df['high'].diff()
            minus_dm = -df['low'].diff()
            df['plus_dm'] = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
            df['minus_dm'] = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
            df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1 / win_rsi, adjust=False).mean() / (df['atr'] + 1e-9))
            df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1 / win_rsi, adjust=False).mean() / (df['atr'] + 1e-9))
            dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9)
            df['adx'] = dx.ewm(alpha=1 / win_rsi, adjust=False).mean()
            for span in [8, 20, 50, 100, 200, 12, 26]:
                df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            delta = df['close'].diff()
            rs = (delta.where(delta > 0, 0)).rolling(win_rsi).mean() / (-delta.where(delta < 0, 0)).rolling(win_rsi).mean()
            df['rsi'] = 100 - (100 / (1 + rs))
            mid = df['close'].rolling(win_boll).mean()
            std = df['close'].rolling(win_boll).std()
            df['boll_high'], df['boll_low'] = mid + 2 * std, mid - 2 * std
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['cci'] = (tp - tp.rolling(win_boll).mean()) / (0.015 * tp.rolling(win_boll).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))
            df['vol_ma24'] = df['vol'].rolling(window=win_vol).mean()
            cum_vol = df['vol'].replace(0, np.nan).cumsum()
            df['vwap'] = ((tp * df['vol']).cumsum() / cum_vol).fillna(df['close'])
            price_std = df['close'].rolling(win_boll).std().fillna(0.0)
            df['vwap_upper'] = df['vwap'] + 1.5 * price_std
            df['vwap_lower'] = df['vwap'] - 1.5 * price_std
            turnover = df['vol_ccy_quote'].fillna(0.0).astype(float)
            candle_range = (df['high'] - df['low']).replace(0, np.nan)
            signed_turnover = (((df['close'] - df['open']) / candle_range).replace([np.inf, -np.inf], np.nan).fillna(0.0) * turnover)
            turnover_ma = turnover.rolling(window=win_vol).mean().replace(0, np.nan)
            df['flow_strength'] = (signed_turnover.rolling(3).mean() / turnover_ma).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if len(df.dropna(subset=['atr', 'ema20', 'ema50', 'ema100'])) < 6:
                return None
            latest, prev, shift2, shift5 = df.iloc[-1], df.iloc[-2], df.iloc[-3], df.iloc[-6]
            recent_closed = df.iloc[-3:-1] if len(df) >= 3 else df.iloc[-2:-1]
            past_window = df.tail(22).iloc[:-2]
            min_price_idx = past_window['low'].idxmin()
            max_price_idx = past_window['high'].idxmax()
            bull_div = (latest['low'] < past_window['low'].min()) and (latest['rsi'] > df.loc[min_price_idx, 'rsi']) and (latest['rsi'] < 45)
            bear_div = (latest['high'] > past_window['high'].max()) and (latest['rsi'] < df.loc[max_price_idx, 'rsi']) and (latest['rsi'] > 55)
            sup, res = self.find_support_resistance(df, latest['close'])
            low_24h = float(df['low'].tail(min(24, len(df))).min())
            high_24h = float(df['high'].tail(min(24, len(df))).max())
            bullish_engulfing_volume = bool(latest['close'] > latest['open'] and prev['close'] < prev['open'] and latest['open'] <= prev['close'] and latest['close'] >= prev['open'] and latest['vol'] > prev['vol'])
            bearish_engulfing_volume = bool(latest['close'] < latest['open'] and prev['close'] > prev['open'] and latest['open'] >= prev['close'] and latest['close'] <= prev['open'] and latest['vol'] > prev['vol'])
            return {
                'price': latest['close'], 'prev_price': prev['close'], 'pattern': self.detect_kline_patterns(df),
                'vol_ratio': round(prev['vol'] / prev['vol_ma24'], 2) if not pd.isna(prev['vol_ma24']) and prev['vol_ma24'] > 0 else 1.0,
                'raw_klines': [f"O:{r['open']} H:{r['high']} L:{r['low']} C:{r['close']} V:{r['vol']}" for _, r in df.tail(3).iloc[::-1].iterrows()],
                'recent_candles': [{'open': float(r['open']), 'high': float(r['high']), 'low': float(r['low']), 'close': float(r['close'])} for _, r in df.tail(3).iterrows()],
                'atr': latest['atr'], 'adx': latest['adx'], 'prev_adx': prev['adx'],
                'defense_low': min(df['low'].tail(3).min(), latest['close'] - latest['atr'] * 1.5),
                'defense_high': max(df['high'].tail(3).max(), latest['close'] + latest['atr'] * 1.5),
                'cci': round(latest['cci'], 2) if not pd.isna(latest['cci']) else 0, 'rsi': latest['rsi'],
                'boll_low': latest['boll_low'], 'boll_high': latest['boll_high'],
                'ema8': latest['ema8'], 'ema20': latest['ema20'], 'ema50': latest['ema50'], 'ema100': latest['ema100'], 'ema200': latest['ema200'],
                'vwap': latest['vwap'], 'vwap_upper': latest['vwap_upper'], 'vwap_lower': latest['vwap_lower'],
                'prev_vwap_upper': prev['vwap_upper'], 'prev_vwap_lower': prev['vwap_lower'],
                'closed_above_ema20_count': int((recent_closed['close'] > recent_closed['ema20']).fillna(False).sum()),
                'closed_below_ema20_count': int((recent_closed['close'] < recent_closed['ema20']).fillna(False).sum()),
                'prev_ema8': prev['ema8'], 'prev_ema20': prev['ema20'], 'ema20_shift5': shift5['ema20'],
                'macd': latest['macd'], 'macd_signal': latest['macd_signal'], 'macd_hist': latest['macd_hist'],
                'bar_open_ms': int(pd.Timestamp(latest['time']).timestamp() * 1000),
                'prev_macd': prev['macd'], 'prev_macd_signal': prev['macd_signal'], 'prev_macd_hist': prev['macd_hist'],
                'prev_cci': prev['cci'], 'prev2_cci': shift2['cci'],
                'flow_strength': round(float(latest['flow_strength']), 4), 'prev_flow_strength': round(float(prev['flow_strength']), 4), 'prev2_flow_strength': round(float(shift2['flow_strength']), 4),
                'low_24h': low_24h, 'high_24h': high_24h,
                'bullish_engulfing_volume': bullish_engulfing_volume, 'bearish_engulfing_volume': bearish_engulfing_volume,
                'bull_div': bull_div, 'bear_div': bear_div, 'support': sup, 'resistance': res, 'df': df,
            }
        except Exception:
            return None


def symbol_to_inst_id(symbol):
    clean = symbol.upper().replace('-', '')
    return f'{clean[:-4]}-USDT-SWAP' if clean.endswith('USDT') else symbol


def download_candles(symbol, interval, bars, proxies=None):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f'{symbol}_{interval}_{bars}.pkl'
    if cache_file.exists():
        try:
            df = pd.read_pickle(cache_file)
            if len(df) >= int(bars * 0.9):
                return symbol, interval, df.tail(bars).reset_index(drop=True)
        except Exception:
            pass
    url = 'https://www.okx.com/api/v5/market/history-candles'
    after = None
    rows = []
    while len(rows) < bars:
        params = {'instId': symbol_to_inst_id(symbol), 'bar': INTERVALS[interval], 'limit': 100}
        if after is not None:
            params['after'] = after
        try:
            data = requests.get(url, params=params, proxies=proxies, timeout=12).json().get('data', [])
        except Exception:
            break
        if not data:
            break
        rows = data + rows
        after = data[-1][0]
        if len(data) < 100:
            break
        time.sleep(0.03)
    if not rows:
        return symbol, interval, pd.DataFrame()
    df = pd.DataFrame(rows[-bars:], columns=['time', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote', 'confirm'])
    df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='last').dropna(subset=['time', 'open', 'high', 'low', 'close']).reset_index(drop=True)
    df.to_pickle(cache_file)
    return symbol, interval, df


def load_history(symbols, hours, proxies=None):
    interval_bars = {
        '15m': max(hours * 4 + 260, 320),
        '1h': max(hours + 260, 320),
        '4h': max(math.ceil(hours / 4) + 260, 320),
        '1d': max(math.ceil(hours / 24) + 260, 320),
    }
    history = {symbol: {} for symbol in symbols}
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for symbol in symbols:
            for interval, bars in interval_bars.items():
                jobs.append(pool.submit(download_candles, symbol, interval, bars, proxies))
        for job in concurrent.futures.as_completed(jobs):
            symbol, interval, df = job.result()
            history[symbol][interval] = df
    return history


class LiveStrategyBacktest:
    def __init__(self, history, symbols, initial_capital, output_dir):
        self.history = history
        self.symbols = symbols
        self.initial_capital = float(initial_capital)
        self.output_dir = Path(output_dir)
        self.trades = []
        self.equity = []
        self.pending_open = None
        self.current_price = 0.0
        self.current_time = None
        self.current_timestamp = 0.0
        self.analyzer = HistoricalMarketAnalyzer(history)
        self.monitor = BacktestMonitor(self.analyzer, self, initial_capital)
        self.account = SubAccount.__new__(SubAccount)
        self.account.name = 'BacktestLiveStrategy'
        self.account.monitor = self.monitor
        self.account.mode = 'REAL'
        self.account.funds_mode = 'SHARED'
        self.account.enable_ai = False
        self.account.enable_ai_close = False
        self.account.prompt_mode = 'aggressive'
        self.account.analyzer = self.analyzer
        self.account.total_trades = 0
        self.account.win_trades = 0
        self.account.recent_closed_pnls = deque(maxlen=getattr(config, 'LOSS_TRACK_WINDOW', 10))
        self.account.cooldown_until = 0.0
        self.account.bucket_cooldowns = {}
        self.account.bucket_loss_streaks = {}
        self.account.symbol_reentry_cooldowns = {}
        self.account.real_positions = {}
        self.account.real_trade_meta = {}
        self.account.real_balance = self.initial_capital
        self.account.engine = BacktestEngine(self)
        self.monitor.accounts = [self.account]

    def spec_for(self, symbol):
        if symbol in getattr(config, 'MAINSTREAM', []):
            return dict(config.SYMBOL_CONFIG['MAINSTREAM'])
        if symbol in getattr(config, 'MID_TIER', []):
            return dict(config.SYMBOL_CONFIG['MID_TIER'])
        return dict(config.SYMBOL_CONFIG.get('MEME_SHIT', {'leverage': 3, 'margin_ratio': 0.03, 'risk_level': 7}))

    def realized_record(self, meta, close_price, close_reason):
        side = str(meta.get('side') or '').upper()
        entry = float(meta.get('entry') or 0.0)
        notional = float(meta.get('notional') or 0.0)
        slippage = float(getattr(config, 'SLIPPAGE_RATE', 0.0005) or 0.0)
        exec_price = close_price * (1 - slippage if side == 'LONG' else 1 + slippage)
        pnl_pct = (exec_price - entry) / entry * (1 if side == 'LONG' else -1) if entry > 0 else 0.0
        gross = notional * pnl_pct
        fee = -notional * float(getattr(config, 'FEE_RATE', 0.0004) or 0.0)
        return {'net_pnl': gross + fee, 'gross_pnl': gross, 'fee': fee, 'avg_price': exec_price, 'close_reason': close_reason}

    def close_position(self, symbol, close_price, close_reason):
        meta = dict(self.account.real_trade_meta.get(symbol, {}) or {})
        if not meta:
            return
        record = self.realized_record(meta, close_price, close_reason)
        net_pnl = float(record['net_pnl'])
        self.account.real_balance += net_pnl
        self.account._record_closed_trade(net_pnl, strategy_bucket=meta.get('strategy_bucket', ''))
        self.account._mark_symbol_reentry_cooldown(symbol, close_reason, now_ts=self.current_timestamp)
        self.trades.append({
            'symbol': symbol,
            'entry_time': meta.get('open_time'),
            'close_time': self.current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'side': meta.get('side'),
            'entry_price': round(float(meta.get('entry') or 0.0), 6),
            'close_price': round(float(record.get('avg_price') or close_price), 6),
            'open_reason': meta.get('open_reason', 'live_strategy_signal'),
            'close_reason': close_reason,
            'scaler': meta.get('scaler', ''),
            'market_regime': meta.get('market_regime', ''),
            'strategy_bucket': meta.get('strategy_bucket', ''),
            'notional': round(float(meta.get('notional') or 0.0), 4),
            'net_pnl': round(net_pnl, 4),
            'balance_after': round(self.account.real_balance, 4),
        })
        self.account.real_trade_meta.pop(symbol, None)
        self.account.real_positions.pop(symbol, None)
        self.monitor.all_trade_meta = dict(self.account.real_trade_meta)

    def simulate_sltp(self):
        for symbol in list(self.account.real_trade_meta.keys()):
            df = self.history[symbol]['1h']
            row = df[df['time'] == self.current_time]
            if row.empty:
                continue
            row = row.iloc[0]
            meta = self.account.real_trade_meta[symbol]
            side = meta['side']
            sl = float(meta['sl'])
            tp = float(meta['tp'])
            if side == 'LONG':
                if float(row['open']) <= sl:
                    self.close_position(symbol, float(row['open']), '止损跳空')
                elif float(row['low']) <= sl:
                    self.close_position(symbol, sl, '触发止损')
                elif float(row['open']) >= tp:
                    self.close_position(symbol, float(row['open']), '止盈跳空')
                elif float(row['high']) >= tp:
                    self.close_position(symbol, tp, '触发止盈')
            else:
                if float(row['open']) >= sl:
                    self.close_position(symbol, float(row['open']), '止损跳空')
                elif float(row['high']) >= sl:
                    self.close_position(symbol, sl, '触发止损')
                elif float(row['open']) <= tp:
                    self.close_position(symbol, float(row['open']), '止盈跳空')
                elif float(row['low']) <= tp:
                    self.close_position(symbol, tp, '触发止盈')

    def apply_pending_open_metadata(self, symbol, before_meta_keys, data):
        if not self.pending_open or self.pending_open.get('symbol') != symbol:
            return
        if symbol not in self.account.real_trade_meta or symbol in before_meta_keys:
            self.pending_open = None
            return
        meta = self.account.real_trade_meta[symbol]
        meta['open_time'] = self.current_time.strftime('%Y-%m-%d %H:%M:%S')
        meta['open_reason'] = 'live_strategy_signal'
        meta['scaler'] = round(float(meta.get('notional') or 0.0) / max(self.account.real_balance, 1e-9), 4)
        meta['risk_cluster'] = 'core' if symbol in ['BTCUSDT', 'ETHUSDT'] else 'opportunity'
        meta['market_regime'] = data.get('market_regime', meta.get('market_regime', ''))
        meta['strategy_bucket'] = data.get('strategy_bucket', meta.get('strategy_bucket', ''))
        self.monitor.all_trade_meta = dict(self.account.real_trade_meta)
        self.pending_open = None

    def run(self):
        base_times = self.history[self.symbols[0]]['1h']['time']
        min_ready_time = max(
            df['time'].iloc[min(len(df) - 1, 59)]
            for symbol in self.symbols
            for interval in INTERVALS
            for df in [self.history[symbol][interval]]
            if not df.empty
        )
        times = [t for t in base_times if t >= min_ready_time]
        for current_time in times:
            self.current_time = pd.Timestamp(current_time).to_pydatetime()
            self.current_timestamp = pd.Timestamp(current_time).timestamp()
            self.analyzer.current_time = pd.Timestamp(current_time)
            self.simulate_sltp()
            market_rows = []
            ranked_candidates = []
            for symbol in self.symbols:
                data = self.analyzer.get_market_data(symbol)
                if not data:
                    continue
                data['signal'] = self.analyzer.analyze_signals(data)
                spec = self.spec_for(symbol)
                market_rows.append((symbol, data, spec))
                candidate = _build_runtime_trade_candidate(symbol, data, spec, self.monitor.accounts)
                if candidate:
                    ranked_candidates.append(candidate)
            assignments = _allocate_trade_candidates_to_accounts(ranked_candidates, self.monitor.accounts)
            allowed_symbols = {item['symbol'] for item in assignments}
            ordered_rows = _sort_market_rows_by_candidates(market_rows, ranked_candidates, self.symbols)
            for symbol, data, spec in ordered_rows:
                if _extract_algo_side_from_signal(data.get('signal')) and symbol not in allowed_symbols and symbol not in self.account.real_positions:
                    continue
                self.current_price = float(data.get('price') or 0.0)
                before_meta_keys = set(self.account.real_trade_meta.keys())
                self.account.process_real_trading(data, spec)
                self.apply_pending_open_metadata(symbol, before_meta_keys, data)
            self.equity.append({'time': self.current_time.strftime('%Y-%m-%d %H:%M:%S'), 'equity': round(self.account.real_balance, 4), 'open_positions': len(self.account.real_positions)})
        for symbol, meta in list(self.account.real_trade_meta.items()):
            last_price = float(self.history[symbol]['1h']['close'].iloc[-1])
            self.close_position(symbol, last_price, '回测结束强制平仓')
        return self.export()

    def export(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trades = pd.DataFrame(self.trades)
        equity = pd.DataFrame(self.equity)
        if not trades.empty:
            trades.to_csv(self.output_dir / 'live_strategy_backtest_results.csv', index=False, encoding='utf-8-sig')
            by_symbol = trades.groupby('symbol').agg(
                trade_count=('net_pnl', 'count'),
                win_rate=('net_pnl', lambda s: round((s > 0).mean() * 100, 2)),
                net_pnl=('net_pnl', 'sum'),
                avg_pnl=('net_pnl', 'mean'),
                final_balance=('balance_after', 'last'),
            ).reset_index()
            by_symbol['net_pnl'] = by_symbol['net_pnl'].round(4)
            by_symbol['avg_pnl'] = by_symbol['avg_pnl'].round(4)
            by_symbol.to_csv(self.output_dir / 'live_strategy_summary_by_symbol.csv', index=False, encoding='utf-8-sig')
            for group_col, filename in [('strategy_bucket', 'live_strategy_summary_by_bucket.csv'), ('market_regime', 'live_strategy_summary_by_regime.csv')]:
                summary = trades.groupby(group_col).agg(
                    trade_count=('net_pnl', 'count'),
                    win_rate=('net_pnl', lambda s: round((s > 0).mean() * 100, 2)),
                    net_pnl=('net_pnl', 'sum'),
                    avg_pnl=('net_pnl', 'mean'),
                ).reset_index()
                summary['net_pnl'] = summary['net_pnl'].round(4)
                summary['avg_pnl'] = summary['avg_pnl'].round(4)
                summary.to_csv(self.output_dir / filename, index=False, encoding='utf-8-sig')
        else:
            pd.DataFrame(columns=['symbol', 'entry_time', 'close_time', 'side', 'entry_price', 'close_price', 'open_reason', 'close_reason', 'net_pnl', 'balance_after']).to_csv(self.output_dir / 'live_strategy_backtest_results.csv', index=False, encoding='utf-8-sig')
        equity.to_csv(self.output_dir / 'live_strategy_equity.csv', index=False, encoding='utf-8-sig')
        rejections = pd.DataFrame(list(getattr(self.monitor, 'recent_open_rejections', []) or []))
        if not rejections.empty:
            rejections.to_csv(self.output_dir / 'live_strategy_rejections.csv', index=False, encoding='utf-8-sig')
        else:
            pd.DataFrame(columns=['time', 'account_name', 'symbol', 'reason']).to_csv(self.output_dir / 'live_strategy_rejections.csv', index=False, encoding='utf-8-sig')
        net_pnl = float(trades['net_pnl'].sum()) if not trades.empty else 0.0
        wins = int((trades['net_pnl'] > 0).sum()) if not trades.empty else 0
        total = int(len(trades))
        summary = {
            'initial_capital': self.initial_capital,
            'final_balance': round(self.account.real_balance, 4),
            'net_pnl': round(net_pnl, 4),
            'return_pct': round(net_pnl / self.initial_capital * 100, 4) if self.initial_capital else 0.0,
            'trade_count': total,
            'win_rate': round(wins / total * 100, 2) if total else 0.0,
            'output_dir': str(self.output_dir),
        }
        pd.DataFrame([summary]).to_csv(self.output_dir / 'live_strategy_summary.csv', index=False, encoding='utf-8-sig')
        return summary


def main():
    parser = argparse.ArgumentParser(description='Backtest live strategy logic without real orders.')
    parser.add_argument('--capital', type=float, default=1000.0)
    parser.add_argument('--hours', type=int, default=720)
    parser.add_argument('--output-dir', default='backtest_1000u_live_strategy')
    parser.add_argument('--symbols', default=','.join(DEFAULT_SYMBOLS))
    args = parser.parse_args()
    symbols = [item.strip().upper() for item in args.symbols.split(',') if item.strip()]
    history = load_history(symbols, args.hours, proxies=getattr(config, 'OKX_PROXIES', None))
    missing = {
        symbol: [
            interval for interval in INTERVALS
            if history.get(symbol, {}).get(interval, pd.DataFrame()).empty
        ]
        for symbol in symbols
    }
    missing = {symbol: intervals for symbol, intervals in missing.items() if intervals}
    if missing:
        detail = '; '.join(f'{symbol}: {",".join(intervals)}' for symbol, intervals in missing.items())
        raise RuntimeError(f'历史K线下载失败: {detail}')
    simulator = LiveStrategyBacktest(history, symbols, args.capital, args.output_dir)
    summary = simulator.run()
    print(f"完整实盘策略逻辑回测完成 | 初始本金 {summary['initial_capital']:.2f}U | 最终余额 {summary['final_balance']:.2f}U | 净盈亏 {summary['net_pnl']:+.2f}U | 收益率 {summary['return_pct']:+.2f}% | 交易数 {summary['trade_count']} | 胜率 {summary['win_rate']:.2f}%")
    print(f"结果目录: {summary['output_dir']}")


if __name__ == '__main__':
    main()
