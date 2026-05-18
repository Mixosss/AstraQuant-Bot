import numpy as np
import pandas as pd


TRADE_ACTIONS = {'LONG', 'SHORT'}


def _copy_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def _numeric_series(df, column):
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors='coerce')


def _text_series(df, column):
    if column not in df.columns:
        return pd.Series([''] * len(df), index=df.index)
    return df[column].fillna('').astype(str)


def _trade_mask(df):
    if 'AI最终判定' not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df['AI最终判定'].isin(TRADE_ACTIONS)


def _append_unique(advice, seen, key, recommendation, severity='medium'):
    if key in seen:
        return
    seen.add(key)
    advice.append({'source': 'review', 'severity': severity, 'recommendation': recommendation})


def build_backtest_parameter_advice(df):
    work_df = _copy_frame(df)
    if work_df.empty:
        return []

    work_df['盈亏'] = _numeric_series(work_df, '盈亏')
    work_df['净RR'] = _numeric_series(work_df, '净RR')
    work_df['Shadow机会收益(%)'] = _numeric_series(work_df, 'Shadow机会收益(%)')
    work_df['Shadow结果'] = _text_series(work_df, 'Shadow结果')
    work_df['_trade'] = _trade_mask(work_df)
    work_df['_win'] = work_df['_trade'] & (work_df['盈亏'].fillna(0) > 0)

    advice = []
    seen = set()

    if '交易对' in work_df.columns:
        for symbol, group in work_df.groupby('交易对', dropna=True):
            symbol_name = str(symbol or '').strip()
            trades = group[group['_trade']]
            if not symbol_name or len(trades) < 3:
                continue
            pnl = float(trades['盈亏'].fillna(0).sum())
            win_rate = float((trades['盈亏'].fillna(0) > 0).mean() * 100)
            if pnl < 0 or win_rate < 35:
                _append_unique(
                    advice,
                    seen,
                    f'symbol:{symbol_name}',
                    f'币种 {symbol_name} 回测/复盘表现偏弱，建议降低 position_scaler 或提高该币净RR门槛。',
                    'high',
                )
                break

    if '策略桶' in work_df.columns:
        for bucket, group in work_df.groupby('策略桶', dropna=True):
            bucket_name = str(bucket or '').strip()
            trades = group[group['_trade']]
            if not bucket_name or len(trades) < 3:
                continue
            pnl = float(trades['盈亏'].fillna(0).sum())
            win_rate = float((trades['盈亏'].fillna(0) > 0).mean() * 100)
            if pnl < 0 or win_rate < 35:
                _append_unique(
                    advice,
                    seen,
                    f'bucket:{bucket_name}',
                    f'策略桶 {bucket_name} 回测/复盘胜率偏弱，建议提高该桶 min_net_rr 或降低 max_scaler。',
                    'high',
                )
                break

    if '市场状态' in work_df.columns:
        for regime, group in work_df.groupby('市场状态', dropna=True):
            regime_name = str(regime or '').strip()
            trades = group[group['_trade']]
            if not regime_name or len(trades) < 3:
                continue
            pnl = float(trades['盈亏'].fillna(0).sum())
            win_rate = float((trades['盈亏'].fillna(0) > 0).mean() * 100)
            if pnl < 0 or win_rate < 35:
                _append_unique(
                    advice,
                    seen,
                    f'regime:{regime_name}',
                    f'市场状态 {regime_name} 回测/复盘表现偏弱，建议降低该 regime 仓位系数。',
                    'medium',
                )
                break

    if '策略桶' in work_df.columns:
        shadow_df = work_df[work_df['Shadow结果'].isin(['错过机会', '避免亏损'])]
        for bucket, group in shadow_df.groupby('策略桶', dropna=True):
            bucket_name = str(bucket or '').strip()
            if not bucket_name or len(group) < 3:
                continue
            missed_rate = float((group['Shadow结果'] == '错过机会').mean() * 100)
            avg_return = float(group['Shadow机会收益(%)'].dropna().mean()) if not group['Shadow机会收益(%)'].dropna().empty else 0.0
            if missed_rate >= 60 and avg_return > 0:
                _append_unique(
                    advice,
                    seen,
                    f'shadow_bucket:{bucket_name}',
                    f'策略桶 {bucket_name} Shadow 错过率偏高且平均机会为正，建议复查 AI/RR 门槛是否过严。',
                    'medium',
                )
                break

    return advice[:8]


def _confidence_band(value):
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.0
    if confidence < 60:
        return '<60'
    if confidence < 70:
        return '60-69'
    if confidence < 80:
        return '70-79'
    return '80+'


def summarize_ai_shadow_quality(df):
    work_df = _copy_frame(df)
    if work_df.empty or 'AI最终判定' not in work_df.columns:
        return []

    work_df['AI置信度'] = _numeric_series(work_df, 'AI置信度')
    work_df['盈亏'] = _numeric_series(work_df, '盈亏')
    work_df['Shadow机会收益(%)'] = _numeric_series(work_df, 'Shadow机会收益(%)')
    work_df['Shadow结果'] = _text_series(work_df, 'Shadow结果')
    work_df['_confidence_band'] = work_df['AI置信度'].apply(_confidence_band)
    work_df['_approved'] = _trade_mask(work_df)
    work_df['_pass'] = work_df['AI最终判定'] == 'PASS'
    work_df['_approved_win'] = work_df['_approved'] & (work_df['盈亏'].fillna(0) > 0)

    rows = []
    for band, group in work_df.groupby(['_confidence_band'], dropna=True):
        band_name = band[0] if isinstance(band, tuple) else band
        approved_count = int(group['_approved'].sum())
        pass_count = int(group['_pass'].sum())
        shadow_group = group[group['Shadow结果'].isin(['错过机会', '避免亏损'])]
        missed_count = int((shadow_group['Shadow结果'] == '错过机会').sum())
        avoided_count = int((shadow_group['Shadow结果'] == '避免亏损').sum())
        rows.append({
            'confidence_band': str(band_name),
            'sample_count': int(len(group)),
            'approved_count': approved_count,
            'pass_count': pass_count,
            'approved_win_rate': round(float(group['_approved_win'].sum()) / approved_count * 100, 2) if approved_count else 0.0,
            'shadow_missed_rate': round(missed_count / len(shadow_group) * 100, 2) if len(shadow_group) else 0.0,
            'shadow_avoided_rate': round(avoided_count / len(shadow_group) * 100, 2) if len(shadow_group) else 0.0,
            'avg_shadow_return_pct': round(float(shadow_group['Shadow机会收益(%)'].dropna().mean()) if not shadow_group['Shadow机会收益(%)'].dropna().empty else 0.0, 2),
        })
    return sorted(rows, key=lambda row: row['confidence_band'])

def build_ai_shadow_recommendations(rows):
    recommendations = []
    for row in rows or []:
        samples = int(row.get('sample_count') or 0)
        approved = int(row.get('approved_count') or 0)
        passed = int(row.get('pass_count') or 0)
        missed_rate = float(row.get('shadow_missed_rate') or 0.0)
        win_rate = float(row.get('approved_win_rate') or 0.0)
        avg_shadow = float(row.get('avg_shadow_return_pct') or 0.0)
        band = row.get('confidence_band')
        if samples >= 4 and approved >= 2 and passed >= 2 and win_rate >= 55 and missed_rate >= 60 and avg_shadow > 0:
            recommendations.append(f'AI置信度 {band} 放行胜率尚可但 PASS 错过率偏高，建议复查该分段 AI 阈值是否过严。')
        elif samples >= 4 and passed >= 2 and missed_rate <= 25 and avg_shadow <= 0:
            recommendations.append(f'AI置信度 {band} PASS 多数避免亏损，建议该分段暂不放宽。')
    return recommendations[:5]


def build_live_ai_review_rows(df):
    work_df = _copy_frame(df)
    if work_df.empty or 'AI最终判定' not in work_df.columns:
        return []

    work_df['AI置信度'] = _numeric_series(work_df, 'AI置信度')
    work_df['盈亏'] = _numeric_series(work_df, '盈亏')
    work_df['Shadow机会收益(%)'] = _numeric_series(work_df, 'Shadow机会收益(%)')
    work_df['AI仓位档位'] = _text_series(work_df, 'AI仓位档位').replace('', '未标记')
    work_df['策略桶'] = _text_series(work_df, '策略桶').replace('', '未标记')
    work_df['交易对'] = _text_series(work_df, '交易对').replace('', '未标记')
    work_df['Shadow结果'] = _text_series(work_df, 'Shadow结果')
    work_df['_confidence_band'] = work_df['AI置信度'].apply(_confidence_band)
    work_df['_approved'] = _trade_mask(work_df)
    work_df['_pass'] = work_df['AI最终判定'] == 'PASS'
    work_df['_approved_win'] = work_df['_approved'] & (work_df['盈亏'].fillna(0) > 0)
    work_df['_shadow_missed'] = work_df['Shadow结果'] == '错过机会'
    work_df['_shadow_avoided'] = work_df['Shadow结果'] == '避免亏损'

    rows = []

    def add_rows(scope, column):
        for value, group in work_df.groupby(column, dropna=True):
            name = str(value or '').strip()
            if not name or name == '未标记':
                continue
            approved_count = int(group['_approved'].sum())
            pass_count = int(group['_pass'].sum())
            shadow_group = group[group['_shadow_missed'] | group['_shadow_avoided']]
            missed_count = int(group['_shadow_missed'].sum())
            avoided_count = int(group['_shadow_avoided'].sum())
            rows.append({
                'scope': scope,
                'name': name,
                'sample_count': int(len(group)),
                'approved_count': approved_count,
                'pass_count': pass_count,
                'win_rate': round(float(group['_approved_win'].sum()) / approved_count * 100, 2) if approved_count else 0.0,
                'realized_pnl': round(float(group.loc[group['_approved'], '盈亏'].fillna(0).sum()), 2) if approved_count else 0.0,
                'shadow_missed_count': missed_count,
                'shadow_avoided_count': avoided_count,
                'shadow_missed_rate': round(missed_count / len(shadow_group) * 100, 2) if len(shadow_group) else 0.0,
                'shadow_avoided_rate': round(avoided_count / len(shadow_group) * 100, 2) if len(shadow_group) else 0.0,
                'shadow_avg_return_pct': round(float(shadow_group['Shadow机会收益(%)'].dropna().mean()) if not shadow_group['Shadow机会收益(%)'].dropna().empty else 0.0, 2),
            })

    add_rows('entry_label', 'AI仓位档位')
    add_rows('confidence', '_confidence_band')
    add_rows('bucket', '策略桶')
    add_rows('symbol', '交易对')
    return sorted(rows, key=lambda row: (row['scope'], -row['sample_count'], row['name']))


def build_live_ai_review_recommendations(rows):
    recommendations = []
    for row in rows or []:
        scope = row.get('scope')
        name = row.get('name')
        sample_count = int(row.get('sample_count') or 0)
        approved_count = int(row.get('approved_count') or 0)
        pass_count = int(row.get('pass_count') or 0)
        win_rate = float(row.get('win_rate') or 0.0)
        pnl = float(row.get('realized_pnl') or 0.0)
        missed_rate = float(row.get('shadow_missed_rate') or 0.0)
        avg_shadow = float(row.get('shadow_avg_return_pct') or 0.0)
        if scope == 'entry_label' and sample_count >= 3 and approved_count >= 3 and (pnl < 0 or win_rate < 35):
            recommendations.append(f'{name} 实盘放行表现偏弱，建议降低该档位仓位或提高 AI 置信度/RR 门槛。')
        elif scope == 'confidence' and sample_count >= 4 and pass_count >= 2 and missed_rate >= 60 and avg_shadow > 0:
            recommendations.append(f'AI置信度 {name} PASS 错过机会偏多，建议复查该置信度分段是否过严。')
        elif scope in {'bucket', 'symbol'} and sample_count >= 4 and approved_count >= 3 and pnl < 0:
            label = '策略桶' if scope == 'bucket' else '币种'
            recommendations.append(f'{label} {name} 的 AI 放行后实盘 PnL 偏弱，建议降低仓位或提高净RR门槛。')
    return recommendations[:6]
