import concurrent.futures
import logging
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logging.getLogger().setLevel(logging.ERROR)

CACHE_DIR = 'Cache_MarketData'
os.makedirs(CACHE_DIR, exist_ok=True)
DEFAULT_BACKTEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']


def symbol_to_inst_id(symbol):
    clean = symbol.upper().replace('-', '')
    return f"{clean[:-4]}-USDT-SWAP" if clean.endswith('USDT') else symbol


def inst_id_to_symbol(inst_id):
    return inst_id.upper().replace('-USDT-SWAP', 'USDT').replace('-', '')


def get_top_usdt_futures_symbols(top_n=50, proxies=None):
    print(f'正在请求 OKX 服务器，获取全市场 Top {top_n} 活跃永续合约...')
    try:
        info = requests.get(
            'https://www.okx.com/api/v5/public/instruments',
            params={'instType': 'SWAP'},
            proxies=proxies,
            timeout=10,
        ).json().get('data', [])
        tickers = requests.get(
            'https://www.okx.com/api/v5/market/tickers',
            params={'instType': 'SWAP'},
            proxies=proxies,
            timeout=10,
        ).json().get('data', [])
        alive = {
            inst_id_to_symbol(row['instId'])
            for row in info
            if row.get('settleCcy') == 'USDT' and row.get('state') == 'live'
        }
        rows = []
        for row in tickers:
            symbol = inst_id_to_symbol(row['instId'])
            vol = float(row.get('volCcy24h') or 0)
            if symbol in alive and vol > 0:
                rows.append((symbol, vol))
        rows.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in rows[:top_n]]
    except Exception as e:
        print(f'获取 OKX 合约列表失败: {e}')
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']


def download_worker(symbol, limit, proxies):
    cache_file = os.path.join(CACHE_DIR, f'{symbol}_{limit}H.pkl')
    if os.path.exists(cache_file):
        try:
            df = pd.read_pickle(cache_file)
            if len(df) >= limit * 0.95:
                return symbol, df
        except Exception:
            pass

    inst_id = symbol_to_inst_id(symbol)
    before = None
    all_rows = []
    url = 'https://www.okx.com/api/v5/market/history-candles'

    while len(all_rows) < limit:
        params = {'instId': inst_id, 'bar': '1H', 'limit': 100}
        if before is not None:
            params['before'] = before
        try:
            data = requests.get(url, params=params, proxies=proxies, timeout=10).json().get('data', [])
        except Exception:
            break
        if not data:
            break
        all_rows = data + all_rows
        before = data[0][0]
        if len(data) < 100:
            break
        time.sleep(0.05)

    if not all_rows:
        return symbol, pd.DataFrame()

    df = pd.DataFrame(all_rows[-limit:], columns=['time', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote', 'confirm'])
    df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
    df[['open', 'high', 'low', 'close', 'vol']] = df[['open', 'high', 'low', 'close', 'vol']].astype(float)
    df = df.sort_values('time').drop_duplicates(subset=['time'], keep='last').reset_index(drop=True)
    df.to_pickle(cache_file)
    return symbol, df


def apply_strategy_indicators(df):
    if df.empty:
        return df

    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    df.set_index('time', inplace=True)
    df_4h = df['close'].resample('4h').last().to_frame(name='close_4h')
    df_4h['ema20_4h'] = df_4h['close_4h'].ewm(span=20, adjust=False).mean()
    df_4h['ema50_4h'] = df_4h['close_4h'].ewm(span=50, adjust=False).mean()

    df_1d = df['close'].resample('1D').last().to_frame(name='close_1d')
    df_1d['ema20_1d'] = df_1d['close_1d'].ewm(span=20, adjust=False).mean()
    df_1d['ema50_1d'] = df_1d['close_1d'].ewm(span=50, adjust=False).mean()

    df = df.join(df_4h[['ema20_4h', 'ema50_4h']]).ffill()
    df = df.join(df_1d[['ema20_1d', 'ema50_1d']]).ffill()
    df.reset_index(inplace=True)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    mid = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['boll_high'] = mid + 2 * std
    df['boll_low'] = mid - 2 * std

    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['cci'] = (tp - sma) / (0.015 * mad)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['ema20_slope_atr'] = (df['ema20'] - df['ema20'].shift(5)) / df['atr']
    df['vol_ma24'] = df['vol'].rolling(window=24).mean()
    df['vol_ratio'] = df['vol'] / df['vol_ma24'].shift(1)

    rolling_low_min = df['low'].rolling(20).min().shift(1)
    rolling_rsi_min = df['rsi'].rolling(20).min().shift(1)
    rolling_high_max = df['high'].rolling(20).max().shift(1)
    rolling_rsi_max = df['rsi'].rolling(20).max().shift(1)
    df['bull_div'] = (df['low'] < rolling_low_min) & (df['rsi'] > rolling_rsi_min) & (df['rsi'] < 40)
    df['bear_div'] = (df['high'] > rolling_high_max) & (df['rsi'] < rolling_rsi_max) & (df['rsi'] > 60)
    return df.dropna().reset_index(drop=True)


def _classify_backtest_market_regime(row):
    vol_ratio = float(row.get('vol_ratio') or 0.0)
    slope = abs(float(row.get('ema20_slope_atr') or 0.0))
    if vol_ratio >= 1.8:
        return 'high_vol_range'
    if slope >= 0.5:
        return 'strong_trend'
    return 'neutral_range'


def _classify_backtest_strategy_bucket(row, side):
    cci = float(row.get('cci') or 0.0)
    trend_state = str(row.get('trend_state') or '')
    if side == 'LONG' and trend_state == 'long':
        return 'trend_continuation'
    if side == 'SHORT' and trend_state == 'short':
        return 'trend_continuation'
    if abs(cci) >= 150 or bool(row.get('bull_div')) or bool(row.get('bear_div')):
        return 'mean_reversion'
    return 'reversal_probe'


def run_backtest(df, symbol, initial_capital=10000.0, margin_ratio=0.1, leverage=10, rr_ratio=2.0, min_score=4.5):
    capital = initial_capital
    positions = []
    equity_curve = []
    trades = []
    commission_rate = 0.0004
    funding_rate = 0.0001

    for i in range(20, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i - 1]
        equity_curve.append({'time': current['time'], 'equity': capital, 'price': current['close']})

        if positions and current['time'].hour in [0, 8, 16]:
            capital -= positions[0]['notional'] * funding_rate

        if positions:
            pos = positions[0]
            close_reason = ''
            if pos['side'] == 'LONG':
                if current['open'] <= pos['sl']:
                    exec_price, close_reason = current['open'], '止损跳空'
                elif current['low'] <= pos['sl']:
                    exec_price, close_reason = pos['sl'], '触发止损'
                elif current['open'] >= pos['tp']:
                    exec_price, close_reason = current['open'], '止盈跳空'
                elif current['high'] >= pos['tp']:
                    exec_price, close_reason = pos['tp'], '触发止盈'
                else:
                    exec_price = None
            else:
                if current['open'] >= pos['sl']:
                    exec_price, close_reason = current['open'], '止损跳空'
                elif current['high'] >= pos['sl']:
                    exec_price, close_reason = pos['sl'], '触发止损'
                elif current['open'] <= pos['tp']:
                    exec_price, close_reason = current['open'], '止盈跳空'
                elif current['low'] <= pos['tp']:
                    exec_price, close_reason = pos['tp'], '触发止盈'
                else:
                    exec_price = None
            if exec_price is not None:
                dynamic_slippage = current['atr'] * 0.1
                exit_price = exec_price - dynamic_slippage if pos['side'] == 'LONG' else exec_price + dynamic_slippage
                pnl_pct = (exit_price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'LONG' else -1)
                net_pnl = (pos['notional'] * pnl_pct) - (pos['notional'] * commission_rate)
                capital += net_pnl
                trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'close_time': current['time'],
                    'side': pos['side'],
                    'entry_price': round(pos['entry'], 5),
                    'close_price': round(exit_price, 5),
                    'open_reason': pos['open_reason'],
                    'close_reason': close_reason,
                    'scaler': f"{pos['scaler']}x",
                    'market_regime': pos.get('market_regime', ''),
                    'strategy_bucket': pos.get('strategy_bucket', ''),
                    'net_pnl': round(net_pnl, 2),
                    'balance_after': round(capital, 2),
                })
                positions.pop()
                continue

        trend_1d = 1 if current['ema20_1d'] > current['ema50_1d'] else -1
        trend_4h = 1 if current['ema20_4h'] > current['ema50_4h'] else -1
        trend_1h = 1 if current['ema20'] > current['ema50'] else -1
        score_trend = trend_1d * 2 + trend_4h + trend_1h
        slope_4h = current['ema20_slope_atr']
        trend_state = 'long' if score_trend >= 2 and abs(slope_4h) > 0.5 else ('short' if score_trend <= -2 and abs(slope_4h) > 0.5 else 'neutral')
        current = current.copy()
        current['trend_state'] = trend_state

        ls, ss = 0.0, 0.0
        p = current['close']
        if current['bull_div']:
            ls += 2.0
        if current['bear_div']:
            ss += 2.0
        if p > current['ema8']:
            ls += 1.0
        if p < current['ema8']:
            ss += 1.0
        if p > current['ema20'] and p > current['ema50']:
            ls += 1.5
        if p < current['ema20'] and p < current['ema50']:
            ss += 1.5
        if trend_state == 'long':
            ls += 1.5
        if trend_state == 'short':
            ss += 1.5
        if current['cci'] < -150:
            ls += 1.0
        if current['cci'] > 150:
            ss += 1.0
        if p <= current['boll_low']:
            ls += 1.0
        if current['rsi'] < 35:
            ls += 1.0
        if p >= current['boll_high']:
            ss += 1.0
        if current['rsi'] > 65:
            ss += 1.0
        if current['vol_ratio'] >= 1.2:
            ls += 0.5
            ss += 0.5

        open_side = None
        if ls >= min_score and ls > ss:
            open_side = 'LONG'
        elif ss >= min_score and ss > ls:
            open_side = 'SHORT'
        if not open_side:
            continue

        if capital <= 0:
            break
        margin_used = capital * margin_ratio
        notional = margin_used * leverage
        entry = current['close']
        risk = current['atr'] * 1.5 if current['atr'] > 0 else current['close'] * 0.02
        sl = entry - risk if open_side == 'LONG' else entry + risk
        tp = entry + risk * rr_ratio if open_side == 'LONG' else entry - risk * rr_ratio
        capital -= notional * commission_rate
        positions.append({
            'side': open_side,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'notional': notional,
            'entry_time': current['time'],
            'open_reason': 'score_signal',
            'scaler': 1.0,
            'market_regime': _classify_backtest_market_regime(current),
            'strategy_bucket': _classify_backtest_strategy_bucket(current, open_side),
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def run_batch_backtest(symbols=None, limit=1500, top_n=20, proxies=None):
    if symbols is None:
        symbols = list(DEFAULT_BACKTEST_SYMBOLS)
    market = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(download_worker, symbol, limit, proxies) for symbol in symbols]
        for future in concurrent.futures.as_completed(futures):
            symbol, df = future.result()
            if not df.empty:
                market[symbol] = apply_strategy_indicators(df)
    results = {}
    for symbol, df in market.items():
        trades, equity = run_backtest(df, symbol)
        results[symbol] = {'trades': trades, 'equity': equity}
    return results


def _flatten_backtest_trades(results):
    frames = []
    for symbol, result in (results or {}).items():
        trades = result.get('trades')
        if trades is None or trades.empty:
            continue
        frame = trades.copy()
        if 'symbol' not in frame.columns:
            frame['symbol'] = symbol
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _validate_backtest_trades(trades):
    if trades is None or trades.empty:
        return
    if 'entry_time' not in trades.columns or 'close_time' not in trades.columns:
        return
    entry_times = pd.to_datetime(trades['entry_time'], errors='coerce')
    close_times = pd.to_datetime(trades['close_time'], errors='coerce')
    invalid = close_times < entry_times
    if invalid.any():
        bad = trades.loc[invalid].iloc[0]
        raise ValueError(
            f"close_time earlier than entry_time for {bad.get('symbol', 'UNKNOWN')}: "
            f"{bad.get('entry_time')} -> {bad.get('close_time')}"
        )


def _max_drawdown_from_equity(equity):
    if equity is None or equity.empty or 'equity' not in equity.columns:
        return 0.0
    series = pd.to_numeric(equity['equity'], errors='coerce').dropna()
    if series.empty:
        return 0.0
    running_peak = series.cummax()
    drawdown = (series - running_peak) / running_peak.replace(0, np.nan) * 100.0
    return round(float(drawdown.min()) if not drawdown.dropna().empty else 0.0, 2)


def _summarize_backtest_group(trades, group_col, results=None):
    if trades is None or trades.empty or group_col not in trades.columns:
        return pd.DataFrame(columns=[group_col, 'trade_count', 'win_rate', 'net_pnl', 'avg_pnl', 'profit_factor'])
    work = trades.copy()
    work['net_pnl'] = pd.to_numeric(work['net_pnl'], errors='coerce').fillna(0.0)
    rows = []
    for key, group in work.groupby(group_col, dropna=True):
        name = str(key or '').strip()
        if not name:
            continue
        wins = group[group['net_pnl'] > 0]
        losses = group[group['net_pnl'] < 0]
        gross_profit = float(wins['net_pnl'].sum())
        gross_loss = abs(float(losses['net_pnl'].sum()))
        row = {
            group_col: name,
            'trade_count': int(len(group)),
            'win_rate': round((len(wins) / len(group)) * 100.0 if len(group) else 0.0, 2),
            'net_pnl': round(float(group['net_pnl'].sum()), 2),
            'avg_pnl': round(float(group['net_pnl'].mean()) if len(group) else 0.0, 2),
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 0.0),
        }
        if group_col == 'symbol' and results:
            equity = (results.get(name) or {}).get('equity')
            row['max_drawdown'] = _max_drawdown_from_equity(equity)
            if equity is not None and not equity.empty and 'equity' in equity.columns:
                equity_series = pd.to_numeric(equity['equity'], errors='coerce').dropna()
                row['final_equity'] = round(float(equity_series.iloc[-1]), 2) if not equity_series.empty else 0.0
            else:
                row['final_equity'] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['net_pnl', 'trade_count'], ascending=[True, False]).reset_index(drop=True)


def export_backtest_results(results, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    trades = _flatten_backtest_trades(results)
    _validate_backtest_trades(trades)
    result_path = os.path.join(output_dir, 'backtest_results.csv')
    symbol_path = os.path.join(output_dir, 'backtest_summary_by_symbol.csv')
    regime_path = os.path.join(output_dir, 'backtest_summary_by_regime.csv')
    bucket_path = os.path.join(output_dir, 'backtest_summary_by_bucket.csv')

    trades.to_csv(result_path, index=False, encoding='utf-8-sig')
    _summarize_backtest_group(trades, 'symbol', results=results).to_csv(symbol_path, index=False, encoding='utf-8-sig')
    _summarize_backtest_group(trades, 'market_regime').to_csv(regime_path, index=False, encoding='utf-8-sig')
    _summarize_backtest_group(trades, 'strategy_bucket').to_csv(bucket_path, index=False, encoding='utf-8-sig')
    return {
        'results': result_path,
        'summary_by_symbol': symbol_path,
        'summary_by_regime': regime_path,
        'summary_by_bucket': bucket_path,
    }


if __name__ == '__main__':
    results = run_batch_backtest()
    paths = export_backtest_results(results)
    print(f'回测完成，共处理 {len(results)} 个 OKX 合约。')
    print('已输出回测文件:')
    for path in paths.values():
        print(f'- {path}')
