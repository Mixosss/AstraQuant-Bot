import concurrent.futures
import logging
import os
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import requests

logging.getLogger().setLevel(logging.ERROR)

CACHE_DIR = 'Cache_MarketData'
os.makedirs(CACHE_DIR, exist_ok=True)


def symbol_to_inst_id(symbol):
    clean = symbol.upper().replace('-', '')
    return f"{clean[:-4]}-USDT-SWAP" if clean.endswith('USDT') else symbol


def inst_id_to_symbol(inst_id):
    return inst_id.upper().replace('-USDT-SWAP', 'USDT').replace('-', '')


def get_top_usdt_futures_symbols(top_n=50):
    print(f'正在获取 OKX 全市场 Top {top_n} 活跃永续合约...')
    try:
        info = requests.get('https://www.okx.com/api/v5/public/instruments', params={'instType': 'SWAP'}, timeout=10).json().get('data', [])
        tickers = requests.get('https://www.okx.com/api/v5/market/tickers', params={'instType': 'SWAP'}, timeout=10).json().get('data', [])
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
    except Exception:
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']


def download_worker(symbol, limit):
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
            data = requests.get(url, params=params, timeout=10).json().get('data', [])
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
    cols = ['open', 'high', 'low', 'close', 'vol']
    df[cols] = df[cols].astype(float)
    df['taker_base_vol'] = df['vol'] * 0.5
    df.to_pickle(cache_file)
    return symbol, df


def apply_classic_indicators(df):
    if df.empty:
        return df
    df['body'] = (df['close'] - df['open']).abs()
    df['tot'] = df['high'] - df['low']
    df['up_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['dn_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['vol_ma24'] = df['vol'].rolling(window=24).mean()
    df['v1h'] = df['vol'] / df['vol_ma24'].shift(1)
    df['taker_sell_vol'] = df['vol'] - df['taker_base_vol']
    df['r1h'] = df['taker_base_vol'] / (df['taker_sell_vol'] + 1e-8)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['defense_low'] = df['low'].rolling(3).min()
    df['defense_high'] = df['high'].rolling(3).max()
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['c1h'] = (tp - sma) / (0.015 * mad)
    df.set_index('time', inplace=True)
    df_4h = df.resample('4h').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    tp_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    df_4h['c4h'] = (tp_4h - tp_4h.rolling(20).mean()) / (0.015 * tp_4h.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))
    df_4h['ema8_4h'] = df_4h['close'].ewm(span=8, adjust=False).mean()
    df_4h['ema20_4h'] = df_4h['close'].ewm(span=20, adjust=False).mean()
    df_4h['ema50_4h'] = df_4h['close'].ewm(span=50, adjust=False).mean()
    df_4h['macd_4h'] = df_4h['close'].ewm(span=12).mean() - df_4h['close'].ewm(span=26).mean()
    df_4h['macd_hist_4h'] = df_4h['macd_4h'] - df_4h['macd_4h'].ewm(span=9).mean()
    df_1d = df.resample('1D').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    df_1d['c1d'] = (tp_1d - tp_1d.rolling(20).mean()) / (0.015 * tp_1d.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True))
    df = df.join(df_4h[['c4h', 'ema8_4h', 'ema20_4h', 'ema50_4h', 'macd_hist_4h']]).ffill()
    df = df.join(df_1d[['c1d']]).ffill()
    df.reset_index(inplace=True)
    return df.dropna().reset_index(drop=True)


def run_classic_backtest(df, symbol, initial_capital=10000.0, margin_ratio=0.1, leverage=10, rr_ratio=2.0):
    capital = initial_capital
    positions = []
    equity_curve = []
    trades = []
    commission_rate = 0.0004

    for i in range(20, len(df)):
        c = df.iloc[i]
        p = df.iloc[i - 1]
        equity_curve.append({'time': c['time'], 'equity': capital, 'price': c['close']})

        if positions:
            pos = positions[0]
            close_reason, exec_price = '', None
            if pos['side'] == 'LONG':
                if c['low'] <= pos['sl']:
                    exec_price, close_reason = pos['sl'], '防守止损'
                elif c['high'] >= pos['tp']:
                    exec_price, close_reason = pos['tp'], '触发止盈'
            else:
                if c['high'] >= pos['sl']:
                    exec_price, close_reason = pos['sl'], '防守止损'
                elif c['low'] <= pos['tp']:
                    exec_price, close_reason = pos['tp'], '触发止盈'
            if exec_price is not None:
                dynamic_slippage = c['atr'] * 0.1
                exit_price = exec_price - dynamic_slippage if pos['side'] == 'LONG' else exec_price + dynamic_slippage
                pnl_pct = (exit_price - pos['entry']) / pos['entry'] * (1 if pos['side'] == 'LONG' else -1)
                net_pnl = (pos['notional'] * pnl_pct) - (pos['notional'] * commission_rate)
                capital += net_pnl
                trades.append({
                    'symbol': symbol,
                    'entry_time': pos['entry_time'],
                    'close_time': c['time'],
                    'side': pos['side'],
                    'entry_price': round(pos['entry'], 5),
                    'close_price': round(exit_price, 5),
                    'open_reason': pos['reason'],
                    'close_reason': close_reason,
                    'net_pnl': round(net_pnl, 2),
                    'balance_after': round(capital, 2),
                })
                positions.pop()
            continue

        open_side = None
        open_reason = ''
        c1h, c4h, c1d = c['c1h'], c['c4h'], c['c1d']
        r1h, v1h = c['r1h'], c['v1h']
        has_lower_shadow = c['dn_shadow'] >= 2 * max(c['body'], 1e-9) and c['dn_shadow'] >= 0.5 * c['tot']
        is_doji = c['body'] <= 0.1 * c['tot'] if c['tot'] > 0 else False
        ema_dead_cross_1h = p['ema8'] >= p['ema20'] and c['ema8'] < c['ema20']
        ema_short_strong_1h = c['close'] < c['ema8'] and c['close'] < c['ema20'] and c['close'] < c['ema50']
        ema_long_strong_1h = c['close'] > c['ema8'] and c['close'] > c['ema20'] and c['close'] > c['ema50']
        ema_long_strong_4h = c['close'] > c['ema8_4h'] and c['close'] > c['ema20_4h'] and c['close'] > c['ema50_4h']
        ema_short_strong_4h = c['close'] < c['ema8_4h'] and c['close'] < c['ema20_4h'] and c['close'] < c['ema50_4h']

        if c1h <= -200 and (ema_dead_cross_1h or ema_short_strong_1h):
            if r1h >= 1.2 and v1h >= 1.5 and (has_lower_shadow or is_doji):
                open_side, open_reason = 'LONG', 'left_reversal'
        elif c1d < -100 and c4h < -100 and c1h < -170 and r1h > 1.05:
            open_side, open_reason = 'LONG', 'oversold_resonance'
        elif c1d > 100 and c4h > 100 and c1h > 170 and r1h < 0.95:
            open_side, open_reason = 'SHORT', 'overbought_resonance'
        elif ema_long_strong_1h and c['macd_hist'] > 0 and ema_long_strong_4h and c['macd_hist_4h'] > 0:
            open_side, open_reason = 'LONG', 'trend_follow_long'
        elif ema_short_strong_1h and c['macd_hist'] < 0 and ema_short_strong_4h and c['macd_hist_4h'] < 0:
            open_side, open_reason = 'SHORT', 'trend_follow_short'

        if not open_side:
            continue

        margin_used = capital * margin_ratio
        notional = margin_used * leverage
        entry = c['close']
        risk = c['atr'] * 1.5 if c['atr'] > 0 else c['close'] * 0.02
        sl = entry - risk if open_side == 'LONG' else entry + risk
        tp = entry + risk * rr_ratio if open_side == 'LONG' else entry - risk * rr_ratio
        capital -= notional * commission_rate
        positions.append({
            'side': open_side,
            'entry': entry,
            'sl': sl,
            'tp': tp,
            'notional': notional,
            'entry_time': c['time'],
            'reason': open_reason,
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def run_batch_backtest(symbols=None, limit=1500, top_n=20):
    if not symbols:
        symbols = get_top_usdt_futures_symbols(top_n=top_n)
    market = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(download_worker, symbol, limit) for symbol in symbols]
        for future in concurrent.futures.as_completed(futures):
            symbol, df = future.result()
            if not df.empty:
                market[symbol] = apply_classic_indicators(df)
    results = {}
    for symbol, df in market.items():
        trades, equity = run_classic_backtest(df, symbol)
        results[symbol] = {'trades': trades, 'equity': equity}
    return results


if __name__ == '__main__':
    results = run_batch_backtest()
    print(f'经典回测完成，共处理 {len(results)} 个 OKX 合约。')
