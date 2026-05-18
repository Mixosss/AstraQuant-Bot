import logging
import time

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    def __init__(self, proxies=None):
        self.proxies = proxies
        self.base_urls = ["https://www.okx.com"]
        self.current_url_index = 0
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        if proxies:
            self.session.proxies.update(proxies)
        self.funding_rate_cache = {}
        self.funding_cache_ttl = int(getattr(config, 'FUNDING_RATE_CACHE_TTL', 600))

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass

    def _symbol_to_inst_id(self, symbol):
        clean = symbol.upper().replace("-", "")
        return f"{clean[:-4]}-USDT-SWAP" if clean.endswith("USDT") else symbol

    def _okx_bar(self, interval):
        return {
            "15m": "15m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }.get(interval, interval)

    def _interval_ms(self, interval):
        return {
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }.get(interval, 60 * 60 * 1000)

    def get_realtime_price(self, symbol):
        try:
            params = {"instId": self._symbol_to_inst_id(symbol)}
            api_timeout = getattr(config, 'API_TIMEOUT', 15)
            response = self._request_with_fallback("/api/v5/market/ticker", params, api_timeout, symbol, "实时价格")
            rows = response.json().get("data", [])
            if not rows:
                return None
            last_price = rows[0].get("last")
            return float(last_price) if last_price not in [None, ""] else None
        except Exception as e:
            logger.warning(f"[{symbol}] 实时价格获取失败: {e}")
            return None
    def _request_with_fallback(self, endpoint, params, timeout, symbol, context):
        for _ in range(len(self.base_urls)):
            base_url = self.base_urls[self.current_url_index]
            try:
                response = self.session.get(f"{base_url}{endpoint}", params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException:
                logger.warning(f"[{symbol}] {context} 节点 {base_url} 访问失败，切换中")
                self.current_url_index = (self.current_url_index + 1) % len(self.base_urls)
                time.sleep(1)
        raise RuntimeError("所有 OKX 行情节点均不可用")

    def _funding_sentiment_label(self, funding_rate):
        rate_pct = float(funding_rate) * 100.0
        if rate_pct >= 0.03:
            return "多头极度拥挤", "提示追多风险"
        if rate_pct >= 0.01:
            return "多头偏热", "追多需确认结构"
        if rate_pct <= -0.03:
            return "空头极度拥挤", "提示追空风险"
        if rate_pct <= -0.01:
            return "空头偏热", "追空需确认结构"
        return "\u4e2d\u6027", "\u65e0\u660e\u663e\u60c5\u7eea\u503e\u5411"

    def get_funding_rate(self, symbol):
        inst_id = self._symbol_to_inst_id(symbol)
        now_ts = time.time()
        cached = self.funding_rate_cache.get(inst_id)
        if cached and (now_ts - float(cached.get('time', 0.0) or 0.0)) < self.funding_cache_ttl:
            return cached.get('data', {})

        try:
            params = {"instId": inst_id}
            api_timeout = getattr(config, 'API_TIMEOUT', 15)
            response = self._request_with_fallback("/api/v5/public/funding-rate", params, api_timeout, symbol, "\u8d44\u91d1\u8d39\u7387")
            rows = response.json().get("data", [])
            row = rows[0] if rows else {}
            funding_rate = float(row.get("fundingRate", 0.0) or 0.0)
            next_funding_time = row.get("nextFundingTime")
            sentiment, note = self._funding_sentiment_label(funding_rate)
            data = {
                'symbol': symbol,
                'inst_id': inst_id,
                'funding_rate': funding_rate,
                'funding_rate_pct': funding_rate * 100.0,
                'sentiment': sentiment,
                'note': note,
                'next_funding_time': next_funding_time,
            }
        except Exception as e:
            logger.warning(f"[{symbol}] \u8d44\u91d1\u8d39\u7387\u83b7\u53d6\u5931\u8d25: {e}")
            data = {
                'symbol': symbol,
                'inst_id': inst_id,
                'funding_rate': 0.0,
                'funding_rate_pct': 0.0,
                'sentiment': '\u672a\u77e5',
                'note': '\u8d44\u91d1\u8d39\u7387\u6682\u4e0d\u53ef\u7528',
                'next_funding_time': None,
            }

        self.funding_rate_cache[inst_id] = {'time': now_ts, 'data': data}
        return data

    def build_funding_sentiment_summary(self, symbols):
        rows = ["\u3010\u5e02\u573a\u60c5\u7eea\u6458\u8981 - \u8d44\u91d1\u8d39\u7387\u3011"]
        seen = []
        for symbol in symbols:
            clean_symbol = str(symbol or '').upper()
            if not clean_symbol or clean_symbol in seen:
                continue
            seen.append(clean_symbol)
            funding = self.get_funding_rate(clean_symbol)
            base_symbol = clean_symbol.replace('USDT', '').replace('-USDT-SWAP', '')
            rows.append(
                f"- {base_symbol}\u8d44\u91d1\u8d39\u7387\uff1a{funding.get('funding_rate_pct', 0.0):+.3f}%\uff08{funding.get('sentiment', '\u672a\u77e5')}\uff0c{funding.get('note', '\u8d44\u91d1\u8d39\u7387\u6682\u4e0d\u53ef\u7528')}\uff09"
            )

        rows.extend([
            "",
            "\u8bf7\u7ed3\u5408\u4ee5\u4e0a\u60c5\u7eea\u80cc\u666f\u5ba1\u6838\u5f53\u524d\u4ea4\u6613\u4fe1\u53f7\uff1a",
            "- \u5982\u679c\u4fe1\u53f7\u65b9\u5411\u4e0e\u6781\u7aef\u60c5\u7eea\u65b9\u5411\u4e00\u81f4\uff0c\u5e94\u66f4\u8c28\u614e\uff0c\u53ef\u9002\u5f53\u964d\u4f4e confidence \u6216\u8981\u6c42\u66f4\u9ad8\u76c8\u4e8f\u6bd4\u3002",
            "- \u5982\u679c\u4fe1\u53f7\u65b9\u5411\u4e0e\u6781\u7aef\u60c5\u7eea\u65b9\u5411\u76f8\u53cd\uff0c\u4e14\u6280\u672f\u9762\u7ed3\u6784\u826f\u597d\uff0c\u53ef\u89c6\u4e3a\u6f5c\u5728\u53cd\u8f6c\u673a\u4f1a\uff0c\u6b63\u5e38\u8bc4\u4f30\u5373\u53ef\u3002",
            "- \u60c5\u7eea\u6570\u636e\u4ec5\u4f5c\u4e3a\u8f85\u52a9\u53c2\u8003\uff0c\u4e0d\u662f\u786c\u6027\u5426\u51b3\u6761\u4ef6\uff0c\u6700\u7ec8\u4ecd\u4ee5\u6280\u672f\u9762\u7ed3\u6784\u4e3a\u6838\u5fc3\u4f9d\u636e\u3002",
        ])
        return "\n".join(rows)

    def format_money(self, value, is_usd=True):
        if isinstance(value, str):
            return value
        abs_val = abs(value)
        sign = "+" if value > 0 else "-"
        prefix = "$" if is_usd else ""
        if abs_val >= 1_000_000:
            return f"{sign}{prefix}{abs_val / 1_000_000:.2f}M"
        if abs_val >= 1_000:
            return f"{sign}{prefix}{abs_val / 1_000:.1f}K"
        return f"{sign}{prefix}{abs_val:.0f}" if is_usd else f"{sign}{abs_val:.1f}"

    def format_cci_value(self, cci_value):
        if isinstance(cci_value, (int, float)):
            return f"**{cci_value:.2f}**" if abs(cci_value) > 170 else f"{cci_value:.2f}"
        return str(cci_value)

    def build_flow_proxy(self, inds, label):
        df = inds.get('df')
        if df is None or len(df) < 4:
            return {'ratio': 'N/A', 'net_usdt': 'N/A', 'net_base': 'N/A', 'bias': '样本不足'}

        completed = df.iloc[:-1].copy() if len(df) > 4 else df.copy()
        if completed.empty:
            return {'ratio': 'N/A', 'net_usdt': 'N/A', 'net_base': 'N/A', 'bias': '样本不足'}

        window_size = 12 if label == '1h' else 8
        window = completed.tail(min(window_size, len(completed))).copy()
        if window.empty:
            return {'ratio': 'N/A', 'net_usdt': 'N/A', 'net_base': 'N/A', 'bias': '样本不足'}

        turnover_col = 'vol_ccy_quote' if 'vol_ccy_quote' in window.columns else 'vol'
        base_col = 'vol_ccy' if 'vol_ccy' in window.columns else 'vol'
        turnover = window[turnover_col].fillna(0.0).astype(float)
        base_turnover = window[base_col].fillna(0.0).astype(float)
        delta = (window['close'] - window['open']).astype(float)
        up_mask = delta > 0
        down_mask = delta < 0
        flat_mask = ~(up_mask | down_mask)

        buy_quote = float(turnover[up_mask].sum() + turnover[flat_mask].sum() * 0.5)
        sell_quote = float(turnover[down_mask].sum() + turnover[flat_mask].sum() * 0.5)
        buy_base = float(base_turnover[up_mask].sum() + base_turnover[flat_mask].sum() * 0.5)
        sell_base = float(base_turnover[down_mask].sum() + base_turnover[flat_mask].sum() * 0.5)

        total_quote = max(buy_quote + sell_quote, 1e-9)
        net_quote = buy_quote - sell_quote
        net_base = buy_base - sell_base
        ratio_value = buy_quote / max(sell_quote, 1e-9)
        ratio_value = min(ratio_value, 9.99)
        flow_score = net_quote / total_quote
        vol_ratio = float(inds.get('vol_ratio', 1.0) or 1.0)

        if flow_score >= 0.18 and vol_ratio >= 1.2:
            bias = '偏多增强'
        elif flow_score <= -0.18 and vol_ratio >= 1.2:
            bias = '偏空增强'
        elif flow_score >= 0.08:
            bias = '轻度偏多'
        elif flow_score <= -0.08:
            bias = '轻度偏空'
        else:
            bias = '中性'

        return {
            'ratio': f"{ratio_value:.2f}",
            'net_usdt': self.format_money(net_quote, is_usd=True),
            'net_base': self.format_money(net_base, is_usd=False),
            'bias': bias,
        }

    def find_support_resistance(self, df, current_price):
        n = len(df)
        window = max(5, int(n * 0.03))
        highs, lows = [], []
        for i in range(window, n - window):
            if df['high'].iloc[i] > df['high'].iloc[i - window:i].max() and df['high'].iloc[i] > df['high'].iloc[i + 1:i + 1 + window].max():
                highs.append(df['high'].iloc[i])
            if df['low'].iloc[i] < df['low'].iloc[i - window:i].min() and df['low'].iloc[i] < df['low'].iloc[i + 1:i + 1 + window].min():
                lows.append(df['low'].iloc[i])

        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        cluster_threshold = atr * 0.5

        def cluster(points):
            if not points:
                return []
            points = sorted(points)
            out = [[points[0]]]
            for point in points[1:]:
                if point - out[-1][-1] <= cluster_threshold:
                    out[-1].append(point)
                else:
                    out.append([point])
            return [float(np.mean(group)) for group in out]

        resistances = [p for p in cluster(highs) if p > current_price]
        supports = [p for p in cluster(lows) if p < current_price]
        res = min(resistances) if resistances else current_price + atr * 3
        sup = max(supports) if supports else current_price - atr * 3
        return round(sup, 4), round(res, 4)

    def detect_kline_patterns(self, df):
        patterns = []
        for i in range(1, min(11, len(df))):
            row = df.iloc[-i]
            o, h, l, c, atr = row['open'], row['high'], row['low'], row['close'], row.get('atr', 0)
            if not atr:
                continue
            body = abs(c - o)
            total = h - l
            if total <= 0:
                continue
            up_shadow = h - max(o, c)
            down_shadow = min(o, c) - l
            idx_name = "当前" if i == 1 else f"前{i - 1}"
            if body > 0.5 * atr and up_shadow <= 0.1 * body and down_shadow <= 0.1 * body:
                patterns.append(f"实体{'大阳' if c > o else '大阴'}({idx_name})")
            if body <= 0.1 * total and total > 0.3 * atr:
                patterns.append(f"十字星({idx_name})")
            elif up_shadow >= 2 * max(body, 1e-9) and up_shadow >= 0.5 * atr:
                patterns.append(f"长上影({idx_name})")
            elif down_shadow >= 2 * max(body, 1e-9) and down_shadow >= 0.5 * atr:
                patterns.append(f"长下影({idx_name})")
        seen = set()
        result = []
        for item in patterns:
            key = item.split('(')[0]
            if key not in seen:
                seen.add(key)
                result.append(item)
        return " | ".join(result[:2]) if result else "无"

    def get_kline_indicators(self, symbol, interval):
        try:
            params = {
                "instId": self._symbol_to_inst_id(symbol),
                "bar": self._okx_bar(interval),
                "limit": 200,
            }
            api_timeout = getattr(config, 'API_TIMEOUT', 15)
            response = self._request_with_fallback("/api/v5/market/candles", params, api_timeout, symbol, f"{interval} K线")
            rows = response.json().get("data", [])
            if not rows:
                return None
            rows = list(reversed(rows))
            df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote', 'confirm'])
            df['close_time'] = pd.to_numeric(df['time'], errors='coerce')
            df[['open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote']] = df[['open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_ccy_quote']].astype(float)

            server_time_ms = int(time.time() * 1000)
            latest_open_ms = int(df['close_time'].iloc[-1])
            interval_ms = self._interval_ms(interval)
            tolerance_ms = min(max(interval_ms // 5, 60 * 1000), 5 * 60 * 1000)
            latest_expire_ms = latest_open_ms + interval_ms + tolerance_ms
            if server_time_ms > latest_expire_ms:
                delay_sec = int((server_time_ms - latest_expire_ms) / 1000)
                logger.warning(f"[{symbol}] {interval} K线数据延迟 {delay_sec} 秒，跳过")
                return None
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

            df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
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
            turnover_col = 'vol_ccy_quote' if 'vol_ccy_quote' in df.columns else 'vol'
            turnover = df[turnover_col].fillna(0.0).astype(float)
            candle_range = (df['high'] - df['low']).replace(0, np.nan)
            signed_turnover = (((df['close'] - df['open']) / candle_range).replace([np.inf, -np.inf], np.nan).fillna(0.0) * turnover)
            turnover_ma = turnover.rolling(window=win_vol).mean().replace(0, np.nan)
            df['flow_strength'] = (signed_turnover.rolling(3).mean() / turnover_ma).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            latest, prev, shift2, shift5 = df.iloc[-1], df.iloc[-2], df.iloc[-3], df.iloc[-6]
            recent_closed = df.iloc[-3:-1] if len(df) >= 3 else df.iloc[-2:-1]
            closed_above_ema20_count = int((recent_closed['close'] > recent_closed['ema20']).fillna(False).sum()) if len(recent_closed) > 0 else 0
            closed_below_ema20_count = int((recent_closed['close'] < recent_closed['ema20']).fillna(False).sum()) if len(recent_closed) > 0 else 0
            prev_vol_ratio = prev['vol'] / prev['vol_ma24'] if not pd.isna(prev['vol_ma24']) and prev['vol_ma24'] > 0 else 1.0
            lookback_24 = min(24, len(df))
            low_24h = float(df['low'].tail(lookback_24).min())
            high_24h = float(df['high'].tail(lookback_24).max())
            bullish_engulfing_volume = bool(
                latest['close'] > latest['open']
                and prev['close'] < prev['open']
                and latest['open'] <= prev['close']
                and latest['close'] >= prev['open']
                and latest['vol'] > prev['vol']
            )
            bearish_engulfing_volume = bool(
                latest['close'] < latest['open']
                and prev['close'] > prev['open']
                and latest['open'] >= prev['close']
                and latest['close'] <= prev['open']
                and latest['vol'] > prev['vol']
            )

            past_window = df.tail(22).iloc[:-2]
            min_price_idx = past_window['low'].idxmin()
            max_price_idx = past_window['high'].idxmax()
            bull_div = (latest['low'] < past_window['low'].min()) and (latest['rsi'] > df.loc[min_price_idx, 'rsi']) and (latest['rsi'] < 45)
            bear_div = (latest['high'] > past_window['high'].max()) and (latest['rsi'] < df.loc[max_price_idx, 'rsi']) and (latest['rsi'] > 55)
            kline_pattern = self.detect_kline_patterns(df)
            sup, res = self.find_support_resistance(df, latest['close'])

            return {
                "price": latest['close'],
                "prev_price": prev['close'],
                "pattern": kline_pattern,
                "vol_ratio": round(prev_vol_ratio, 2),
                "raw_klines": [f"O:{r['open']} H:{r['high']} L:{r['low']} C:{r['close']} V:{r['vol']}" for _, r in df.tail(3).iloc[::-1].iterrows()],
                "recent_candles": [{"open": float(r['open']), "high": float(r['high']), "low": float(r['low']), "close": float(r['close'])} for _, r in df.tail(3).iterrows()],
                "atr": latest['atr'],
                "adx": latest['adx'],
                "prev_adx": prev['adx'],
                "defense_low": min(df['low'].tail(3).min(), latest['close'] - latest['atr'] * 1.5),
                "defense_high": max(df['high'].tail(3).max(), latest['close'] + latest['atr'] * 1.5),
                "cci": round(latest['cci'], 2) if not pd.isna(latest['cci']) else 0,
                "rsi": latest['rsi'],
                "boll_low": latest['boll_low'],
                "boll_high": latest['boll_high'],
                "ema8": latest['ema8'],
                "ema20": latest['ema20'],
                "ema50": latest['ema50'],
                "ema100": latest['ema100'],
                "ema200": latest['ema200'],
                "vwap": latest['vwap'],
                "vwap_upper": latest['vwap_upper'],
                "vwap_lower": latest['vwap_lower'],
                "prev_vwap_upper": prev['vwap_upper'],
                "prev_vwap_lower": prev['vwap_lower'],
                "closed_above_ema20_count": closed_above_ema20_count,
                "closed_below_ema20_count": closed_below_ema20_count,
                "prev_ema8": prev['ema8'],
                "prev_ema20": prev['ema20'],
                "ema20_shift5": shift5['ema20'],
                "macd": latest['macd'],
                "macd_signal": latest['macd_signal'],
                "macd_hist": latest['macd_hist'],
                "bar_open_ms": int(pd.Timestamp(latest.name).timestamp() * 1000),
                "prev_macd": prev['macd'],
                "prev_macd_signal": prev['macd_signal'],
                "prev_macd_hist": prev['macd_hist'],
                "prev_cci": prev['cci'],
                "prev2_cci": shift2['cci'],
                "flow_strength": round(float(latest['flow_strength']), 4),
                "prev_flow_strength": round(float(prev['flow_strength']), 4),
                "prev2_flow_strength": round(float(shift2['flow_strength']), 4),
                "low_24h": low_24h,
                "high_24h": high_24h,
                "bullish_engulfing_volume": bullish_engulfing_volume,
                "bearish_engulfing_volume": bearish_engulfing_volume,
                "bull_div": bull_div,
                "bear_div": bear_div,
                "support": sup,
                "resistance": res,
                "df": df,
            }
        except Exception as e:
            logger.warning(f"[{symbol}] {interval} 行情解析异常: {type(e).__name__} - {e}")
            return None

    def trend_bias(self, inds_1d, inds_4h, inds_1h):
        def trend(inds):
            if inds['ema20'] > inds['ema50'] > inds['ema100']:
                return 1
            if inds['ema20'] < inds['ema50'] < inds['ema100']:
                return -1
            return 0

        t_1d, t_4h, t_1h = trend(inds_1d), trend(inds_4h), trend(inds_1h)
        if t_1d == 1 and t_1h == 1:
            return "long"
        if t_1d == -1 and t_1h == -1:
            return "short"
        if t_1d == 0 and t_4h != 0 and t_4h == t_1h:
            return "long" if t_1h == 1 else "short"
        return "neutral"

    def calculate_score(self, inds_1h, inds_4h, trend_state):
        long_s, short_s = 0.0, 0.0
        p = inds_1h['price']

        if inds_1h['bull_div']:
            long_s += 2.0
        if inds_1h['bear_div']:
            short_s += 2.0
        if p > inds_1h['ema8']:
            long_s += 1.0
        if p < inds_1h['ema8']:
            short_s += 1.0
        if p > inds_1h['ema20'] and p > inds_1h['ema50']:
            long_s += 1.5
        if p < inds_1h['ema20'] and p < inds_1h['ema50']:
            short_s += 1.5
        if trend_state == "long":
            long_s += 1.5
        if trend_state == "short":
            short_s += 1.5

        adx = inds_1h.get('adx', 20)
        is_strong_trend = adx > 25
        macd_val, prev_macd = inds_1h['macd_hist'], inds_1h['prev_macd_hist']
        if macd_val < prev_macd:
            short_s += 1.5
            long_s -= 1.5
        elif macd_val > prev_macd:
            long_s += 1.5
            short_s -= 1.5

        cci_val = inds_1h['cci']
        if cci_val < -150:
            long_s += 1.5 if (not is_strong_trend and macd_val > prev_macd) else 0.5
        if cci_val > 150:
            short_s += 1.5 if (not is_strong_trend and macd_val < prev_macd) else 0.5

        # Avoid chasing extreme extension in either direction.
        if cci_val < -170 and short_s > 0:
            short_s -= 0.5
        if cci_val > 170 and long_s > 0:
            long_s -= 0.5

        rsi_val = float(inds_1h.get('rsi') or 50.0)
        if p <= inds_1h['boll_low'] and macd_val > prev_macd:
            long_s += 1.0
        if rsi_val < 35:
            long_s += 1.0
        if p >= inds_1h['boll_high'] and macd_val < prev_macd:
            short_s += 1.0
        if rsi_val > 65:
            short_s += 1.0

        # Penalize chasing when trend is already overstretched.
        if p >= inds_1h['boll_high'] and rsi_val >= 70 and long_s > 0:
            long_s -= 1.0
        if p <= inds_1h['boll_low'] and rsi_val <= 30 and short_s > 0:
            short_s -= 1.0

        atr_gap = float(inds_1h.get('atr') or 0.0)
        ema20_gap = abs(p - float(inds_1h.get('ema20') or p))
        if atr_gap > 0:
            if p > inds_1h['ema20'] and (ema20_gap / atr_gap) > 1.2 and long_s > 0:
                long_s -= 1.0
            if p < inds_1h['ema20'] and (ema20_gap / atr_gap) > 1.2 and short_s > 0:
                short_s -= 1.0

        # Reward healthier early continuation: back above/below EMA20 but not far away yet.
        if atr_gap > 0 and 0 < (p - inds_1h['ema20']) <= (0.6 * atr_gap) and macd_val > prev_macd:
            long_s += 0.5
        if atr_gap > 0 and 0 < (inds_1h['ema20'] - p) <= (0.6 * atr_gap) and macd_val < prev_macd:
            short_s += 0.5

        vol_ratio = inds_1h.get('vol_ratio', 1.0)
        if vol_ratio >= 1.2 and macd_val > prev_macd:
            long_s += 1.0
        elif vol_ratio < 0.9:
            long_s -= 0.5
        if vol_ratio >= 1.2 and macd_val < prev_macd:
            short_s += 1.0
        elif vol_ratio < 0.9:
            short_s -= 0.5

        atr_val = float(inds_1h.get('atr') or 0.0)
        low_24h = float(inds_1h.get('low_24h') or 0.0)
        high_24h = float(inds_1h.get('high_24h') or 0.0)
        if atr_val > 0 and low_24h and ((p - low_24h) / atr_val) > 0.5:
            long_s += 1.0
        if atr_val > 0 and high_24h and ((high_24h - p) / atr_val) > 0.5:
            short_s += 1.0

        if inds_4h.get('bullish_engulfing_volume'):
            long_s += 1.5
        if inds_4h.get('bearish_engulfing_volume'):
            short_s += 1.5

        flow_now = float(inds_1h.get('flow_strength') or 0.0)
        flow_prev = float(inds_1h.get('prev_flow_strength') or 0.0)
        flow_prev2 = float(inds_1h.get('prev2_flow_strength') or 0.0)
        if flow_prev2 <= 0 < flow_prev and flow_now > 0:
            long_s += 1.0
        if flow_prev2 >= 0 > flow_prev and flow_now < 0:
            short_s += 1.0

        prev_price = float(inds_1h.get('prev_price') or p)
        vwap_upper = inds_1h.get('vwap_upper')
        prev_vwap_upper = inds_1h.get('prev_vwap_upper')
        vwap_lower = inds_1h.get('vwap_lower')
        prev_vwap_lower = inds_1h.get('prev_vwap_lower')
        if vwap_upper is not None and prev_vwap_upper is not None and p > float(vwap_upper) and prev_price <= float(prev_vwap_upper):
            long_s += 1.0
        if vwap_lower is not None and prev_vwap_lower is not None and p < float(vwap_lower) and prev_price >= float(prev_vwap_lower):
            short_s += 1.0

        return max(round(long_s, 1), 0.0), max(round(short_s, 1), 0.0)

    def calculate_position_scaler(self, score, atr, price, min_score=5.0):
        if price == 0 or atr == 0 or score < min_score:
            return 0.0
        score_weight = min(0.3 + ((score - min_score) / 4.0) * 1.2, 1.5)
        return round(score_weight * min(max(0.02 / (atr / price), 0.2), 2.0), 2)

    def evaluate_trend(self, inds):
        p, e8, e20, e50 = inds['price'], inds['ema8'], inds['ema20'], inds['ema50']
        vol_tag = "放量" if inds.get('vol_ratio', 1.0) >= 2.0 else ""
        if inds['prev_ema8'] <= inds['prev_ema20'] and e8 > e20:
            ema_s = f"{vol_tag}金叉"
        elif inds['prev_ema8'] >= inds['prev_ema20'] and e8 < e20:
            ema_s = f"{vol_tag}死叉"
        elif p > e8 and p > e20 and p > e50:
            ema_s = "多头排列"
        elif p < e8 and p < e20 and p < e50:
            ema_s = "空头排列"
        else:
            ema_s = "震荡"

        if inds['prev_macd'] <= inds['prev_macd_signal'] and inds['macd'] > inds['macd_signal']:
            macd_s = "MACD金叉"
        elif inds['prev_macd'] >= inds['prev_macd_signal'] and inds['macd'] < inds['macd_signal']:
            macd_s = "MACD死叉"
        else:
            macd_s = "多" if inds['macd'] > inds['macd_signal'] else "空"

        h, ph = inds['macd_hist'], inds['prev_macd_hist']
        if h > 0 and ph > 0:
            hist_s = "多增" if h > ph else "多减"
        elif h < 0 and ph < 0:
            hist_s = "空增" if h < ph else "空减"
        elif h > 0 >= ph:
            hist_s = "翻绿"
        elif h < 0 <= ph:
            hist_s = "翻红"
        else:
            hist_s = "平"
        return {
            "ema_status": ema_s,
            "macd_status": macd_s,
            "hist_status": hist_s,
            "hist_val": h,
            "pattern": inds['pattern'],
            "text": f"{ema_s} | {macd_s} | {hist_s}",
        }

    def get_market_data(self, symbol):
        try:
            inds = {}
            for tf in ['15m', '1h', '4h', '1d']:
                inds[tf] = self.get_kline_indicators(symbol, tf)
                time.sleep(0.2)

            if not all(inds.values()):
                return None

            price = inds['1h']['price']
            realtime_price = self.get_realtime_price(symbol)
            sentiment_symbols = list(getattr(config, 'MAINSTREAM', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])[:3])
            if symbol not in sentiment_symbols:
                sentiment_symbols = [symbol] + sentiment_symbols[:2]
            funding_snapshot = self.get_funding_rate(symbol)
            market_sentiment_summary = self.build_funding_sentiment_summary(sentiment_symbols)
            taker = {
                '1h': self.build_flow_proxy(inds['1h'], '1h'),
                '4h': self.build_flow_proxy(inds['4h'], '4h'),
            }
            trend_state = self.trend_bias(inds['1d'], inds['4h'], inds['1h'])
            long_score, short_score = self.calculate_score(inds['1h'], inds['4h'], trend_state)
            trend_1h = self.evaluate_trend(inds['1h'])
            trend_4h = self.evaluate_trend(inds['4h'])
            trend_1d = self.evaluate_trend(inds['1d'])

            return {
                "symbol": symbol,
                "price": price,
                "price_15m": inds['15m']['price'],
                "ema20_15m": inds['15m']['ema20'],
                "vwap_15m": inds['15m']['vwap'],
                "closed_above_ema20_count_15m": inds['15m']['closed_above_ema20_count'],
                "closed_below_ema20_count_15m": inds['15m']['closed_below_ema20_count'],
                "macd_15m": inds['15m']['macd'],
                "macd_signal_15m": inds['15m']['macd_signal'],
                "prev_macd_15m": inds['15m']['prev_macd'],
                "prev_macd_signal_15m": inds['15m']['prev_macd_signal'],
                "cci_15m": inds['15m']['cci'],
                "prev_cci_15m": inds['15m']['prev_cci'],
                "closed_cci_15m": inds['15m']['prev_cci'],
                "prev_closed_cci_15m": inds['15m']['prev2_cci'],
                "bar_open_15m_ms": inds['15m']['bar_open_ms'],
                "realtime_price": realtime_price,
                "cci_1h": inds['1h']['cci'],
                "prev_cci_1h": inds['1h']['prev_cci'],
                "cci_4h": inds['4h']['cci'],
                "prev_cci_4h": inds['4h']['prev_cci'],
                "cci_1d": inds['1d']['cci'],
                "vol_ratio_1h": inds['1h']['vol_ratio'],
                "taker_1h": taker['1h'],
                "funding_snapshot": funding_snapshot,
                "market_sentiment_summary": market_sentiment_summary,
                "defense_low": inds['1h']['defense_low'],
                "defense_high": inds['1h']['defense_high'],
                "atr": inds['1h']['atr'],
                "adx_1h": inds['1h']['adx'],
                "prev_adx_1h": inds['1h']['prev_adx'],
                "rsi_1h": inds['1h']['rsi'],
                "boll_high_1h": inds['1h']['boll_high'],
                "boll_low_1h": inds['1h']['boll_low'],
                "ema20_1h": inds['1h']['ema20'],
                "trend_state": trend_state,
                "ema200_4h": inds['4h']['ema200'],
                "ema200_1d": inds['1d']['ema200'],
                "ema200_trend_1d": "上方" if price >= inds['1d']['ema200'] else "下方",
                "ema200_gap_pct_1d": round(((price - inds['1d']['ema200']) / inds['1d']['ema200']) * 100, 2) if inds['1d']['ema200'] else 0.0,
                "long_score": long_score,
                "short_score": short_score,
                "ema_status_1h": trend_1h['ema_status'],
                "ema_status_4h": trend_4h['ema_status'],
                "ema_status_1d": trend_1d['ema_status'],
                "ema_trend": f"[1H] {trend_1h['ema_status']} | [4H] {trend_4h['ema_status']} | [1D] {trend_1d['ema_status']} | [1D EMA200] {"上方" if price >= inds['1d']['ema200'] else "下方"}",
                "macd_info": f"[1H] {trend_1h['macd_status']}({trend_1h['hist_status']}) | [4H] {trend_4h['macd_status']}({trend_4h['hist_status']})",
                "naked_k": f"[1H] {inds['1h']['pattern']} | [4H] {inds['4h']['pattern']}",
                "cci_info": f"[1H] {self.format_cci_value(inds['1h']['cci'])} | [4H] {self.format_cci_value(inds['4h']['cci'])} | [1D] {self.format_cci_value(inds['1d']['cci'])}",
                "money_flow": f"[1H] 买卖比 {taker['1h']['ratio']} | 净额 {taker['1h']['net_usdt']} | 量能:{inds['1h']['vol_ratio']}x\n[4H] 买卖比 {taker['4h']['ratio']} | 净额 {taker['4h']['net_usdt']} | 量能:{inds['4h']['vol_ratio']}x",
                "raw_klines_1h": inds['1h']['raw_klines'],
                "recent_candles_1h": inds['1h']['recent_candles'],
                "recent_candles_4h": inds['4h']['recent_candles'],
                "raw_klines_4h": inds['4h']['raw_klines'],
                "raw_klines_15m": inds['15m']['raw_klines'],
                "bar_open_1h_ms": inds['1h']['bar_open_ms'],
                "support_1h": inds['1h']['support'],
                "resistance_1h": inds['1h']['resistance'],
                "support_4h": inds['4h']['support'],
                "resistance_4h": inds['4h']['resistance'],
                "support_1d": inds['1d']['support'],
                "resistance_1d": inds['1d']['resistance'],
                "ema200_support_4h": inds['4h']['ema200'] if price >= inds['4h']['ema200'] else None,
                "ema200_resistance_4h": inds['4h']['ema200'] if price < inds['4h']['ema200'] else None,
                "ema200_support_1d": inds['1d']['ema200'] if price >= inds['1d']['ema200'] else None,
                "ema200_resistance_1d": inds['1d']['ema200'] if price < inds['1d']['ema200'] else None,
            }
        except Exception as e:
            logger.warning(f"[{symbol}] 整合数据失败: {e}")
            return None

    def analyze_signals(self, data):
        if getattr(config, 'ENABLE_QUANT_SCORING', False):
            return self._analyze_signals_scoring(data)
        return self._analyze_signals_scoring(data)

    def _analyze_signals_scoring(self, data):
        ls, ss = data['long_score'], data['short_score']
        trend, dl, dh = data['trend_state'], data['defense_low'], data['defense_high']
        atr, price = data['atr'], data['price']

        sup_1d, res_1d = data.get('support_1d'), data.get('resistance_1d')
        close_ratio = getattr(config, 'CLOSE_TO_SUPPORT_RESISTANCE_RATIO', 0.003)
        if sup_1d and abs(price - sup_1d) / price <= close_ratio:
            ls += 1.5
            data['long_score'] = ls
        if res_1d and abs(price - res_1d) / price <= close_ratio:
            ss += 1.5
            data['short_score'] = ss

        base_min_score = getattr(config, 'MIN_OPEN_SCORE', 5.0)
        dynamic_threshold = (base_min_score + 1.0) if (atr / price) > 0.04 else base_min_score
        data['suggested_scaler'] = 0.0

        if ls >= dynamic_threshold and ls > ss:
            scaler = self.calculate_position_scaler(ls, atr, price, dynamic_threshold)
            data['suggested_scaler'] = scaler
            msg = f"量化评分看多 (得分:{ls}/9) | {'顺势突破' if trend == 'long' else '逆势抄底'}"
            return f"{msg} | ATR防守:{dl:.4f} | 建议缩放: {scaler}x"

        if ss >= dynamic_threshold and ss > ls:
            scaler = self.calculate_position_scaler(ss, atr, price, dynamic_threshold)
            data['suggested_scaler'] = scaler
            msg = f"量化评分看空 (得分:{ss}/9) | {'顺势下杀' if trend == 'short' else '逆势逃顶'}"
            return f"{msg} | ATR防守:{dh:.4f} | 建议缩放: {scaler}x"

        return f"震荡观望 (多 {ls} 空 {ss} 动态阈值 {dynamic_threshold})"

