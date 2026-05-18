import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.okx.com/api/v5/market/history-candles"

COL_SYMBOL = "交易对"
COL_ALGO_SIDE = "策略触发方向"
COL_TRIGGER_TIME = "触发时间"
COL_AI_ACTION = "AI最终判定"
COL_FUTURE_4H = "后续4H涨跌(%)"
COL_REVIEW = "拦截评估结果"

LEGACY_COLUMN_MAP = {
    "交易对": COL_SYMBOL,
    "策略触发方向": COL_ALGO_SIDE,
    "触发时间": COL_TRIGGER_TIME,
    "AI最终判定": COL_AI_ACTION,
    "后续4H涨跌(%)": COL_FUTURE_4H,
    "拦截评估结果": COL_REVIEW,
}


def get_future_price(symbol, timestamp_ms):
    params = {
        "instId": f"{symbol.replace('USDT', '')}-USDT-SWAP",
        "bar": "1H",
        "before": timestamp_ms,
        "limit": 5,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data or len(data) < 2:
            return None
        data = list(reversed(data))
        return {"entry": float(data[0][1]), "last": float(data[-1][4])}
    except Exception:
        return None


def normalize_columns(df):
    rename_map = {old: new for old, new in LEGACY_COLUMN_MAP.items() if old in df.columns and new not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def ensure_columns(df):
    required = [COL_SYMBOL, COL_ALGO_SIDE, COL_TRIGGER_TIME, COL_AI_ACTION]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"日志缺少必要字段: {', '.join(missing)}")

    if COL_FUTURE_4H not in df.columns:
        df[COL_FUTURE_4H] = np.nan
    if COL_REVIEW not in df.columns:
        df[COL_REVIEW] = ""
    return df


def verify_and_update_logs():
    log_file = "ai_decision_logs.csv"
    if not os.path.exists(log_file):
        print("未找到 AI 日志文件。")
        return

    df = pd.read_csv(log_file, encoding="utf-8-sig")
    df = normalize_columns(df)
    df = ensure_columns(df)

    mask = (df[COL_AI_ACTION] == "PASS") & (df[COL_FUTURE_4H].isna())
    pending_indices = df[mask].index
    if len(pending_indices) == 0:
        print("所有 PASS 记录都已经复盘完成。")
        return

    print(f"发现 {len(pending_indices)} 条待复盘记录，开始抓取 OKX 价格...\n")
    results = []
    for idx in pending_indices:
        row = df.loc[idx]
        symbol = row[COL_SYMBOL]
        algo_side = row[COL_ALGO_SIDE]
        trigger_time = row[COL_TRIGGER_TIME]

        try:
            dt = datetime.strptime(trigger_time, "%Y-%m-%d %H:%M:%S")
            ts_ms = int(dt.timestamp() * 1000)
            price_move = get_future_price(symbol, ts_ms)
            if not price_move:
                continue

            change_pct = (price_move["last"] - price_move["entry"]) / price_move["entry"] * 100
            is_hero = (algo_side == "LONG" and change_pct < 0) or (algo_side == "SHORT" and change_pct > 0)
            status = "成功拦截反转" if is_hero else "拦截导致踏空"
            df.at[idx, COL_FUTURE_4H] = round(change_pct, 2)
            df.at[idx, COL_REVIEW] = status
            print(f"[{trigger_time}] {symbol} | 倾向:{algo_side} | 涨跌:{change_pct:+.2f}% | {status}")
            results.append(is_hero)
        except Exception as e:
            print(f"处理 {symbol} 出现异常: {e}")
        time.sleep(0.2)

    df.to_csv(log_file, index=False, encoding="utf-8-sig")
    print(f"\n已将 {len(results)} 条新复盘结果写回 {log_file}。")

    completed_mask = (df[COL_AI_ACTION] == "PASS") & (df[COL_REVIEW].isin(["成功拦截反转", "拦截导致踏空"]))
    total_completed = int(completed_mask.sum())
    total_success = int((df[COL_REVIEW] == "成功拦截反转").sum())
    if total_completed > 0:
        accuracy = total_success / total_completed * 100
        print(f"AI 拦截真实反转率: {accuracy:.2f}% ({total_success}/{total_completed})")


if __name__ == "__main__":
    verify_and_update_logs()
