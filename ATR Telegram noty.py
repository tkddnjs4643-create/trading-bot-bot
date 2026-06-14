import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 환율 설정
KRW_USD_RATE = 1350

def fmt_krw(usd):
    return f"₩{int(usd * KRW_USD_RATE):,}"

# ──────────────────────────────────────────────────────────────
# 1. 설정 (종목명과 티커 정보)
# ──────────────────────────────────────────────────────────────
CFG = {
    "QLD":   {"name": "QLD (나스닥 2배)", "ticker": "QQQ", "ma": 200, "mult": 1.8, "whip": 30},
    "SPTL":  {"name": "SPTL (장기채)",    "ticker": "SPTL", "ma": 200, "mult": 2.5, "whip": 30},
    "KOSPI": {"name": "코스피",          "ticker": "^KS11", "ma": 120, "mult": 2.0, "whip": 30},
}

# ──────────────────────────────────────────────────────────────
# 2. 분석 엔진 (필터링 로직 + 원화 계산 포함)
# ──────────────────────────────────────────────────────────────
def calculate_atr(df, window=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def analyze(key):
    cfg = CFG[key]
    raw = yf.download(cfg["ticker"], period="400d", auto_adjust=True, progress=False)
    close, high, low = raw["Close"].squeeze(), raw["High"].squeeze(), raw["Low"].squeeze()
    ma, atr = close.rolling(cfg["ma"]).mean(), calculate_atr(raw)
    
    last, prev, ma_last, atr_last = float(close.iloc[-1]), float(close.iloc[-2]), float(ma.iloc[-1]), float(atr.iloc[-1])
    move, above = last - prev, last > ma_last

    # ATR을 원화 기준으로 환산
    atr_krw = atr_last * KRW_USD_RATE if cfg["ticker"] != "^KS11" else atr_last
    last_krw = last * KRW_USD_RATE if cfg["ticker"] != "^KS11" else last

    # [필터 1] 서킷브레이커 로직
    if not above and move <= -cfg["mult"] * atr_last:
        return {"signal": "SELL", "msg": f"이탈! 현재가 {fmt_krw(last_krw)}"}
    if not above and move >= cfg["mult"] * atr_last:
        return {"signal": "BUY", "msg": f"복귀! 현재가 {fmt_krw(last_krw)}"}

    # [필터 2] WHIPSAW 로직
    rc_close, rc_ma = close.iloc[-cfg["whip"]:], ma.iloc[-cfg["whip"]:]
    all_below = all(c < m for c, m in zip(rc_close, rc_ma) if not np.isnan(m))
    all_above = all(c > m for c, m in zip(rc_close, rc_ma) if not np.isnan(m))
    
    if all_below and last < (ma_last - 0.2 * atr_last):
        return {"signal": "SELL", "msg": f"연속이탈! 현재가 {fmt_krw(last_krw)}"}
    if all_above and last > (ma_last + 0.2 * atr_last):
        return {"signal": "BUY", "msg": f"연속상회! 현재가 {fmt_krw(last_krw)}"}

    return {"signal": "HOLD" if above else "WATCH", "msg": f"현재가 {fmt_krw(last_krw)}"}

# ──────────────────────────────────────────────────────────────
# 3. 텔레그램 전송 및 실행
# ──────────────────────────────────────────────────────────────
def send_telegram(msg):
    token = os.environ.get("TG_TOKEN")
    chat_id = os.environ.get("TG_CHAT_ID")
    if token and chat_id:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": msg})

results = {key: analyze(key) for key in CFG}
signals = [f"🚨 [{r['signal']}] {CFG[key]['name']}: {r['msg']}" for key, r in results.items() if r['signal'] in ["SELL", "BUY"]]

if signals:
    send_telegram("\n".join(signals))
else:
    send_telegram(f"✅ [데일리 체크] {datetime.now().strftime('%m/%d')} 정상 보유 중.")