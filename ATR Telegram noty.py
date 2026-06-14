# ============================================================
#  ATR 신호 체커 + 텔레그램 알림
#  매일 오전 9시 깃허브 Actions 자동 실행
# ============================================================
# !pip install yfinance pandas numpy requests

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# ★ 텔레그램 설정
# ──────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TG_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID")

# 로컬/Colab 테스트 시 주석 해제
# TELEGRAM_TOKEN   = "직접_토큰_입력"
# TELEGRAM_CHAT_ID = "직접_채팅ID_입력"

KRW_RATE = 1350

# ──────────────────────────────────────────────────────────────
# 전략 설정
# ──────────────────────────────────────────────────────────────
CFG = {
    "QLD": {
        "name": "QLD (나스닥 2배)", "ticker": "QQQ",
        "ticker_label": "QQQ",
        "hold_label": "QLD", "defense_label": "QQQ(나스닥100)",
        "ma": 200, "mult": 1.8, "whip": 30, "cur": "USD",
        "defense": "QQQ", "weight": 0.35
    },
    "SPTL": {
        "name": "SPTL (장기채)", "ticker": "SPTL",
        "ticker_label": "SPTL",
        "hold_label": "SPTL", "defense_label": "USFR(단기채)",
        "ma": 200, "mult": 2.5, "whip": 30, "cur": "USD",
        "defense": "USFR", "weight": 0.20
    },
    "KOSPI": {
        "name": "KODEX 코리아밸류업", "ticker": "^KS11",
        "ticker_label": "코스피 지수 기준",
        "hold_label": "KODEX 밸류업", "defense_label": "QQQ(나스닥100)",
        "ma": 120, "mult": 2.0, "whip": 30, "cur": "KRW",
        "defense": "QQQ", "weight": 0.05
    },
}

# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────
def fmt_krw(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "-"
    won = v * KRW_RATE
    if won >= 1e8: return f"₩{won/1e8:.2f}억"
    if won >= 1e4: return f"₩{won/1e4:.0f}만"
    return f"₩{won:,.0f}"

def fmt_price(v, cur):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "-"
    if cur == "KRW": return f"{int(v):,}pt"
    return f"${v:,.2f} ({fmt_krw(v)})"

# ──────────────────────────────────────────────────────────────
# ATR 계산
# ──────────────────────────────────────────────────────────────
def calculate_atr(df, window=14):
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    c = df["Close"].squeeze()
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# ──────────────────────────────────────────────────────────────
# 신호 분석
# ──────────────────────────────────────────────────────────────
def analyze(key):
    cfg           = CFG[key]
    defense_label = cfg.get("defense_label", cfg["defense"])
    hold_label    = cfg.get("hold_label", cfg["ticker"])
    ma_label      = cfg.get("ticker_label", cfg["ticker"])

    try:
        raw = yf.download(cfg["ticker"], period="400d", auto_adjust=True, progress=False)
        if len(raw) < cfg["ma"] + 10:
            return {"signal": "error", "msg": "데이터 부족"}

        close    = raw["Close"].squeeze()
        ma       = close.rolling(cfg["ma"]).mean()
        atr      = calculate_atr(raw)
        last     = float(close.iloc[-1])
        prev     = float(close.iloc[-2])
        ma_last  = float(ma.iloc[-1])
        atr_last = float(atr.iloc[-1])
        move     = last - prev
        above    = last > ma_last
        ratio    = (last / ma_last - 1) * 100
        position   = hold_label if above else defense_label
        is_defense = not above

        # 1순위: 서킷브레이커
        if not above and move <= -cfg["mult"] * atr_last:
            return {"signal": "SELL", "position": position, "is_defense": is_defense, "priority": 1,
                    "msg": f"[1순위] ATR {cfg['mult']}배 급락 감지 → {defense_label} 방어 진입",
                    "price": last, "ma": ma_last, "atr": atr_last, "ratio": ratio, "above": above}
        if not above and move >= cfg["mult"] * atr_last:
            return {"signal": "BUY", "position": position, "is_defense": is_defense, "priority": 1,
                    "msg": f"[1순위] ATR {cfg['mult']}배 급등 감지 → {hold_label} 복귀",
                    "price": last, "ma": ma_last, "atr": atr_last, "ratio": ratio, "above": above}

        # 2순위: WHIPSAW
        rc_close  = close.iloc[-cfg["whip"]:]
        rc_ma     = ma.iloc[-cfg["whip"]:]
        all_below = all(c < m for c, m in zip(rc_close, rc_ma) if not np.isnan(m))
        all_above = all(c > m for c, m in zip(rc_close, rc_ma) if not np.isnan(m))
        buf_low   = ma_last - 0.2 * atr_last
        buf_high  = ma_last + 0.2 * atr_last

        if all_below and last < buf_low:
            return {"signal": "SELL", "position": position, "is_defense": is_defense, "priority": 2,
                    "msg": f"[2순위] {ma_label} {cfg['ma']}일선 {cfg['whip']}일 연속 하회 → {defense_label} 방어 진입",
                    "price": last, "ma": ma_last, "atr": atr_last, "ratio": ratio, "above": above}
        if all_above and last > buf_high:
            return {"signal": "BUY", "position": position, "is_defense": is_defense, "priority": 2,
                    "msg": f"[2순위] {ma_label} {cfg['ma']}일선 {cfg['whip']}일 연속 회복 → {hold_label} 복귀",
                    "price": last, "ma": ma_last, "atr": atr_last, "ratio": ratio, "above": above}

        # 유지
        if above:
            msg = f"{ma_label} ATR 필터 이상 없음 → {hold_label} 보유 유지"
        else:
            msg = f"{ma_label} {cfg['ma']}일선 하회 중 (ATR 버퍼 내) → {defense_label} 방어 유지"

        return {"signal": "HOLD" if above else "WATCH",
                "position": position, "is_defense": is_defense, "priority": 0,
                "msg": msg, "price": last, "ma": ma_last,
                "atr": atr_last, "ratio": ratio, "above": above}

    except Exception as e:
        return {"signal": "error", "position": "-", "is_defense": False, "priority": 0, "msg": str(e)}

# ──────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────
print("📥 데이터 다운로드 중...")
results = {}
for key in CFG:
    print(f"   {CFG[key]['name']} 분석 중...")
    results[key] = analyze(key)
print("✅ 완료!\n")

for key, r in results.items():
    sig  = r.get("signal", "error")
    icon = "🔴" if sig=="SELL" else "🟢" if sig=="BUY" else "🟡" if sig=="WATCH" else "🔵"
    pos  = r.get("position", "-")
    defe = " [방어중]" if r.get("is_defense") else ""
    print(f"{icon} {CFG[key]['name']}{defe}: {r.get('msg','')}")
    if r.get("price"):
        print(f"   포지션: {pos} | 현재가: {fmt_price(r['price'], CFG[key]['cur'])} | 이격도: {r['ratio']:+.2f}%")

# ──────────────────────────────────────────────────────────────
# 텔레그램
# ──────────────────────────────────────────────────────────────
def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            print("📨 텔레그램 전송 완료!")
        else:
            print(f"❌ 텔레그램 오류: {r.text}")
    except Exception as e:
        print(f"❌ 텔레그램 연결 실패: {e}")

def build_telegram_msg():
    now   = datetime.now().strftime("%Y/%m/%d %H:%M")
    today = datetime.now().strftime("%Y/%m/%d")
    sigs  = [r.get("signal") for r in results.values()]

    has_sell  = "SELL"  in sigs
    has_buy   = "BUY"   in sigs
    has_watch = "WATCH" in sigs

    if has_sell:
        header = "🔴 매도 신호 유지 중 — 포지션 확인"
    elif has_buy:
        header = "🟢 복귀 신호 유지 중 — 포지션 확인"
    elif has_watch:
        header = "🟡 방어 유지 중 — 매매 없음"
    else:
        header = "📋 어제와 동일 — 매매 없음"

    lines = [
        f"📅 <b>{today}</b>  |  ATR 신호 체커",
        f"🕐 업데이트: {now}",
        "",
        header,
        "─" * 28,
    ]

    for key, r in results.items():
        cfg   = CFG[key]
        sig   = r.get("signal", "error")
        icon  = "🔴" if sig=="SELL" else "🟢" if sig=="BUY" else "🟡" if sig=="WATCH" else "🔵"
        pos   = r.get("position", "-")
        defe  = "🛡방어중" if r.get("is_defense") else "📈보유중"
        pr    = r.get("price")
        ma    = r.get("ma")
        ratio = r.get("ratio")

        lines.append(f"\n{icon} <b>{cfg['name']}</b>")
        lines.append(f"  [{defe}: {pos}]")
        if pr:
            if cfg["cur"] == "USD":
                lines.append(f"  현재가: ${pr:,.2f} ({fmt_krw(pr)})")
                lines.append(f"  {cfg['ma']}일선: ${ma:,.2f} | 이격도: {ratio:+.2f}%")
            else:
                lines.append(f"  현재가: {int(pr):,}pt")
                lines.append(f"  {cfg['ma']}일선: {int(ma):,}pt | 이격도: {ratio:+.2f}%")
        lines.append(f"  → {r.get('msg','')}")

    lines += ["", "─" * 28]

    if has_sell or has_buy:
        lines.append("⚡ 오늘 장 시작 시 매매 대응")
    else:
        lines.append("💤 오늘 매매 없음 — 오전 9시 다시 확인")

    return "\n".join(lines)

msg = build_telegram_msg()
print("\n" + "="*50)
print(msg)
print("="*50)

send_telegram(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, msg)

# ──────────────────────────────────────────────────────────────
# HTML 생성
# ──────────────────────────────────────────────────────────────
def badge(sig):
    styles = {
        "SELL":  ("background:#4d1019;color:#f85149", "🔴 매도 신호"),
        "BUY":   ("background:#0d4429;color:#3fb950", "🟢 복귀 신호"),
        "WATCH": ("background:#3d2200;color:#e3b341", "🟡 주시"),
    }
    s, t = styles.get(sig, ("background:#21262d;color:#8b949e", "🔵 보유 유지"))
    return f'<span style="{s};padding:3px 10px;border-radius:20px;font-size:11px">{t}</span>'

def pos_badge(is_defense, pos):
    if is_defense:
        return f'<span style="background:#3d2200;color:#e3b341;padding:2px 8px;border-radius:10px;font-size:11px">🛡 방어중: {pos}</span>'
    return f'<span style="background:#0d4429;color:#3fb950;padding:2px 8px;border-radius:10px;font-size:11px">📈 보유중: {pos}</span>'

def card_html(key, r):
    cfg   = CFG[key]
    ratio = r.get("ratio")
    pr    = r.get("price")
    ma    = r.get("ma")
    atr   = r.get("atr")
    above = r.get("above")
    rc    = "#3fb950" if (ratio or 0) >= 0 else "#f85149"
    rs    = f"{ratio:+.2f}%" if ratio is not None else "-"
    ab    = "▲상회" if above else "▼하회" if above is not None else "-"

    if cfg["cur"] == "USD":
        pr_str  = f"${pr:,.2f} <span style='color:#8b949e;font-size:11px'>({fmt_krw(pr)})</span>" if pr else "-"
        ma_str  = f"${ma:,.2f} <span style='color:#8b949e;font-size:11px'>({fmt_krw(ma)})</span>" if ma else "-"
        atr_str = f"${atr:,.3f}" if atr else "-"
    else:
        pr_str  = f"{int(pr):,}pt" if pr else "-"
        ma_str  = f"{int(ma):,}pt" if ma else "-"
        atr_str = f"{int(atr):,}pt" if atr else "-"

    return f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
        <div>
          <div style="font-size:15px;font-weight:500">{cfg['name']}</div>
          <div style="font-size:11px;color:#8b949e;margin-top:2px">{cfg['ticker']}</div>
        </div>
        {badge(r.get('signal','error'))}
      </div>
      <div style="margin-bottom:12px">{pos_badge(r.get('is_defense',False), r.get('position','-'))}</div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #21262d;font-size:13px">
        <span style="color:#8b949e">현재가</span><span style="font-weight:500">{pr_str}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #21262d;font-size:13px">
        <span style="color:#8b949e">{cfg['ma']}일선</span>
        <span style="font-weight:500">{ma_str} ({ab})</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #21262d;font-size:13px">
        <span style="color:#8b949e">ATR (14일)</span><span style="font-weight:500">{atr_str}</span>
      </div>
      <div style="display:flex;justify-content:space-between;padding:7px 0;font-size:13px">
        <span style="color:#8b949e">이격도</span>
        <span style="font-weight:500;color:{rc}">{rs}</span>
      </div>
      <div style="margin-top:10px;background:#0d1117;border-radius:6px;padding:9px 12px;font-size:12px;color:#8b949e">
        {r.get('msg','')}
      </div>
    </div>"""

def summary_banner():
    sigs = [r.get("signal") for r in results.values()]
    if "SELL" in sigs:
        return '<div style="background:#4d1019;border:1px solid #f85149;border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:10px;margin-bottom:20px"><span style="font-size:22px">🚨</span><div><div style="font-size:15px;font-weight:500">매도 신호 발생!</div><div style="font-size:12px;color:#8b949e;margin-top:2px">오늘 장 시작 시 매도 대응</div></div></div>'
    if "BUY" in sigs:
        return '<div style="background:#0d4429;border:1px solid #238636;border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:10px;margin-bottom:20px"><span style="font-size:22px">✅</span><div><div style="font-size:15px;font-weight:500">복귀 신호 발생!</div><div style="font-size:12px;color:#8b949e;margin-top:2px">오늘 장 시작 시 매수 대응</div></div></div>'
    if "WATCH" in sigs:
        return '<div style="background:#3d2200;border:1px solid #e3b341;border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:10px;margin-bottom:20px"><span style="font-size:22px">⚠️</span><div><div style="font-size:15px;font-weight:500">주시 구간</div><div style="font-size:12px;color:#8b949e;margin-top:2px">이평선 하회 중 — 포지션 유지</div></div></div>'
    return '<div style="background:#0d4429;border:1px solid #238636;border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:10px;margin-bottom:20px"><span style="font-size:22px">✅</span><div><div style="font-size:15px;font-weight:500">모든 자산 정상 보유</div><div style="font-size:12px;color:#8b949e;margin-top:2px">매매 신호 없음</div></div></div>'

def sig_rows():
    rows = ""
    for key, r in results.items():
        cfg = CFG[key]
        sig = r.get("signal", "error")
        dc  = "#f85149" if sig=="SELL" else "#3fb950" if sig=="BUY" else "#e3b341" if sig=="WATCH" else "#3fb950"
        act = "→ 매도 대응" if sig=="SELL" else "→ 매수 대응" if sig=="BUY" else "→ 유지(주시)" if sig=="WATCH" else "→ 보유 유지"
        pos = r.get("position", "-")
        def_txt = f" [🛡{pos}]" if r.get("is_defense") else f" [📈{pos}]"
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid #21262d">
          <span style="width:8px;height:8px;border-radius:50%;background:{dc};flex-shrink:0;display:inline-block"></span>
          <span style="font-size:13px;font-weight:500;min-width:130px">{cfg['name']}{def_txt}</span>
          <span style="font-size:12px;color:#8b949e;flex:1">{r.get('msg','')}</span>
          <span style="font-size:12px;font-weight:500;color:{dc}">{act}</span>
        </div>"""
    return rows

now_str = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
cards   = "".join(card_html(k, v) for k, v in results.items())

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>ATR 신호 체커 | {now_str}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;padding:24px;min-height:100vh">
  <h1 style="font-size:20px;font-weight:500;margin-bottom:4px">📊 ATR 신호 체커</h1>
  <p style="font-size:13px;color:#8b949e;margin-bottom:4px">동적 자산배분 전략 | QLD · SPTL · 코스피</p>
  <p style="font-size:12px;color:#8b949e;margin-bottom:20px">업데이트: {now_str} | 환율: ₩{KRW_RATE:,}/달러</p>
  {summary_banner()}
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:20px">
    {cards}
  </div>
  <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px">
    <div style="font-size:13px;color:#8b949e;margin-bottom:10px">오늘의 매매 신호 요약</div>
    {sig_rows()}
  </div>
</body>
</html>"""

with open("signal.html", "w", encoding="utf-8") as f:
    f.write(html)

print("\n💾 signal.html 저장 완료!")
try:
    from google.colab import files
    files.download("signal.html")
    print("⬇️  자동 다운로드!")
except:
    print("💡 signal.html 파일을 직접 열어주세요.")
