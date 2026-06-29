"""
notify_telegram.py — push the latest predict signal to your phone via a Telegram bot.

Reads the last row of btc_predict_cross_nf_signals.csv and posts it through the Telegram Bot
API. Run it right after btc_predict_cross_nf.py (run_predict.bat already does).

CONFIG (never commit your token):
  - env vars  TELEGRAM_BOT_TOKEN  and  TELEGRAM_CHAT_ID,  OR
  - a local   telegram_config.json  next to this file:  {"bot_token":"...","chat_id":"..."}
    (telegram_config.json is gitignored.)

Run:  py -3.10 notify_telegram.py            # send the latest signal
      py -3.10 notify_telegram.py --dry-run  # print the message without sending
"""
import os, json, sys, argparse
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIG_CSV = os.path.join(HERE, 'btc_predict_cross_nf_signals.csv')


def get_config():
    tok = os.environ.get('TELEGRAM_BOT_TOKEN'); chat = os.environ.get('TELEGRAM_CHAT_ID')
    cfg = os.path.join(HERE, 'telegram_config.json')
    if (not tok or not chat) and os.path.exists(cfg):
        c = json.load(open(cfg)); tok = tok or c.get('bot_token'); chat = chat or c.get('chat_id')
    return tok, chat


def build_message():
    r = pd.read_csv(SIG_CSV).iloc[-1]
    sig = str(r['signal'])
    icon = {'LONG': '🟢 LONG', 'SHORT': '🔴 SHORT', 'NEUTRAL': '⚪ NEUTRAL'}.get(sig, sig)
    conf = r.get('m2_confidence'); size = r.get('recommended_size')
    conf_s = f"{conf:.3f}" if pd.notna(conf) else "—"
    size_s = f"{size:.2f}×" if pd.notna(size) else "—"
    return (f"📊 BTC 4h signal — {icon}\n"
            f"bar:    {r['timestamp']} UTC\n"
            f"close:  {r['close']:,.0f}\n"
            f"P(long):{r['prob_long']:.3f}   regime: {r['vol_regime']}\n"
            f"M2 conf:{conf_s}   size: {size_s}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); args = ap.parse_args()
    if not os.path.exists(SIG_CSV):
        print("no signals CSV yet — run btc_predict_cross_nf.py first"); return
    msg = build_message()
    if args.dry_run:
        print("[dry-run] would send:\n" + msg); return
    tok, chat = get_config()
    if not (tok and chat):
        print("Telegram not configured (env vars or telegram_config.json). Message preview:\n" + msg); return
    import requests
    try:
        resp = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                             data={'chat_id': chat, 'text': msg}, timeout=20)
        print("sent ✓" if resp.ok else f"telegram error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"telegram send failed: {e}")


if __name__ == '__main__':
    main()
