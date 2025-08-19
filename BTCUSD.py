import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import time
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import os
from matplotlib import animation
import numpy as np
import smtplib
from email.message import EmailMessage
import ssl

# -----------------------
# Configuration
# -----------------------
BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSD"
PRODUCT_TYPE = "COIN-FUTURES"
INTERVAL = "2H"        # timeframe used in API call
LIMIT = 100
REFRESH_INTERVAL = 1   # seconds between animation frames (chart update frequency)

# Strategy (Pine) parameters
STRAT_A = 1.0
STRAT_ATR_PERIOD = 10
STRAT_USE_HEIKIN = False
STRAT_START_DATE = datetime(2025, 5, 1, tzinfo=timezone.utc)

# -----------------------
# Recalculate options (from your UI)
# -----------------------
RECALCULATE_ON_TICK = False
RECALCULATE_AFTER_FILL = True

# -----------------------
# API credentials (your values)
# -----------------------
API_KEY = "bg_8a495e6040fd44c3032e50cf46546c9f"
SECRET_KEY = "565a7f1af310d7edb780e6d08d58e8f364475bba6456aec80f4fac38dc817ed7"
PASSPHRASE = "Gughan200"

# -----------------------
# SMTP / Email settings (fill these in)
# -----------------------
SMTP_SERVER = "smtp.gmail.com"      # e.g. smtp.gmail.com
SMTP_PORT = 587                     # 587 for STARTTLS, 465 for SSL
SMTP_USER = "gughanjojo@gmail.com"
SMTP_PASS = "hglv loqb jbmo jrou"
FROM_EMAIL = "gughanjojo@gmail.com"
TO_EMAILS = ["gughanjojo@gmail.com"]  # list of recipients

# -----------------------
# Globals for plotting and recalc control
# -----------------------
current_price = None
price_line = None
price_text = None
last_candle_update = None

# Recalculation control flags
force_recalc = False        # set True by notify_order_filled()
pending_order = False
last_recalc_time = None

# -----------------------
# Utility functions
# -----------------------
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_timestamp():
    return str(int(time.time() * 1000))

def generate_signature(timestamp, method, request_path, body=None):
    if body is None:
        body = ""
    message = timestamp + method.upper() + request_path + body
    signature = hmac.new(
        SECRET_KEY.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode('utf-8')

def get_headers(request_path, method="GET", body=None):
    timestamp = get_timestamp()
    signature = generate_signature(timestamp, method, body or "")
    return {
        "Content-Type": "application/json",
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "locale": "en-US"
    }

# -----------------------
# Email sending
# -----------------------
def send_email(subject: str, body: str, to_emails=None):
    """Send an email via configured SMTP server. Returns True on success."""
    if to_emails is None:
        to_emails = TO_EMAILS
    try:
        msg = EmailMessage()
        msg["From"] = FROM_EMAIL
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = subject
        msg.set_content(body)

        # Use STARTTLS
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"\n[email] Sent: {subject}")
        return True
    except Exception as e:
        print(f"\n[email] Failed to send email: {e}")
        return False

# -----------------------
# Market data functions
# -----------------------
def get_historical_candles():
    endpoint = "/api/v2/mix/market/history-candles"
    params = {
        "symbol": SYMBOL,
        "productType": PRODUCT_TYPE.lower(),
        "granularity": INTERVAL,
        "limit": str(LIMIT)
    }
    request_path = f"{endpoint}?{urlencode(params)}"

    try:
        headers = get_headers(request_path)
        response = requests.get(BASE_URL + request_path, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") not in (None, "00000"):
            print(f"API Error: {data.get('msg', 'Unknown error')}")
            return None

        candles = data.get("data", [])
        if not candles:
            print("Warning: Received empty candle data")
            return None

        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close",
            "volume", "quote_volume"
        ])

        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # tz-naive
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.set_index("timestamp", inplace=True)
        return df.sort_index()

    except Exception as e:
        print(f"Error fetching candles: {str(e)}")
        return None

def get_current_price():
    endpoint = "/api/mix/v1/market/ticker"
    params = {"symbol": f"{SYMBOL}_DMCBL"}
    request_path = f"{endpoint}?{urlencode(params)}"

    try:
        headers = get_headers(request_path)
        response = requests.get(BASE_URL + request_path, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == "00000" and 'data' in data and data['data'].get('last') is not None:
                return float(data['data']['last'])
            else:
                d = data.get('data')
                if isinstance(d, dict) and 'last' in d:
                    return float(d['last'])
                return None
        else:
            return None
    except Exception:
        return None

# -----------------------
# Heikin-Ashi conversion
# -----------------------
def convert_to_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    ha_open = [df['open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['ha_close'].iloc[i-1]) / 2.0)

    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = ha_df[['ha_open', 'ha_close', 'high']].max(axis=1)
    ha_df['ha_low']  = ha_df[['ha_open', 'ha_close', 'low']].min(axis=1)
    return ha_df

# -----------------------
# UT Bot strategy implementation
# -----------------------
def compute_ut_signals(df, a=1.0, c=10, h=False, start_date=None):
    if df is None or df.empty:
        return None

    df = df.copy()
    if h:
        ha = convert_to_heikin_ashi(df)
        src = ha['ha_close'].copy()
    else:
        src = df['close'].copy()
        ha = convert_to_heikin_ashi(df)

    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0/float(c), adjust=False).mean().bfill()
    nLoss = a * atr

    stop = pd.Series(index=df.index, dtype='float64')
    pos = pd.Series(index=df.index, dtype='int64')
    pos.iloc[0] = 0
    stop.iloc[0] = src.iloc[0] - nLoss.iloc[0]

    for i in range(1, len(df)):
        cur = src.iloc[i]
        prev = src.iloc[i-1]
        prev_stop = float(stop.iloc[i-1])
        cur_nloss = float(nLoss.iloc[i])

        if (cur > prev_stop) and (prev > prev_stop):
            s = max(prev_stop, cur - cur_nloss)
        elif (cur < prev_stop) and (prev < prev_stop):
            s = min(prev_stop, cur + cur_nloss)
        elif cur > prev_stop:
            s = cur - cur_nloss
        else:
            s = cur + cur_nloss

        stop.iloc[i] = s

        if (prev < prev_stop) and (cur > prev_stop):
            pos.iloc[i] = 1
        elif (prev > prev_stop) and (cur < prev_stop):
            pos.iloc[i] = -1
        else:
            pos.iloc[i] = int(pos.iloc[i-1])

    cross_up = (src > stop) & (src.shift(1) <= stop.shift(1))
    cross_down = (src < stop) & (src.shift(1) >= stop.shift(1))

    buy = (src > stop) & cross_up
    sell = (src < stop) & cross_down

    signals = pd.DataFrame({
        'src': src,
        'stop': stop,
        'pos': pos,
        'buy': buy,
        'sell': sell
    }, index=df.index)

    if start_date is not None:
        if getattr(start_date, 'tzinfo', None) is not None:
            start_date_naive = start_date.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            start_date_naive = start_date
        signals.loc[signals.index < start_date_naive, ['buy', 'sell']] = False

    return signals

# -----------------------
# Recalculation helpers
# -----------------------
def notify_order_filled():
    global force_recalc, pending_order
    force_recalc = True
    pending_order = False
    print("\n[notify_order_filled] order fill notified -> will force recalculation on next update.")

def place_order_simulation():
    global pending_order
    pending_order = True
    notify_order_filled()

# -----------------------
# Plotting helpers
# -----------------------
def remove_price_line_and_text(ax):
    global price_line, price_text
    if price_line is not None:
        try:
            price_line.remove()
        except Exception:
            pass
    if price_text is not None:
        try:
            price_text.remove()
        except Exception:
            pass

def prepare_chart(df, ax1, ax2, signals=None):
    global current_price

    ha_df = convert_to_heikin_ashi(df)

    ax1.clear()
    ax2.clear()

    candle_width = 1 / 24.0
    for idx, row in ha_df.iterrows():
        color = '#2ecc71' if row['ha_close'] >= row['ha_open'] else '#e74c3c'
        ax1.plot([idx, idx], [row['ha_low'], row['ha_high']], color=color, linewidth=1)
        rect_x = idx - timedelta(hours=(candle_width * 24.0 / 2.0))
        rect_height = abs(row['ha_close'] - row['ha_open'])
        rect_y = min(row['ha_open'], row['ha_close'])
        rect = Rectangle(
            (mdates.date2num(rect_x), rect_y),
            candle_width,
            rect_height,
            facecolor=color,
            edgecolor=color
        )
        ax1.add_patch(rect)

    ax2.bar(df.index, df['volume'], width=candle_width,
            color=['#2ecc71' if c >= o else '#e74c3c' for c, o in zip(ha_df['ha_close'], ha_df['ha_open'])])

    if signals is not None:
        buys = signals[signals['buy']]
        sells = signals[signals['sell']]

        price_range = df['high'].max() - df['low'].min()
        if price_range == 0:
            price_range = 1.0
        offset = price_range * 0.005

        if not buys.empty:
            ax1.scatter(buys.index.to_pydatetime(),
                        ha_df.loc[buys.index, 'ha_low'].values - offset,
                        marker='^', s=60, zorder=6, label='Buy')
        if not sells.empty:
            ax1.scatter(sells.index.to_pydatetime(),
                        ha_df.loc[sells.index, 'ha_high'].values + offset,
                        marker='v', s=60, zorder=6, label='Sell')

    ax1.set_title(f"BTC/USDT Heikin Ashi {INTERVAL} Chart | Live Price: {current_price or 'N/A'}")
    ax1.set_ylabel("Price (USDT)")
    ax2.set_ylabel("Volume")

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle=':', alpha=0.5)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    plt.tight_layout()

# -----------------------
# Animation update with email notification
# -----------------------
def update_chart(frame, df_holder, ax1, ax2, fig):
    global current_price, last_candle_update, price_line, price_text, force_recalc, last_recalc_time

    new_price = get_current_price()
    if new_price is not None:
        current_price = new_price

    now = datetime.now(timezone.utc)
    candles_updated = False

    # Refresh candles every 3 hours
    if last_candle_update is None or (now - last_candle_update) >= timedelta(hours=3):
        new_df = get_historical_candles()
        if new_df is not None:
            df_holder['df'] = new_df
            last_candle_update = now
            candles_updated = True

    df = df_holder['df']
    if df is None or df.empty:
        return []

    # Decide whether to recompute signals:
    should_recalc = False
    if RECALCULATE_ON_TICK:
        should_recalc = True
    if RECALCULATE_AFTER_FILL and force_recalc:
        should_recalc = True
        force_recalc = False
    if candles_updated:
        should_recalc = True
    if last_recalc_time is None:
        should_recalc = True

    # Previous signals for comparison (may be None)
    prev_signals = df_holder.get('signals', None)

    # If should_recalc, compute new signals
    if should_recalc:
        signals = compute_ut_signals(df, a=STRAT_A, c=STRAT_ATR_PERIOD, h=STRAT_USE_HEIKIN, start_date=STRAT_START_DATE)
        df_holder['signals'] = signals
        last_recalc_time = now

        # Notify emails for NEW signals only (if prev_signals exists)
        if prev_signals is not None and signals is not None:
            # New buys: True now, False before
            new_buys = signals['buy'] & (~prev_signals['buy'].reindex(signals.index, fill_value=False))
            new_sells = signals['sell'] & (~prev_signals['sell'].reindex(signals.index, fill_value=False))

            # df_holder['notified'] keeps track of already-sent signals to avoid duplicates
            if 'notified' not in df_holder:
                df_holder['notified'] = set()

            # send email for each new buy
            for ts in new_buys[new_buys].index:
                key = (ts.isoformat(), 'buy')
                if key in df_holder['notified']:
                    continue
                price_at_signal = float(signals.loc[ts, 'src'])
                subject = f"UT Bot BUY signal - {SYMBOL} {INTERVAL}"
                body = (f"UT Bot BUY signal detected\n\n"
                        f"Symbol: {SYMBOL}\n"
                        f"Time (UTC): {ts.isoformat()}\n"
                        f"Interval: {INTERVAL}\n"
                        f"Price: {price_at_signal:.2f}\n\n"
                        f"Recalculate on tick: {RECALCULATE_ON_TICK}\n"
                        f"Recalculate after fill: {RECALCULATE_AFTER_FILL}\n")
                send_email(subject, body)
                df_holder['notified'].add(key)

            # send email for each new sell
            for ts in new_sells[new_sells].index:
                key = (ts.isoformat(), 'sell')
                if key in df_holder['notified']:
                    continue
                price_at_signal = float(signals.loc[ts, 'src'])
                subject = f"UT Bot SELL signal - {SYMBOL} {INTERVAL}"
                body = (f"UT Bot SELL signal detected\n\n"
                        f"Symbol: {SYMBOL}\n"
                        f"Time (UTC): {ts.isoformat()}\n"
                        f"Interval: {INTERVAL}\n"
                        f"Price: {price_at_signal:.2f}\n\n"
                        f"Recalculate on tick: {RECALCULATE_ON_TICK}\n"
                        f"Recalculate after fill: {RECALCULATE_AFTER_FILL}\n")
                send_email(subject, body)
                df_holder['notified'].add(key)
    else:
        signals = df_holder.get('signals', None)

    # Draw chart
    prepare_chart(df, ax1, ax2, signals=signals)

    # Draw price line / text
    remove_price_line_and_text(ax1)
    if current_price is not None:
        try:
            price_line = ax1.axhline(y=current_price, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
            last_x = df.index[-1]
            price_text = ax1.text(last_x, current_price, f' {current_price:.2f}', color='cyan',
                                  va='center', ha='left', backgroundcolor='black', fontsize=8)
        except Exception:
            pass

    print(f"\rLast update: {now.strftime('%Y-%m-%d %H:%M:%S')} | Price: {current_price or 'N/A'}", end='')

    return []

# -----------------------
# Main
# -----------------------
def main():
    global current_price, last_candle_update

    print(f"Bitget {PRODUCT_TYPE} Heikin Ashi {INTERVAL} Chart with UT Bot signals")
    df = get_historical_candles()
    if df is None or df.empty:
        print("Failed to get initial candle data. Exiting.")
        return

    current_price = get_current_price()
    last_candle_update = datetime.now(timezone.utc)

    # initial signals
    # signals = compute_ut_signals(df, a=STRAT_A, c=STRAT_ATR_PERIOD, h=STRAT_USE_HEIKIN, start_date=STRAT_START_DATE)

    # plt.style.use('dark_background')
    # fig = plt.figure(figsize=(14, 10))
    # gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])

    # prepare_chart(df, ax1, ax2, signals=signals)

    # # holder to allow swapping df and signals inside animation & track notifications
    # df_holder = {'df': df, 'signals': signals, 'notified': set()}

    # ani = animation.FuncAnimation(
    #     fig, update_chart,
    #     fargs=(df_holder, ax1, ax2, fig),
    #     interval=REFRESH_INTERVAL * 1000,
    #     cache_frame_data=False
    # )

    # plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by user")
