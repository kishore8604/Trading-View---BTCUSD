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

# -----------------------
# Configuration
# -----------------------
BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSD"
PRODUCT_TYPE = "COIN-FUTURES"
INTERVAL = "2H"        # timeframe used in API call
LIMIT = 100
REFRESH_INTERVAL = 1   # seconds between animation frames (chart update frequency)

# Strategy (Pine) parameters (mapped from Pine inputs)
STRAT_A = 1.0                # sensitivity 'a'
STRAT_ATR_PERIOD = 10       # 'c'
STRAT_USE_HEIKIN = False    # 'h' — if True, use Heikin-Ashi close as src
STRAT_START_DATE = datetime(2025, 5, 1, tzinfo=timezone.utc)  # gating from Pine script

# API credentials (from your code) — keep safe
API_KEY = "bg_8a495e6040fd44c3032e50cf46546c9f"
SECRET_KEY = "565a7f1af310d7edb780e6d08d58e8f364475bba6456aec80f4fac38dc817ed7"
PASSPHRASE = "Gughan200"

# Globals for plotting
current_price = None
price_line = None
price_text = None
last_candle_update = None

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
    signature = generate_signature(timestamp, method, request_path, body)
    return {
        "Content-Type": "application/json",
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": PASSPHRASE,
        "locale": "en-US"
    }

# -----------------------
# Market data functions
# -----------------------
def get_historical_candles():
    """
    Fetch historical candles from Bitget.
    Returns a DataFrame indexed by tz-naive timestamps.
    """
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
    """
    Fetch current ticker price from Bitget.
    Adjust endpoint/params if your market/ticker differs.
    """
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
                # sometimes Bitget returns a different structure; be defensive
                d = data.get('data')
                if isinstance(d, dict) and 'last' in d:
                    return float(d['last'])
                print(f"API Error / Unexpected response format: {data}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching current price: {str(e)}")
        return None

# -----------------------
# Heikin-Ashi conversion
# -----------------------
def convert_to_heikin_ashi(df):
    """Convert regular candlesticks to Heikin Ashi candles (returns DataFrame with ha_* columns)."""
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    # Initialize ha_open list
    ha_open = [df['open'].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['ha_close'].iloc[i-1]) / 2.0)

    ha_df['ha_open'] = ha_open
    ha_df['ha_high'] = ha_df[['ha_open', 'ha_close', 'high']].max(axis=1)
    ha_df['ha_low']  = ha_df[['ha_open', 'ha_close', 'low']].min(axis=1)
    return ha_df

# -----------------------
# UT Bot strategy implementation (converted from Pine)
# -----------------------
def compute_ut_signals(df, a=1.0, c=10, h=False, start_date=None):
    """
    Compute UT Bot buy/sell signals following PineScript logic.
    Returns DataFrame with columns: src, stop, pos, buy, sell (index: df timestamps)
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # Source (ha_close if requested)
    if h:
        ha = convert_to_heikin_ashi(df)
        src = ha['ha_close'].copy()
    else:
        src = df['close'].copy()
        ha = convert_to_heikin_ashi(df)  # still compute HA for plotting offsets

    # True Range and ATR (Wilder's RMA via ewm alpha=1/c)
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use .bfill() to handle NaNs in early ATR
    atr = tr.ewm(alpha=1.0/float(c), adjust=False).mean().bfill()
    nLoss = a * atr

    # Initialize series
    stop = pd.Series(index=df.index, dtype='float64')
    pos = pd.Series(index=df.index, dtype='int64')
    pos.iloc[0] = 0
    stop.iloc[0] = src.iloc[0] - nLoss.iloc[0]

    # Iterate to compute trailing stop and pos
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

        # pos compute
        if (prev < prev_stop) and (cur > prev_stop):
            pos.iloc[i] = 1
        elif (prev > prev_stop) and (cur < prev_stop):
            pos.iloc[i] = -1
        else:
            pos.iloc[i] = int(pos.iloc[i-1])

    # Cross detection
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

    # Timezone-safe start_date gating:
    if start_date is not None:
        # Convert a tz-aware start_date to tz-naive UTC for safe comparison with df.index
        if getattr(start_date, 'tzinfo', None) is not None:
            start_date_naive = start_date.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            start_date_naive = start_date
        signals.loc[signals.index < start_date_naive, ['buy', 'sell']] = False

    return signals

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
        price_line = None
    if price_text is not None:
        try:
            price_text.remove()
        except Exception:
            pass

def prepare_chart(df, ax1, ax2, signals=None):
    """Draw Heikin-Ashi candles + volume, and plot only buy/sell markers if 'signals' provided."""
    global current_price

    ha_df = convert_to_heikin_ashi(df)

    ax1.clear()
    ax2.clear()

    # Candle drawing
    candle_width = 1 / 24.0
    for idx, row in ha_df.iterrows():
        color = '#2ecc71' if row['ha_close'] >= row['ha_open'] else '#e74c3c'
        # wick
        ax1.plot([idx, idx], [row['ha_low'], row['ha_high']], color=color, linewidth=1)
        # body
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

    # Volume
    ax2.bar(df.index, df['volume'], width=candle_width,
            color=['#2ecc71' if c >= o else '#e74c3c' for c, o in zip(ha_df['ha_close'], ha_df['ha_open'])])

    # Plot buy/sell signals (only markers)
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

    # Current price line will be drawn separately in update function
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
# Animation update function
# -----------------------
def update_chart(frame, df_holder, ax1, ax2, fig):
    """
    Animation callback. df_holder is a dict{'df': DataFrame} so we can replace its value.
    """
    global current_price, last_candle_update, price_line, price_text

    # Update current price
    new_price = get_current_price()
    if new_price is not None:
        current_price = new_price

    now = datetime.now(timezone.utc)
    # Refresh candles every 3 hours (as per your original logic)
    if last_candle_update is None or (now - last_candle_update) >= timedelta(hours=3):
        print("\nFetching new candle data...")
        new_df = get_historical_candles()
        if new_df is not None:
            df_holder['df'] = new_df
            last_candle_update = now

    df = df_holder['df']
    if df is None or df.empty:
        return []

    # Compute signals (recompute on each chart refresh to ensure up-to-date markers)
    signals = compute_ut_signals(df, a=STRAT_A, c=STRAT_ATR_PERIOD, h=STRAT_USE_HEIKIN, start_date=STRAT_START_DATE)

    # Prepare chart drawing buy/sell markers
    prepare_chart(df, ax1, ax2, signals=signals)

    # Remove previous price line/text (if any) then draw new
    remove_price_line_and_text(ax1)
    if current_price is not None:
        try:
            price_line = ax1.axhline(y=current_price, color='cyan', linestyle='--', linewidth=1, alpha=0.8)
            # place the text slightly to the right of the last x coordinate
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

    # Initial signal computation
    signals = compute_ut_signals(df, a=STRAT_A, c=STRAT_ATR_PERIOD, h=STRAT_USE_HEIKIN, start_date=STRAT_START_DATE)

    # Setup plotting
    # plt.style.use('dark_background')
    # fig = plt.figure(figsize=(14, 10))
    # gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    # ax1 = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])

    # prepare_chart(df, ax1, ax2, signals=signals)

    # # Hold df in a mutable object for the animation callback
    # df_holder = {'df': df}

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