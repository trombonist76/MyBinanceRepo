import websocket
import json
import numpy as np
import talib
import pprint
from binance.client import Client
from binance.enums import *
import config
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates



RSI_PERİOD = 14
OVERSOLD_TRESHOLD = 30
OVERBOUGHT_TRESHOLD = 70
TRADE_QUANTİTY = 0.05
TRADE_SYMBOL = "TVKBUSD"
SOCKET = "wss://stream.binance.com:9443/ws/tvkbusd@kline_1m"
client = Client(config.API_KEY,config.API_SECRET)

closes = []

def on_open(ws):
    print("open")

def on_close(ws):
    print("close")

def on_message(ws,message):
    global closes
    message = json.loads(message)
    candle = message["k"]
    is_candle_closed = candle["x"]
    int_to_time = lambda x: datetime.fromtimestamp(x / 1e3)
    if is_candle_closed:
        print("-"*30 + message['s'] + "-"*30)
        
        acilis_zamani = int_to_time(candle['t'])
        close = candle["c"]
        print(f"Açılış zamanı: {acilis_zamani}")
        print(f"Açılış değeri: {candle['o']}")
        print(f"Kapanış değeri: {close}")
        print(f"En yüksek değer: {candle['h']}")
        print(f"En düşük değer: {candle['l']}")

        closes.append(float(close))
        print(len(closes))
        if len(closes) > RSI_PERİOD:
            np_closes = np.array(closes)
            rsi = talib.RSI(np_closes,RSI_PERİOD)
            print(rsi)

ws = websocket.WebSocketApp(SOCKET,on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()


def candles_to_df(candles):
    candle_df = pd.DataFrame(candles)
    candle_df.columns = ["AcilisZamani","Acilis","EnYuksek","EnDusuk","Kapanis","Volume","KapanisZamani","QuoteAssetVolume","NumberOfTrades","TakerBuyBaseAssetVolume","TakerBuyQuoteAssetVolume","Ignore"]
    return candle_df

def int_to_datetime(candle_df,columns:list):
    for col in columns:
        candle_df[col] = candle_df[col].apply(lambda x: datetime.fromtimestamp(x / 1e3))
    return candle_df

def visualize_data(candle_df):
    candle_data = candle_df.loc[:, ['AcilisZamani', 'Acilis', 'EnYuksek', 'EnDusuk', 'Kapanis']]
    candle_data = candle_data.astype(float)
    fig, ax = plt.subplots()

    candlestick_ohlc(ax, candle_data.values, width=16.0, colorup='green', colordown='red', alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle('Daily Candlestick Chart of NIFTY50')

    fig.tight_layout()
    plt.show()


# candles = client.get_klines(symbol='BTCUSDT',interval="30m")
# df = candles_to_df(candles)
# visualize_data(df)