from re import S
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
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import math
from recognizing import recognize_candlestick
from graph_ind import gi
import asyncio

RSI_PERİOD = 14
OVERSOLD_TRESHOLD = 30
OVERBOUGHT_TRESHOLD = 70
TRADE_QUANTİTY = 0.05
TRADE_SYMBOL = "MATICUSDT"
SOCKET = "wss://stream.binance.com:9443/ws/tvkbusd@aggTrade"


def progressbar():
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        import time
    # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'


class StreamlitView():

    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.candles = self.client.get_klines(
        symbol=TRADE_SYMBOL, interval=Client.KLINE_INTERVAL_15MINUTE)
        self.get_candle_data()

    @property
    def get_supertrend(self):

        atr_period, atr_multiplier = 10,2
        try:
            atr = talib.ATR(self.high, self.low, self.close, atr_period)
        except:
            return False, False

        previous_final_upperband = 0
        previous_final_lowerband = 0
        final_upperband = 0
        final_lowerband = 0
        previous_close = 0
        previous_supertrend = 0
        supertrend = []
        supertrendc = 0

        for i in range(0, len(self.close)):
            if np.isnan(self.close[i]):
                pass
            else:
                highc = self.high[i]
                lowc = self.low[i]
                atrc = atr[i]
                closec = self.close[i]

                if math.isnan(atrc):
                    atrc = 0

                basic_upperband = (highc + lowc) / 2 + atr_multiplier * atrc
                basic_lowerband = (highc + lowc) / 2 - atr_multiplier * atrc

                if basic_upperband < previous_final_upperband or previous_close > previous_final_upperband:
                    final_upperband = basic_upperband
                else:
                    final_upperband = previous_final_upperband

                if basic_lowerband > previous_final_lowerband or previous_close < previous_final_lowerband:
                    final_lowerband = basic_lowerband
                else:
                    final_lowerband = previous_final_lowerband

                if previous_supertrend == previous_final_upperband and closec <= final_upperband:
                    supertrendc = final_upperband
                else:
                    if previous_supertrend == previous_final_upperband and closec >= final_upperband:
                        supertrendc = final_lowerband
                    else:
                        if previous_supertrend == previous_final_lowerband and closec >= final_lowerband:
                            supertrendc = final_lowerband
                        elif previous_supertrend == previous_final_lowerband and closec <= final_lowerband:
                            supertrendc = final_upperband

                supertrend.append(supertrendc)

                previous_close = closec

                previous_final_upperband = final_upperband

                previous_final_lowerband = final_lowerband

                previous_supertrend = supertrendc

        return supertrend

    @property
    def get_rsi(self):
        rsi = talib.RSI(self.close, RSI_PERİOD)
        return rsi

    @property
    def get_stokrsi(self):
        rsi = self.get_rsi(self.close)
        stochrsif, stochrsis = talib.STOCH(rsi, rsi, rsi, fastk_period=14, slowk_period=3, slowd_period=3)
        return stochrsif, stochrsis

    @property
    def get_bband(self):
        uband,mband,lband = talib.BBANDS(self.close)
        return uband,mband,lband

    @property
    def get_ema(self):
        ema = talib.EMA(self.close)
        return ema
    
    @property
    def candle_chart(self):
        candle_chart = go.Figure(data=go.Candlestick(x=self.tarih, close=self.close, 
        open=self.acilis, high=self.high, low=self.low,name=TRADE_SYMBOL))
        return candle_chart

    @property
    def get_macd(self):
        macd, macdsignal, macdhist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
        return macd,macdsignal,macdhist

    @staticmethod
    def control_key_in_gi(name):
        if name in gi.keys():
            return gi[name]["name"] ,gi[name]["type"] 
        else:
            raise KeyError("Girdiğiniz 'indikatör_name' parametresi 'self.signals' fonksiyonunda bulunmuyor.")

    def get_candle_data(self,last_price=False):
        df = StreamlitView.candles_to_df(self.candles)
        if last_price:
            df["self.close"].values[:-1]
        else:
            self.close = df["self.close"].values[:-1].astype(np.float64)
            self.high = df["EnYuksek"].values[:-1].astype(np.float64)
            self.low = df["EnDusuk"].values[:-1].astype(np.float64)
            self.acilis = df["Acilis"].values[:-1].astype(np.float64)
            self.tarih = df["AcilisZamani"].apply(lambda x: datetime.fromtimestamp(x / 1e3))

            return self.tarih, self.acilis, self.close, self.high, self.low

    @staticmethod
    def candles_to_df(candles):
        candle_df = pd.DataFrame(candles)
        candle_df.columns = ["AcilisZamani", "Acilis", "EnYuksek", "EnDusuk", "self.close", "Volume", "KapanisZamani",
                            "QuoteAssetVolume", "NumberOfTrades", "TakerBuyBaseAssetVolume", "TakerBuyQuoteAssetVolume", "Ignore"]
        return candle_df

    def signal_to_text(self,indicator_name,signal):
        prettyname,type = self.control_key_in_gi(indicator_name)
        if signal == 1:
            message = f":dollar::moneybag: {prettyname} {type} **Al sinyali üretiyor**."
            st.info(message)
            return message

        elif signal == 2:
            message = f":dollar::moneybag: {prettyname} {type} **Sat sinyali üretiyor**."
            st.success(message)
            return message

        else:
            message = f":pensive: Üzgünüm **almak veya satmak için biraz daha beklemelisin** {prettyname} {type} **net bir sinyal üretmiyor**."
            st.info(message)
            return message
    
    @staticmethod
    def view_side_bar():
        add_selectbox = st.sidebar.selectbox(
        "Lütfen bir sekme seçin.",
        ("Twitter", "Kripto Para Analizi", "Haberler")
        )
        return add_selectbox

    def lastprev_close_indicator(self,indicator_results):
        last_close = self.close[-1]
        previous_close = self.close[-2]
        last_indicator_result = indicator_results[-1]
        previous_indicator_result = indicator_results[-2]

        return last_close, previous_close, last_indicator_result, previous_indicator_result

    def signals(self,indicator_name,indicator_results):
        """ 
        ! RETURN 1 al sinyali
        ! RETURN 2 sat sinyali
        ! RETURN False hiçbirşey yapma sinyali

        """
        self.control_key_in_gi(indicator_name)
        
        if indicator_name == "supertrend":
            lc, pc,lic,pic = self.lastprev_close_indicator(indicator_results=self.get_supertrend)
            #Trend yeni döndüyse son 3 kapanışta döndüyse
            #Alım Sinyali
            
            ispclowest = list(filter(lambda x: x < pc, self.close[-3:]))
            #Satış Sinyali
            ispichighest = list(filter(lambda x: x > pic, self.close[-3:]))
        

            if lc > lic and len(ispclowest)>=1:
                result =  1

            elif lc < lic and len(ispichighest)>=1:
                result = 2 

            else: result = False

        elif indicator_name == "rsi":
            last_rsi = indicator_results[-1]
            if last_rsi <= 35:
                result =  1
            
            elif last_rsi > 68:
                result =  2

            else: result = False

        elif indicator_name == "macd":
            macd,macdsignal,macdhist = indicator_results
            last_macd = macd[-1]
            last_macd_signal = macdsignal[-1]

            previous_macd = macd[-2]
            previous_macd_signal = macdsignal[-2]

            macd_cross_up = last_macd > last_macd_signal and previous_macd < previous_macd_signal
            if macd_cross_up:
                result =  1 
            
            else: result = False

        elif indicator_name == "ema":
            last_ema = indicator_results[-1]
            last_close = self.close[-1]

            if last_close  < (last_close - last_close * 0.05):
                result = 1 

            elif last_close > (last_ema + last_ema * 0.05):
                result = 2
            
            else: result = False

        elif indicator_name == "bband":
            uband,mband,lband = indicator_results
            last_uband = uband[-1]
            last_lband = lband[-1]
            last_close = self.close[-1]

            if last_lband > last_close:
                result = 1

            elif last_uband < last_close:
                result = 2

            else: result = False

        signal_text = self.signal_to_text(indicator_name=indicator_name,signal=result)
        return result,signal_text

    @staticmethod
    def page_config():
        st.set_page_config(
        page_title="Ex-stream-ly Cool App",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="auto",
        )
        StreamlitView.view_side_bar()
        st.title("My Binance Trading Bot")

    def get_prices_from_symbol(self,symbol):
        theta_price =  self.client.get_aggregate_trades(symbol=symbol)
        return theta_price[-1]["p"]

    def ask_symbol(self):
        symbol = st.text_input(label="Fiyatını Öğrenmek İçin Sembol Giriniz",value="", max_chars=10, key=None, type='default')
        if symbol:
            try:
                st.write(f"{symbol} piyasa değeri ",self.get_prices_from_symbol(symbol=symbol))
            except Exception:
                raise Exception("Üzgünüz Böyle Bir Sembol Bulamadık.")

    def supertrend_chart(self):
        super_trend_fig = go.Figure(data=go.Candlestick(
        x=self.tarih, close=self.close, open=self.acilis, high=self.high, low=self.low,name=TRADE_SYMBOL))
        super_trend_fig.add_trace(go.Scatter(x=self.tarih, y=self.get_supertrend, name='SuperTrend'))
        return super_trend_fig
        
    def rsi_chart(self):
        rsi = self.get_rsi
        fig = px.line(rsi,self.tarih[:-1],rsi)
        return fig

    def ema_chart(self):
        ema = self.get_ema
        ema_fig = go.Figure(data=go.Candlestick(
            x=self.tarih, close=self.close, open=self.acilis, high=self.high, low=self.low,name=TRADE_SYMBOL))
        ema_fig.add_trace(go.Scatter(x=self.tarih, y=ema, name='EMA',line=go.scatter.Line(color="black")))
        return ema_fig     

    def bband_chart(self):
        uband,mband,lband = self.get_bband
        bband_fig = go.Figure(data=go.Candlestick(
        x=self.tarih, close=self.close, open=self.acilis, high=self.high, low=self.low,name=TRADE_SYMBOL))
        bband_fig.add_trace(go.Scatter(x=self.tarih, y=uband, name='Upper-Band',line=go.scatter.Line(color="black")))
        bband_fig.add_trace(go.Scatter(x=self.tarih, y=mband, name='Middle-Band',line=go.scatter.Line(color="black")))
        bband_fig.add_trace(go.Scatter(x=self.tarih, y=lband, name='Lower-Band',line=go.scatter.Line(color="black")))
        return bband_fig

    def macd_chart(self):
        macd = self.get_macd
        self.signals("macd",macd)

    def generate_fig(self):
        pass

    def show_chart(self,name):
        prettyname, type = self.control_key_in_gi(name)

        if name == "candlestick":
            st.subheader(f"{TRADE_SYMBOL} {prettyname} {type}")
            st.plotly_chart(self.candle_chart)

        elif name == "supertrend":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.supertrend_chart())
            self.signals(indicator_name="supertrend",indicator_results=self.get_supertrend)

        elif name == "rsi":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.rsi_chart())
            self.signals("rsi",self.get_rsi)

        elif name == "bband":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.bband_chart())
            self.signals("bband",self.get_bband)

        elif name == "ema":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.ema_chart())
            self.signals("ema",self.get_ema)

        elif name == "stokrsi":
            pass

        elif name == "macd":
            pass

        else:
                st.subheader(f"{prettyname} {type}")

    def askSymbol(self):
        title = st.text_input("Fiyatını öğrenmek istediğiniz sembolü giriniz",max_chars=7)
        if title:
            price = self.client.get_aggregate_trades(symbol='BNBBTC')

        st.write(price, title)

    def run(self):

        StreamlitView.view_side_bar()
        self.ask_symbol()
        self.show_chart("candlestick")
        self.show_chart("supertrend")
        self.show_chart("rsi")
        self.show_chart("bband")
        self.show_chart("ema")

s = StreamlitView()
s.run()