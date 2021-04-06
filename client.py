
import numpy as np
import talib
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
import requests
from pprint import pprint
import time

RSI_PERİOD = 14
OVERSOLD_TRESHOLD = 30
OVERBOUGHT_TRESHOLD = 70
TRADE_QUANTİTY = 0.05

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
    is_executed_golden_cross = False

    def __init__(self):
        self.TRADE_SYMBOL = "SUPERUSDT"
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.get_candle_data()
        self.money = 500
        self.amount = 0
        self.coin_list = ["TVKBUSD", "SUPERUSDT",
                          "ETHUSDT", "ALGOUSDT", "SOLUSDT"]
        self.winners = []

    @property
    def get_supertrend(self):

        atr_period, atr_multiplier = 10, 2
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
        stochrsif, stochrsis = talib.STOCH(
            rsi, rsi, rsi, fastk_period=14, slowk_period=3, slowd_period=3)
        return stochrsif, stochrsis

    @property
    def get_bband(self):
        uband, mband, lband = talib.BBANDS(self.close)
        return uband, mband, lband

    @property
    def get_ema(self):
        # ema = talib.EMA(self.close)
        ema = talib.EMA(self.close)
        return ema

    @property
    def candle_chart(self, trade_symbol=None):
        if trade_symbol:
            candle_chart = go.Figure(data=go.Candlestick(x=self.tarih, close=self.close,
                                                         open=self.acilis, high=self.high, low=self.low, name=self.TRADE_SYMBOL))
            return candle_chart

        else:

            candle_chart = go.Figure(data=go.Candlestick(x=self.tarih, close=self.close,
                                                         open=self.acilis, high=self.high, low=self.low, name=self.TRADE_SYMBOL))
            return candle_chart

    @property
    def get_macd(self):
        macd, macdsignal, macdhist = talib.MACD(
            self.close, fastperiod=12, slowperiod=26, signalperiod=9)
        return macd, macdsignal, macdhist

    def candles(self, trade_symbol=None):
        if trade_symbol:
            candless = self.client.get_klines(
                symbol=trade_symbol, interval=Client.KLINE_INTERVAL_4HOUR)
            return candless

        else:
            candless = self.client.get_klines(
                symbol=self.TRADE_SYMBOL, interval=Client.KLINE_INTERVAL_4HOUR)
            return candless

    def get_triple_ma(self, close=None):
        if close is not None:
            seventeen = talib.MA(close, 17)
            fifty = talib.MA(close, 50)
            hundred = talib.MA(close, 200)
            return seventeen, fifty, hundred

        else:
            seventeen = talib.MA(self.close, 17)
            fifty = talib.MA(self.close, 50)
            hundred = talib.MA(self.close, 200)
            return seventeen, fifty, hundred

    @staticmethod
    def control_key_in_gi(name):
        if name in gi.keys():
            return gi[name]["name"], gi[name]["type"]
        else:
            raise KeyError(
                "Girdiğiniz 'indikatör_name' parametresi 'self.signals' fonksiyonunda bulunmuyor.")

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

    @staticmethod
    def candles_to_df(candles):
        candle_df = pd.DataFrame(candles)
        candle_df.columns = ["AcilisZamani", "Acilis", "EnYuksek", "EnDusuk", "Kapanis", "Volume", "KapanisZamani",
                             "QuoteAssetVolume", "NumberOfTrades", "TakerBuyBaseAssetVolume", "TakerBuyQuoteAssetVolume", "Ignore"]
        return candle_df

    @staticmethod
    def view_side_bar():
        add_selectbox = st.sidebar.selectbox(
            "Lütfen bir sekme seçin.",
            ("Twitter", "Kripto Para Analizi", "Coindesk", "Coinbase")
        )
        return add_selectbox

    def get_news_from_marketcal(self):
        url = "https://developers.coinmarketcal.com/v1/events"
        querystring = {"max": "10", "coins": "terra-virtua-kolect"}
        payload = ""
        headers = {
            'x-api-key': config.MARKET_CAL_API_KEY,
            'Accept-Encoding': "deflate, gzip",
            'Accept': "application/json"
        }
        response = requests.request(
            "GET", url, data=payload, headers=headers, params=querystring)
        # response = requests.request("GET", url, headers=headers)
        print(response.text)

    def get_candle_data(self, last_price=False, testing_number: int = -1, df=None):

        if df is not None:
            close = df["Kapanis"].values[:testing_number].astype(np.float64)
            high = df["EnYuksek"].values[:testing_number].astype(np.float64)
            low = df["EnDusuk"].values[:testing_number].astype(np.float64)
            acilis = df["Acilis"].values[:testing_number].astype(np.float64)
            if testing_number != -1:
                self.tarih = df["AcilisZamani"].apply(
                    lambda x: datetime.fromtimestamp(x / 1e3)).values[:testing_number+1]
            if testing_number == -1:

                tarih = df["AcilisZamani"].apply(
                    lambda x: datetime.fromtimestamp(x / 1e3))[:-1]

            return tarih, acilis, close, high, low

        else:
            df = self.candles_to_df(self.candles())
            if last_price:
                return float(df["Kapanis"].values[:testing_number][-1])

            else:
                self.close = df["Kapanis"].values[:testing_number].astype(
                    np.float64)
                self.high = df["EnYuksek"].values[:testing_number].astype(
                    np.float64)
                self.low = df["EnDusuk"].values[:testing_number].astype(
                    np.float64)
                self.acilis = df["Acilis"].values[:testing_number].astype(
                    np.float64)
                if testing_number != -1:
                    self.tarih = df["AcilisZamani"].apply(
                        lambda x: datetime.fromtimestamp(x / 1e3)).values[:testing_number+1]
                if testing_number == -1:

                    self.tarih = df["AcilisZamani"].apply(
                        lambda x: datetime.fromtimestamp(x / 1e3))[:-1]

                return self.tarih, self.acilis, self.close, self.high, self.low

    def signal_to_text(self, indicator_name, signal):
        prettyname, type = self.control_key_in_gi(indicator_name)
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

    def lastprev_close_indicator(self, indicator_results):
        last_close = self.close[-1]
        previous_close = self.close[-2]
        last_indicator_result = indicator_results[-1]
        previous_indicator_result = indicator_results[-2]

        return last_close, previous_close, last_indicator_result, previous_indicator_result

    def signals(self, indicator_name, print_signal=True, closes=None):
        """ 
        ! RETURN 1 al sinyali
        ! RETURN 2 sat sinyali
        ! RETURN False hiçbirşey yapma sinyali
        ! Closes Golden cross bulmak için kullanılır. Her coinin kapanislari aynı olmadığından dolayı kapanislar dışarıdan verilir.
        ! print_signal parametresi Golden cross ararken ekrana boş yere sinyal yazılmasın diye konulmuştur.

        """
        self.control_key_in_gi(indicator_name)

        if indicator_name == "supertrend":
            lc, pc, lic, pic = self.lastprev_close_indicator(
                indicator_results=self.get_supertrend)
            # Trend yeni döndüyse son 3 kapanışta döndüyse
            # Alım Sinyali

            ispclowest = list(filter(lambda x: x < pc, self.close[-3:]))
            # Satış Sinyali
            ispichighest = list(filter(lambda x: x > pic, self.close[-3:]))

            if lc > lic and len(ispclowest) >= 1:
                result = 1

            elif lc < lic and len(ispichighest) >= 1:
                result = 2

            else:
                result = False

        elif indicator_name == "rsi":
            last_rsi = self.get_rsi[-1]
            if last_rsi <= 45:
                result = 1

            elif last_rsi > 68:
                result = 2

            else:
                result = False

        elif indicator_name == "macd":
            macd, macdsignal, macdhist = self.get_macd
            last_macd = macd[-1]
            last_macd_signal = macdsignal[-1]

            previous_macd = macd[-2]
            previous_macd_signal = macdsignal[-2]

            macd_cross_up = last_macd > last_macd_signal and previous_macd < previous_macd_signal
            if macd_cross_up:
                result = 1

            else:
                result = False

        elif indicator_name == "ema":
            last_ema = self.get_ema[-1]
            last_close = self.close[-1]

            if last_close < (last_close - last_close * 0.05):
                result = 1

            elif last_close > (last_ema + last_ema * 0.05):
                result = 2

            else:
                result = False

        elif indicator_name == "bband":
            uband, mband, lband = self.get_bband
            last_uband = uband[-1]
            last_lband = lband[-1]
            last_close = self.close[-1]

            if last_lband > last_close:
                result = 1

            elif last_uband < last_close:
                result = 2

            else:
                result = False

        elif indicator_name == "tripleMA":

            if closes is not None:
                seventeen, fifty, hundred2 = self.get_triple_ma(closes)
                last_seventeen = seventeen[-1]
                prev_seventeen = seventeen[-2]
                lastfifty = fifty[-1]
                previousfifty = fifty[-2]
                lasthundred2 = hundred2[-1]
                previoushundred2 = hundred2[-2]

            else:
                seventeen, fifty, hundred2 = self.get_triple_ma()
                last_seventeen = seventeen[-1]
                prev_seventeen = seventeen[-2]
                lastfifty = fifty[-1]
                previousfifty = fifty[-2]
                lasthundred2 = hundred2[-1]
                previoushundred2 = hundred2[-2]

            # MA-50 MA-200 ü keserse
            # if previousfifty < previoushundred2 and lastfifty > lasthundred2:
            #     result = 1
            last_three_period = hundred2[-4:]
            is_signal = False
            for index, i in enumerate(last_three_period):
                if -len(last_three_period)+index+1 == 0:
                    break

                if i > seventeen[-len(last_three_period)+index] and i < seventeen[-len(last_three_period)+index+1]:
                    print(i, seventeen[-len(last_three_period)+index])
                    print(i, seventeen[-len(last_three_period)+index+1])
                    print(is_signal)
                    print("\n"*2)
                    is_signal = True

            if is_signal:
                result = 1

            elif previousfifty > previoushundred2 and lastfifty < lasthundred2:
                result = 2

            else:
                result = False

        if print_signal:
            self.signal_to_text(indicator_name=indicator_name, signal=result)

        return result

    def get_prices_from_symbol(self, symbol):
        theta_price = self.client.get_aggregate_trades(symbol=symbol)
        return theta_price[-1]["p"]

    def ask_symbol(self):
        symbol = st.text_input(label="Piyasa Değerini Öğrenmek İçin Sembol Giriniz",
                               value="", max_chars=10, key=None, type='default')
        if symbol and symbol.upper() != self.TRADE_SYMBOL:
            self.TRADE_SYMBOL = symbol.upper()
            self.get_candle_data()
            try:
                st.write(
                    f"**{symbol.upper()}** piyasa değeri **{self.get_prices_from_symbol(symbol=symbol.upper())}**")
            except Exception as e:
                raise Exception("Üzgünüz Böyle Bir Sembol Bulamadık.")

    def supertrend_chart(self):
        super_trend_fig = self.candle_chart
        super_trend_fig.add_trace(go.Scatter(
            x=self.tarih, y=self.get_supertrend, name='SuperTrend'))
        return super_trend_fig

    def rsi_chart(self):
        rsi = self.get_rsi
        data = {
            "rsi": rsi,
            "tarih": self.tarih
        }
        df = pd.DataFrame(data=data)
        fig = px.line(df, x="tarih", y="rsi")
        return fig

    def ema_chart(self):
        ema = self.get_ema
        ema_fig = self.candle_chart
        ema_fig.add_trace(go.Scatter(x=self.tarih, y=ema,
                                     name='EMA', line=go.scatter.Line(color="black")))
        return ema_fig

    def triple_ma_chart(self):
        seventeen, fifty, hundred = self.get_triple_ma()
        fig = self.candle_chart
        fig.add_trace(go.Scatter(x=self.tarih, y=seventeen,
                                 name='17', line=go.scatter.Line(color="orange")))
        fig.add_trace(go.Scatter(x=self.tarih, y=fifty, name='50',
                                 line=go.scatter.Line(color="purple")))
        fig.add_trace(go.Scatter(x=self.tarih, y=hundred,
                                 name='100', line=go.scatter.Line(color="blue")))
        return fig

    def bband_chart(self):
        uband, mband, lband = self.get_bband
        bband_fig = self.candle_chart
        bband_fig.add_trace(go.Scatter(
            x=self.tarih, y=uband, name='Upper-Band', line=go.scatter.Line(color="black")))
        bband_fig.add_trace(go.Scatter(
            x=self.tarih, y=mband, name='Middle-Band', line=go.scatter.Line(color="black")))
        bband_fig.add_trace(go.Scatter(
            x=self.tarih, y=lband, name='Lower-Band', line=go.scatter.Line(color="black")))
        return bband_fig

    def macd_chart(self):
        macd = self.get_macd
        self.signals("macd", macd)

    def generate_fig(self):
        pass

    def show_chart(self, name):
        prettyname, type = self.control_key_in_gi(name)
        signal = None
        if name == "candlestick":
            st.subheader(f"{self.TRADE_SYMBOL} {prettyname} {type}")
            st.plotly_chart(self.candle_chart)

        elif name == "supertrend":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.supertrend_chart())
            signal = self.signals(indicator_name="supertrend")

        elif name == "rsi":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.rsi_chart())
            signal = self.signals("rsi")

        elif name == "bband":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.bband_chart())
            signal = self.signals("bband")

        elif name == "ema":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.ema_chart())
            signal = self.signals("ema")

        elif name == "stokrsi":
            pass

        elif name == "macd":
            pass

        elif name == "tripleMA":
            st.subheader(f"{prettyname} {type}")
            st.plotly_chart(self.triple_ma_chart())
            signal = self.signals("tripleMA")

        else:
            st.subheader(f"{prettyname} {type}")

        return signal

    def show_orderbook(self):
        tickers = self.client.get_order_book(symbol=self.TRADE_SYMBOL)
        df = pd.DataFrame(tickers)

        print(df)

    def buy(self, testing_number):
        if self.money >= 500:
            lp = self.get_candle_data(
                last_price=True, testing_number=testing_number)
            self.amount = self.money / lp
            self.money = 0

    def sell(self, testing_number):
        if self.amount:
            lp = self.get_candle_data(
                last_price=True, testing_number=testing_number)
            st.write(lp)
            self.money = self.money + self.amount * lp
            self.amount = 0

    def show_wallet(self):
        print(f"Cüzdanınızda bulunan para miktarı {self.money} $")
        print(
            f"Cüzdanınızda bulunan {self.TRADE_SYMBOL}, {self.amount} kadardır.")

    def testing(self):
        # ma perioddan dolayı
        for i in range(200, 500):
            self.get_candle_data(testing_number=i)
            # self.view_side_bar()
            # self.ask_symbol()
            # self.show_chart("candlestick")
            lp = self.get_candle_data(last_price=True, testing_number=i)
            st.write(lp)
            s1 = self.signals("supertrend")
            s2 = self.signals("rsi")
            # s3 = self.signals("bband")
            # s4 = self.signals("ema")
            s5 = self.signals("tripleMA")
            signals = [s1, s2, s5]
            buy_sig = list(filter(lambda x: x == 1, signals))
            sell_sig = list(filter(lambda x: x == 2, signals))

            print(f"buy_sig= {len(buy_sig)}")
            print(f"sell_sig= {len(sell_sig)}")
            if len(buy_sig) > 0 and len(buy_sig) > len(sell_sig):
                self.buy(testing_number=i)

            elif len(sell_sig) > 0 and len(sell_sig) > len(buy_sig):
                self.sell(testing_number=i)

            self.show_wallet()

        self.show_wallet()

    def ask_symbol_and_golden_crosses(self):
        col1, col2 = st.beta_columns(2)
        with col1:
            self.ask_symbol()
        with col2:
            self.show_golden_crosses()

    def run(self):
        self.view_side_bar()
        self.ask_symbol_and_golden_crosses()
        self.show_chart("candlestick")
        self.show_chart("supertrend")
        self.show_chart("rsi")
        self.show_chart("bband")
        self.show_chart("ema")
        self.show_chart("tripleMA")

    def find_golden_cross(self, liste=None):
        if liste is not None:
            self.coin_list = liste
            for i in self.coin_list:
                ohlc = self.candles(trade_symbol=i)
                df = self.candles_to_df(ohlc)
                tarih, acilis, close, high, low = self.get_candle_data(df=df)
                self.get_triple_ma(close=close)
                result = self.signals("tripleMA", print_signal=False)
                print(result)
                if result == 1:
                    self.TRADE_SYMBOL = i
            else:
                print("Golden Cross Bulunamadı")
        else:
            # self.coin_list = self.get_all_coins_list()[-500:]
            self.coin_list = ["UNIUSDT", "TVKBUSD", "SUPERUSDT"]

            for index, i in enumerate(self.coin_list):
                parites = ["BTC", "ETH", "BNB", "PAX", "SDS",
                           "SDC", "BRL", "AUD", "GBP", "EUR", "TRY"]
                if i[-3:] in parites:
                    continue

                if index % 10 == 0:
                    time.sleep(5)
                print(i)
                try:
                    ohlc = self.candles(trade_symbol=i)
                except Exception:
                    continue
                df = self.candles_to_df(ohlc)
                tarih, acilis, close, high, low = self.get_candle_data(df=df)
                result = self.signals(
                    "tripleMA", print_signal=False, closes=close)
                if result == 1:
                    print(f"Golden Cross Bulundu Symbol: {i}")
                    self.winners.append(i)

            StreamlitView.is_executed_golden_cross = True
            print(self.winners)
            return self.winners

    def get_all_coins_list(self):
        info = self.client.get_exchange_info()
        liste = [i["symbol"] for i in info["symbols"]]
        return liste

    def show_golden_crosses(self):

        if not StreamlitView.is_executed_golden_cross or len(self.winners) < 1:
            yes_or_no = st.sidebar.button("Go Golden Cross!")
            if yes_or_no:
                self.find_golden_cross()
                if len(self.winners) > 0:
                    add_selectbox = st.selectbox(
                        "Golden Cross Yapan Coinler", [
                            "TVKBUSD", "BTCUSDT", "ALGOUSDT"]
                    )
                    st.write(add_selectbox)
                    if add_selectbox is not None:
                        self.TRADE_SYMBOL = add_selectbox
                        self.get_candle_data()

    def json(self):
        data = [
            {
                "id": "0001",
                "type": "donut",
                "name": "Cake",
                "ppu": 0.55,
                "batters":
                        {
                            "batter":
                            [
                                {"id": "1001", "type": "Regular"},
                                {"id": "1002", "type": "Chocolate"},
                                {"id": "1003", "type": "Blueberry"},
                                {"id": "1004", "type": "Devil's Food"}
                            ]
                        },
                "topping":
                [
                                {"id": "5001", "type": "None"},
                                {"id": "5002", "type": "Glazed"},
                                {"id": "5005", "type": "Sugar"},
                                {"id": "5007", "type": "Powdered Sugar"},
                                {"id": "5006", "type": "Chocolate with Sprinkles"},
                                {"id": "5003", "type": "Chocolate"},
                                {"id": "5004", "type": "Maple"}
                        ]
            },
            {
                "id": "0002",
                "type": "donut",
                "name": "Raised",
                "ppu": 0.55,
                "batters":
                        {
                            "batter":
                            [
                                {"id": "1001", "type": "Regular"}
                            ]
                        },
                "topping":
                [
                                {"id": "5001", "type": "None"},
                                {"id": "5002", "type": "Glazed"},
                                {"id": "5005", "type": "Sugar"},
                                {"id": "5003", "type": "Chocolate"},
                                {"id": "5004", "type": "Maple"}
                        ]
            },
            {
                "id": "0003",
                "type": "donut",
                "name": "Old Fashioned",
                "ppu": 0.55,
                "batters":
                        {
                            "batter":
                            [
                                {"id": "1001", "type": "Regular"},
                                {"id": "1002", "type": "Chocolate"}
                            ]
                        },
                "topping":
                [
                                {"id": "5001", "type": "None"},
                                {"id": "5002", "type": "Glazed"},
                                {"id": "5003", "type": "Chocolate"},
                                {"id": "5004", "type": "Maple"}
                        ]
            }
        ]
        st.write(data)


if __name__ == "__main__":
    stLit = StreamlitView()
    stLit.run()
