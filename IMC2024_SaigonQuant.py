import collections
from collections import defaultdict
from datamodel import OrderDepth, TradingState, Order,ConversionObservation, Observation
from typing import List, Dict
import numpy as np
import math
import pandas as pd
import json
import statistics
import copy

class Trader:   
    def __init__(self):
        self.INF = int(1e9)
        self.amethysts_history = []
        self.starfruit_history = []
        self.starfruit_cache_size = 35
        self.amethysts_cache_size = 5 # 10
        self.starfruit_spread_cache = []
        self.position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
        self.POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.amethysts_default_price = 10_000
        self.orchid_coef = [-2.16359544e-03, 9.82450923e-03, -1.23079864e-02, 1.00442531e+00, 8.65723543e+00, -2.78822090e+01, 2.97898002e+01, -1.05648098e+01, 2.34006780e+02, -1.29033746e+03, 1.87744151e+03, -8.21110222e+02]
        self.orchid_intercept = 0.14551195562876273
        self.orchid_cache = []
        self.sunlight_cache = []
        self.humidity_cache = []
        self.gift_basket_std = 50
        self.etf_returns = []
        self.assets_returns = []
        self.chocolate_returns = []
        self.chocolate_estimated_returns = []
        self.strawberries_returns = []
        self.strawberries_estimated_returns = []
        self.roses_returns = []
        self.roses_estimated_returns = []
        self.coupon_return = []
        self.coupon_black_scholes_return = []
        self.coconut_returns = []
        self.coconut_predicted_return = []
        self.rhianna_buy = False
        self.rhianna_trade = False
        self.N = statistics.NormalDist(mu=0, sigma=1)

    def calculate_macd(self, prices):
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean() # 12-period EMA
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean() # 26-period EMA
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean() # signal line
        return macd.iloc[-1] - signal.iloc[-1]
    
    def calculate_volatility(self, prices):
        log_returns = np.log(prices / np.roll(prices, 1))[1:]
        return np.std(log_returns)
    
    def predict_amethysts_price(self, historical_prices):
        if len(historical_prices) < 30:
            return self.amethysts_default_price

        macd_value = self.calculate_macd(historical_prices)
        volatility = self.calculate_volatility(historical_prices)
        
        # If MACD value is positive and volatility is less than 5% of the mean price,predict the price to increase by 2%
        if macd_value > 0 and volatility < np.mean(historical_prices) * 0.05:
            predicted_price = int(round(historical_prices[-1] * 1.02))
        # If MACD value is negative or volatility is greater than 5% of the mean price, predict the price to decrease by 2%.
        elif macd_value < 0 or volatility > np.mean(historical_prices) * 0.05:
            predicted_price = int(round(historical_prices[-1] * 0.98))
        # Otherwise, maintain the same price as the last observed price.
        else:
            predicted_price = int(round(historical_prices[-1]))

        return predicted_price
    
    def predict_starfruit_price(self, cache, smoothing_level=0.2):
        # exponential smoothing/noise reduction data by calculating the exponentially weighted moving average of the data
        data = pd.Series(cache)
        predicted_price = data.ewm(alpha=smoothing_level, adjust=False).mean().iloc[-1]
        return int(round(predicted_price))

    def black_scholes(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.N.cdf(d1) - K * np.exp(-r * T) * self.N.cdf(d2)

    def vwap(self, order_depth):
        total_ask, total_bid = 0, 0
        ask_vol, bid_vol = 0, 0

        for ask, vol in order_depth.sell_orders.items():
            total_ask += ask * abs(vol)
            ask_vol += abs(vol)

        for bid, vol in order_depth.buy_orders.items():
            total_bid += bid * vol
            bid_vol += vol

        ask_price = total_ask / ask_vol
        bid_price = total_bid / bid_vol
        vwap_price = (ask_price + bid_price) / 2

        return vwap_price
    

    def compute_basket_orders(self, state: TradingState):
        products = ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
        position, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "GIFT_BASKET": []}

        for product in products:
            position[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        predicted_price = 4.0 * prices["CHOCOLATE"] + 6.0 * prices["STRAWBERRIES"] + prices["ROSES"]

        price_diff = prices["GIFT_BASKET"] - predicted_price

        self.etf_returns.append(prices["GIFT_BASKET"])
        self.assets_returns.append(price_diff)

        if len(self.etf_returns) < 100 or len(self.assets_returns) < 100:
            return orders
        
        # slow MA
        assets_rolling_mean = statistics.fmean(self.assets_returns[-200:])
        # fast MA
        assets_rolling_mean_fast = statistics.fmean(self.assets_returns[-100:])

        # Fine tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if assets_rolling_mean_fast > assets_rolling_mean + 4:
            limit_mult = 3
            limit_mult = min(limit_mult, self.POSITION_LIMIT["GIFT_BASKET"] - position["GIFT_BASKET"],
                             self.POSITION_LIMIT["GIFT_BASKET"])

            print("GIFT_BASKET position:", position["GIFT_BASKET"])
            print("BUY", "GIFT_BASKET", str(limit_mult) + "x", best_asks["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_asks["GIFT_BASKET"], limit_mult))

        elif assets_rolling_mean_fast < assets_rolling_mean - 4:
            limit_mult = -3
            limit_mult = max(limit_mult, -self.POSITION_LIMIT["GIFT_BASKET"] - position["GIFT_BASKET"],
                             -self.POSITION_LIMIT["GIFT_BASKET"])

            print("GIFT_BASKET position:", position["GIFT_BASKET"])
            print("SELL", "GIFT_BASKET", str(limit_mult) + "x", best_bids["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_bids["GIFT_BASKET"], limit_mult))

        return orders

    def compute_coupon_orders(self, state: TradingState):
        products = ["COCONUT_COUPON", "COCONUT"]
        position, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {"COCONUT_COUPON": [], "COCONUT": []}

        for product in products:
            position[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        S = prices["COCONUT"]
        K = 10000
        T = 250
        r = 0
        sigma = 0.01
        pred_price = self.black_scholes(S, K, T, r, sigma)

        self.coupon_return.append(prices["COCONUT_COUPON"])
        self.coupon_black_scholes_return.append(pred_price)

        self.coconut_returns.append(prices["COCONUT"])
        self.coconut_predicted_return.append(prices["COCONUT"])

        if len(self.coupon_return) < 2 or len(self.coupon_black_scholes_return) < 2:
            return orders

        coupon_rolling_mean = statistics.fmean(self.coupon_return[-200:])
        coupon_rolling_std = statistics.stdev(self.coupon_return[-200:])

        coupon_black_scholes_rolling_mean = statistics.fmean(self.coupon_black_scholes_return[-200:])
        coupon_black_scholes_rolling_std = statistics.stdev(self.coupon_black_scholes_return[-200:])

        if coupon_rolling_std != 0:
            coupon_zscore = (self.coupon_return[-1] - coupon_rolling_mean) / coupon_rolling_std
        else:
            coupon_zscore = 0

        if coupon_black_scholes_rolling_std != 0:
            coupon_black_scholes_zscore = (self.coupon_black_scholes_return[-1] - coupon_black_scholes_rolling_mean) / coupon_black_scholes_rolling_std
        else:
            coupon_black_scholes_zscore = 0

        coupon_zscore_diff = coupon_zscore - coupon_black_scholes_zscore

        # Option is underpriced
        if coupon_zscore_diff < -1.2:
            coupon_best_ask_vol = sell_orders["COCONUT_COUPON"][best_asks["COCONUT_COUPON"]]

            limit_mult = -coupon_best_ask_vol

            limit_mult = round(limit_mult * abs(coupon_zscore_diff) / 2)

            limit_mult = min(limit_mult, self.POSITION_LIMIT["COCONUT_COUPON"] - position["COCONUT_COUPON"],
                             self.POSITION_LIMIT["COCONUT_COUPON"])

            print("COCONUT_COUPON position:", position["COCONUT_COUPON"])
            print("BUY", "COCONUT_COUPON", str(limit_mult) + "x", best_asks["COCONUT_COUPON"])
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_asks["COCONUT_COUPON"], limit_mult))

        # Option is overpriced
        elif coupon_zscore_diff > 1.2:
            coupon_best_bid_vol = buy_orders["COCONUT_COUPON"][best_bids["COCONUT_COUPON"]]

            limit_mult = coupon_best_bid_vol

            limit_mult = round(-limit_mult * abs(coupon_zscore_diff) / 2)

            limit_mult = max(limit_mult, -self.POSITION_LIMIT["COCONUT_COUPON"] - position["COCONUT_COUPON"],
                             -self.POSITION_LIMIT["COCONUT_COUPON"])

            print("COCONUT_COUPON position:", position["COCONUT_COUPON"])
            print("SELL", "COCONUT_COUPON", str(limit_mult) + "x", best_bids["COCONUT_COUPON"])
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_bids["COCONUT_COUPON"], limit_mult))

        return orders

    def compute_roses_orders(self, state: TradingState):
        orders = []

        roses_pos = state.position["ROSES"] if "ROSES" in state.position else 0
        best_bid = max(state.order_depths["ROSES"].buy_orders.keys())
        bid_vol = state.order_depths["ROSES"].buy_orders[best_bid]
        best_ask = min(state.order_depths["ROSES"].sell_orders.keys())
        ask_vol = state.order_depths["ROSES"].sell_orders[best_ask]

        if "ROSES" not in state.market_trades:
            return orders

        for trade in state.market_trades["ROSES"]:
            if trade.buyer == "Rhianna":
                self.rhianna_buy = True
                self.rhianna_trade = True
            elif trade.seller == "Rhianna":
                self.rhianna_buy = False
                self.rhianna_trade = True

            # Buy signal
            if self.rhianna_buy:
                vol = max(-bid_vol, -self.POSITION_LIMIT["ROSES"] - min(0, roses_pos))
                print("SELL", "ROSES", str(vol) + "x", best_bid)
                orders.append(Order("ROSES", best_bid, vol))
                self.rhianna_buy = False
            # Sell signal
            elif self.rhianna_trade:
                vol = min(-ask_vol, self.POSITION_LIMIT["ROSES"] - max(0, roses_pos))
                print("BUY", "ROSES", str(vol) + "x", best_bid)
                orders.append(Order("ROSES", best_ask, vol))
                self.rhianna_buy = True

        return orders

    def compute_chocolate_orders(self, state: TradingState):
        products = ["CHOCOLATE"]
        position, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "CHOCOLATE": []}

        for product in products:
            position[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.chocolate_returns.append(prices["CHOCOLATE"])

        if len(self.chocolate_returns) < 100:
            return orders
        # slow MA
        chocolate_rolling_mean = statistics.fmean(self.chocolate_returns[-200:])
        # fast MA
        chocolate_rolling_mean_fast = statistics.fmean(self.chocolate_returns[-100:])

        if chocolate_rolling_mean_fast > chocolate_rolling_mean + 1.5:
            limit_mult = 12
            limit_mult = min(limit_mult, self.POSITION_LIMIT["CHOCOLATE"] - position["CHOCOLATE"],
                             self.POSITION_LIMIT["CHOCOLATE"])

            print("CHOCOLATE position:", position["CHOCOLATE"])
            print("BUY", "CHOCOLATE", str(limit_mult) + "x", best_asks["CHOCOLATE"])
            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_asks["CHOCOLATE"], limit_mult))

        elif chocolate_rolling_mean_fast < chocolate_rolling_mean - 1.5:
            limit_mult = -12
            limit_mult = max(limit_mult, -self.POSITION_LIMIT["CHOCOLATE"] - position["CHOCOLATE"],
                             -self.POSITION_LIMIT["CHOCOLATE"])

            print("CHOCOLATE position:", position["CHOCOLATE"])
            print("SELL", "CHOCOLATE", str(limit_mult) + "x", best_bids["CHOCOLATE"])
            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_bids["CHOCOLATE"], limit_mult))

        return orders

    def compute_strawberries_orders(self, state: TradingState):
        products = ["STRAWBERRIES"]
        position, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "STRAWBERRIES": []}

        for product in products:
            position[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.strawberries_returns.append(prices["STRAWBERRIES"])

        if len(self.strawberries_returns) < 100:
            return orders

        strawberries_rolling_mean = statistics.fmean(self.strawberries_returns[-200:])
        strawberries_rolling_mean_fast = statistics.fmean(self.strawberries_returns[-100:])

        if strawberries_rolling_mean_fast > strawberries_rolling_mean + 1.5:
            limit_mult = 18
            limit_mult = min(limit_mult, self.POSITION_LIMIT["STRAWBERRIES"] - position["STRAWBERRIES"],
                             self.POSITION_LIMIT["STRAWBERRIES"])

            print("STRAWBERRIES position:", position["STRAWBERRIES"])
            print("BUY", "STRAWBERRIES", str(limit_mult) + "x", best_asks["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_asks["STRAWBERRIES"], limit_mult))

        elif strawberries_rolling_mean_fast < strawberries_rolling_mean - 1.5:

            limit_mult = -18
            limit_mult = max(limit_mult, -self.POSITION_LIMIT["STRAWBERRIES"] - position["STRAWBERRIES"],
                             -self.POSITION_LIMIT["STRAWBERRIES"])

            print("STRAWBERRIES position:", position["STRAWBERRIES"])
            print("SELL", "STRAWBERRIES", str(limit_mult) + "x", best_bids["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_bids["STRAWBERRIES"], limit_mult))

        return orders

    def compute_coconut_orders(self, state: TradingState):
        products = ["COCONUT"]
        position, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "COCONUT": []}

        for product in products:
            position[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.coconut_returns.append(prices["COCONUT"])

        if len(self.coconut_returns) < 100:
            return orders

        if int(round(self.person_position['Vladimir']['COCONUT'])) > 0:
            self.buy_berries = True
            self.sell_berries = False
        if int(round(self.person_position['Vladimir']['COCONUT'])) < 0:
            self.sell_berries = True
            self.buy_berries = False
            
        coconut_rolling_mean = statistics.fmean(self.coconut_returns[-200:])
        coconut_rolling_mean_fast = statistics.fmean(self.coconut_returns[-100:])

        if coconut_rolling_mean_fast > coconut_rolling_mean + 4:
            limit_mult = 30
            limit_mult = min(limit_mult, self.POSITION_LIMIT["COCONUT"] - position["COCONUT"],
                             self.POSITION_LIMIT["COCONUT"])

            print("COCONUT position:", position["COCONUT"])
            print("BUY", "COCONUT", str(limit_mult) + "x", best_asks["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_asks["COCONUT"], limit_mult))

        elif coconut_rolling_mean_fast < coconut_rolling_mean - 4:
            limit_mult = -30
            limit_mult = max(limit_mult, -self.POSITION_LIMIT["COCONUT"] - position["COCONUT"],
                             -self.POSITION_LIMIT["COCONUT"])

            print("COCONUT position:", position["COCONUT"])
            print("SELL", "COCONUT", str(limit_mult) + "x", best_bids["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_bids["COCONUT"], limit_mult))

        return orders

    def packageData(self) -> str: 
        return json.dumps({"starfruit_history": self.starfruit_history, "starfruit_spread_cache": self.starfruit_spread_cache, "orchid_cache": self.orchid_cache, "orchid_spread_cache": [], "sunlight_cache": self.sunlight_cache, "humidity_cache": self.humidity_cache, "etf_returns": self.etf_returns, "assets_returns": self.assets_returns, "strawberries_returns": self.strawberries_returns, "strawberries_estimated_returns": self.strawberries_estimated_returns, "chocolate_returns": self.chocolate_returns, "chocolate_estimated_returns": self.chocolate_estimated_returns, "roses_returns": self.roses_returns, "roses_estimated_returns": self.roses_estimated_returns, "coupon_return": self.coupon_return, "coupon_black_scholes_return": self.coupon_black_scholes_return, "coconut_returns": self.coconut_returns, "coconut_predicted_return": self.coconut_predicted_return, "rhianna_buy": self.rhianna_buy, "rhianna_trade": self.rhianna_trade})

    def unpackData(self, state: TradingState): 
        if not state.traderData:
            state.traderData = json.dumps({"starfruit_history": [], "starfruit_spread_cache": [], "orchid_cache": [], "orchid_spread_cache": [], "sunlight_cache": [], "humidity_cache": [], "etf_returns": [], "assets_returns": [], "strawberries_returns": [], "strawberries_estimated_returns": [], "chocolate_returns": [], "chocolate_estimated_returns": [], "roses_returns": [], "roses_estimated_returns": [], "coupon_return": [], "coupon_black_scholes_return": [], "coconut_returns": [], "coconut_predicted_return": [], "rhianna_buy": False, "rhianna_trade": False})
        
        traderDataDict = json.loads(state.traderData)
        self.starfruit_history = traderDataDict["starfruit_history"]
        self.starfruit_spread_cache = traderDataDict["starfruit_spread_cache"]
        
        self.orchid_cache = traderDataDict["orchid_cache"]
        self.orchid_spread_cache = traderDataDict["orchid_spread_cache"]
        self.sunlight_cache = traderDataDict["sunlight_cache"]
        self.humidity_cache = traderDataDict["humidity_cache"]

        self.etf_returns = traderDataDict["etf_returns"]
        self.assets_returns = traderDataDict["assets_returns"]
        self.chocolate_returns = traderDataDict["chocolate_returns"]
        self.chocolate_estimated_returns = traderDataDict["chocolate_estimated_returns"]
        self.strawberries_returns = traderDataDict["strawberries_returns"]
        self.strawberries_estimated_returns = traderDataDict["strawberries_estimated_returns"]
        self.roses_returns = traderDataDict["roses_returns"]
        self.roses_estimated_returns = traderDataDict["roses_estimated_returns"]

        self.coconut_returns = traderDataDict["coconut_returns"]
        self.coconut_predicted_return = traderDataDict["coconut_predicted_return"]
        self.coupon_return = traderDataDict["coupon_return"]
        self.coupon_black_scholes_return = traderDataDict["coupon_black_scholes_return"]

        self.rhianna_buy = traderDataDict["rhianna_buy"]
        self.rhianna_trade = traderDataDict["rhianna_trade"]

    def run(self, state: TradingState):
        for product, position in state.position.items():
            self.position[product] = position

        self.unpackData(state)

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            market_price = self.vwap(order_depth)

            conversions = 0

            if product == "AMETHYSTS":
                cpos = self.position[product]
                for ask, vol in order_depth.sell_orders.items():
                    if ((ask < self.amethysts_default_price) or ((self.position[product] < 0) and (ask == self.amethysts_default_price))) and cpos < self.POSITION_LIMIT[product]:
                        order_vol = min(-vol, self.POSITION_LIMIT[product] - cpos)
                        cpos += order_vol
                        orders.append(Order(product, ask, order_vol))

                undercut_sell = best_ask - 1
                undercut_buy = best_bid + 1

                acc_ask = max(undercut_sell, self.amethysts_default_price + 1)
                acc_bid = min(undercut_buy, self.amethysts_default_price - 1)

                if cpos < self.POSITION_LIMIT[product]:
                    if self.position[product] < 0:
                        ask = min(undercut_buy + 1, self.amethysts_default_price - 1)
                    elif self.position[product] > 15:
                        ask = min(undercut_buy - 1, self.amethysts_default_price - 1)
                    else:
                        ask = acc_bid

                    order_vol = min(self.POSITION_LIMIT[product], self.POSITION_LIMIT[product] - cpos)
                    cpos += order_vol
                    orders.append(Order(product, ask, order_vol))

                cpos = self.position[product]

                for bid, vol in order_depth.buy_orders.items():
                    if ((bid > self.amethysts_default_price) or ((self.position[product] > 0) and (bid == self.amethysts_default_price))) and cpos > -self.POSITION_LIMIT[product]:
                        order_vol = max(-vol, -self.POSITION_LIMIT[product] - cpos)
                        cpos += order_vol
                        orders.append(Order(product, bid, order_vol))

                if cpos > -self.POSITION_LIMIT[product]:
                    if self.position[product] < 0:
                        bid = max(undercut_sell - 1, self.amethysts_default_price + 1)
                    elif self.position[product] < -15:
                        bid = max(undercut_sell + 1, self.amethysts_default_price + 1)
                    else:
                        bid = acc_ask

                    order_vol = max(-self.POSITION_LIMIT[product], -self.POSITION_LIMIT[product] - cpos)
                    cpos += order_vol
                    orders.append(Order(product, bid, order_vol))

            elif product == "STARFRUIT":
                if len(self.starfruit_history) == self.starfruit_cache_size:
                    self.starfruit_history.pop(0)
                if len(self.starfruit_spread_cache) == self.starfruit_cache_size:
                    self.starfruit_spread_cache.pop(0)
                self.starfruit_spread_cache.append(best_ask - best_bid)
                self.starfruit_history.append(market_price)

                if len(self.starfruit_history) == self.starfruit_cache_size:
                    starfruit_predicted_price = self.predict_starfruit_price(self.starfruit_history)
                    spread = round(statistics.fmean(self.starfruit_spread_cache[-31:]))

                    lower_bound = starfruit_predicted_price - (spread // 2)
                    upper_bound = starfruit_predicted_price + (spread // 2)
                else:
                    lower_bound = -int(1e9)
                    upper_bound = int(1e9)

                cpos = self.position[product]
                for ask, vol in order_depth.sell_orders.items():
                    if ((ask <= lower_bound) or ((self.position[product] < 0) and (ask <= lower_bound + (spread // 2)))) and cpos < self.POSITION_LIMIT[product]:
                        order_vol = min(-vol, self.POSITION_LIMIT[product] - cpos)
                        cpos += order_vol
                        orders.append(Order(product, ask, order_vol))

                undercut_sell = best_ask - 1
                undercut_buy = best_bid + 1

                acc_ask = max(undercut_sell, upper_bound)
                acc_bid = min(undercut_buy, lower_bound)

                if cpos < self.POSITION_LIMIT[product]:
                    order_vol = self.POSITION_LIMIT[product] - cpos
                    cpos += order_vol
                    orders.append(Order(product, acc_bid, order_vol))

                cpos = self.position[product]

                for bid, vol in order_depth.buy_orders.items():
                    if ((bid >= upper_bound) or ((self.position[product] > 0) and (bid >= upper_bound - (spread // 2)))) and cpos > -self.POSITION_LIMIT[product]:
                        order_vol = max(-vol, -self.POSITION_LIMIT[product] - cpos)
                        cpos += order_vol
                        orders.append(Order(product, bid, order_vol))

                if cpos > -self.POSITION_LIMIT[product]:
                    order_vol = max(-self.POSITION_LIMIT[product], -self.POSITION_LIMIT[product] - cpos)
                    cpos += order_vol
                    orders.append(Order(product, acc_ask, order_vol))

            result[product] = orders

        basket_orders = self.compute_basket_orders(state)

        for product, orders in basket_orders.items():
            result[product] = orders

        coupon_orders = self.compute_coupon_orders(state)

        for product, orders in coupon_orders.items():
            result[product] = orders

        result["ROSES"] = self.compute_roses_orders(state)

        chocolate_orders = self.compute_chocolate_orders(state)

        for product, orders in chocolate_orders.items():
            result[product] = orders

        strawberries_orders = self.compute_strawberries_orders(state)

        for product, orders in strawberries_orders.items():
            result[product] = orders

        coconut_orders = self.compute_coconut_orders(state)

        for product, orders in coconut_orders.items():
            result[product] = orders

        traderData = self.packageData()

        return result, conversions, traderData