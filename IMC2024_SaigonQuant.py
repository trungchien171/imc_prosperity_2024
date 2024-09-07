import collections
from collections import defaultdict
from datamodel import OrderDepth, TradingState, Order, ConversionObservation, Observation
from typing import List, Dict
import numpy as np
import pandas as pd
import json
import statistics

class Trader:   
    def __init__(self):
        self.INF = int(1e9)
        self.amethysts_history = []
        self.starfruit_history = []
        self.starfruit_cache_size = 35
        self.amethysts_cache_size = 5
        self.starfruit_spread_cache = []
        self.position = defaultdict(int)
        self.POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 
                               'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.amethysts_default_price = 10_000
        self.rhianna_buy = False
        self.rhianna_trade = False

        self.rolling_cache = {
            "etf_returns": [], "assets_returns": [], "chocolate_returns": [], "strawberries_returns": [],
            "roses_returns": [], "coupon_return": [], "coupon_black_scholes_return": [], "coconut_returns": []
        }
        
        self.N = statistics.NormalDist(mu=0, sigma=1)

    def calculate_macd(self, prices):
        exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean() 
        exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()  
        return macd.iloc[-1] - signal.iloc[-1]
    
    def calculate_volatility(self, prices):
        log_returns = np.log(prices / np.roll(prices, 1))[1:]
        return np.std(log_returns)
    
    def predict_amethysts_price(self, historical_prices):
        if len(historical_prices) < 30:
            return self.amethysts_default_price

        macd_value = self.calculate_macd(historical_prices)
        volatility = self.calculate_volatility(historical_prices)
        
        if macd_value > 0 and volatility < np.mean(historical_prices) * 0.05:
            return int(round(historical_prices[-1] * 1.02))
        elif macd_value < 0 or volatility > np.mean(historical_prices) * 0.05:
            return int(round(historical_prices[-1] * 0.98))
        else:
            return int(round(historical_prices[-1]))

    def predict_starfruit_price(self, cache, smoothing_level=0.2):
        data = pd.Series(cache)
        predicted_price = data.ewm(alpha=smoothing_level, adjust=False).mean().iloc[-1]
        return int(round(predicted_price))

    def black_scholes(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.N.cdf(d1) - K * np.exp(-r * T) * self.N.cdf(d2)

    def vwap(self, order_depth):
        total_ask, total_bid, ask_vol, bid_vol = 0, 0, 0, 0

        for ask, vol in order_depth.sell_orders.items():
            total_ask += ask * abs(vol)
            ask_vol += abs(vol)

        for bid, vol in order_depth.buy_orders.items():
            total_bid += bid * vol
            bid_vol += vol

        ask_price = total_ask / ask_vol if ask_vol else 0
        bid_price = total_bid / bid_vol if bid_vol else 0
        return (ask_price + bid_price) / 2 if ask_price and bid_price else 0

    def compute_orders(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []

            if product == "AMETHYSTS":
                orders = self.process_amethysts_orders(order_depth)

            elif product == "STARFRUIT":
                orders = self.process_starfruit_orders(order_depth)

            result[product] = orders

        basket_orders = self.compute_basket_orders(state)
        coupon_orders = self.compute_coupon_orders(state)
        roses_orders = self.compute_roses_orders(state)
        chocolate_orders = self.compute_chocolate_orders(state)
        strawberries_orders = self.compute_strawberries_orders(state)
        coconut_orders = self.compute_coconut_orders(state)

        for order_type in [basket_orders, coupon_orders, roses_orders, chocolate_orders, strawberries_orders, coconut_orders]:
            result.update(order_type)

        state.traderData = self.packageData()
        return result

    def process_amethysts_orders(self, order_depth):
        orders = []
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        position = self.position["AMETHYSTS"]

        if position < self.POSITION_LIMIT["AMETHYSTS"]:
            for ask, vol in order_depth.sell_orders.items():
                if ask <= self.amethysts_default_price:
                    vol_to_buy = min(-vol, self.POSITION_LIMIT["AMETHYSTS"] - position)
                    orders.append(Order("AMETHYSTS", ask, vol_to_buy))
                    position += vol_to_buy

        if position > -self.POSITION_LIMIT["AMETHYSTS"]:
            for bid, vol in order_depth.buy_orders.items():
                if bid >= self.amethysts_default_price:
                    vol_to_sell = max(-vol, -self.POSITION_LIMIT["AMETHYSTS"] - position)
                    orders.append(Order("AMETHYSTS", bid, vol_to_sell))
                    position += vol_to_sell

        return orders

    def process_starfruit_orders(self, order_depth):
        orders = []
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        market_price = self.vwap(order_depth)
        position = self.position["STARFRUIT"]

        self.update_cache(self.starfruit_history, market_price, self.starfruit_cache_size)
        self.update_cache(self.starfruit_spread_cache, best_ask - best_bid, self.starfruit_cache_size)

        if len(self.starfruit_history) == self.starfruit_cache_size:
            predicted_price = self.predict_starfruit_price(self.starfruit_history)
            spread = statistics.fmean(self.starfruit_spread_cache)
            lower_bound = predicted_price - (spread // 2)
            upper_bound = predicted_price + (spread // 2)
        else:
            lower_bound, upper_bound = -self.INF, self.INF

        if position < self.POSITION_LIMIT["STARFRUIT"]:
            for ask, vol in order_depth.sell_orders.items():
                if ask <= lower_bound:
                    vol_to_buy = min(-vol, self.POSITION_LIMIT["STARFRUIT"] - position)
                    orders.append(Order("STARFRUIT", ask, vol_to_buy))
                    position += vol_to_buy

        if position > -self.POSITION_LIMIT["STARFRUIT"]:
            for bid, vol in order_depth.buy_orders.items():
                if bid >= upper_bound:
                    vol_to_sell = max(-vol, -self.POSITION_LIMIT["STARFRUIT"] - position)
                    orders.append(Order("STARFRUIT", bid, vol_to_sell))
                    position += vol_to_sell

        return orders

    def update_cache(self, cache, new_value, cache_size):
        if len(cache) == cache_size:
            cache.pop(0)
        cache.append(new_value)

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

        self.rolling_cache["etf_returns"].append(prices["GIFT_BASKET"])
        self.rolling_cache["assets_returns"].append(price_diff)

        if len(self.rolling_cache["etf_returns"]) < 100 or len(self.rolling_cache["assets_returns"]) < 100:
            return orders
        
        # Slow MA and Fast MA
        assets_rolling_mean = statistics.fmean(self.rolling_cache["assets_returns"][-200:])
        assets_rolling_mean_fast = statistics.fmean(self.rolling_cache["assets_returns"][-100:])

        # Buy/Sell signals
        if assets_rolling_mean_fast > assets_rolling_mean + 4:
            limit_mult = min(3, self.POSITION_LIMIT["GIFT_BASKET"] - position["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_asks["GIFT_BASKET"], limit_mult))

        elif assets_rolling_mean_fast < assets_rolling_mean - 4:
            limit_mult = max(-3, -self.POSITION_LIMIT["GIFT_BASKET"] - position["GIFT_BASKET"])
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

        self.rolling_cache["coupon_return"].append(prices["COCONUT_COUPON"])
        self.rolling_cache["coupon_black_scholes_return"].append(pred_price)

        if len(self.rolling_cache["coupon_return"]) < 2 or len(self.rolling_cache["coupon_black_scholes_return"]) < 2:
            return orders

        coupon_zscore = self.calculate_zscore(self.rolling_cache["coupon_return"], prices["COCONUT_COUPON"])
        coupon_black_scholes_zscore = self.calculate_zscore(self.rolling_cache["coupon_black_scholes_return"], pred_price)
        coupon_zscore_diff = coupon_zscore - coupon_black_scholes_zscore

        if coupon_zscore_diff < -1.2:
            vol = sell_orders["COCONUT_COUPON"][best_asks["COCONUT_COUPON"]]
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_asks["COCONUT_COUPON"], -vol))
        elif coupon_zscore_diff > 1.2:
            vol = buy_orders["COCONUT_COUPON"][best_bids["COCONUT_COUPON"]]
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_bids["COCONUT_COUPON"], vol))

        return orders

    def calculate_zscore(self, data, current_value):
        mean_val = statistics.fmean(data[-200:])
        std_val = statistics.stdev(data[-200:])
        if std_val != 0:
            return (current_value - mean_val) / std_val
        return 0

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

            if self.rhianna_buy:
                vol = max(-bid_vol, -self.POSITION_LIMIT["ROSES"] - min(0, roses_pos))
                orders.append(Order("ROSES", best_bid, vol))
                self.rhianna_buy = False
            elif self.rhianna_trade:
                vol = min(-ask_vol, self.POSITION_LIMIT["ROSES"] - max(0, roses_pos))
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

        self.rolling_cache["chocolate_returns"].append(prices["CHOCOLATE"])

        if len(self.rolling_cache["chocolate_returns"]) < 100:
            return orders
        
        chocolate_rolling_mean = statistics.fmean(self.rolling_cache["chocolate_returns"][-200:])
        chocolate_rolling_mean_fast = statistics.fmean(self.rolling_cache["chocolate_returns"][-100:])

        if chocolate_rolling_mean_fast > chocolate_rolling_mean + 1.5:
            limit_mult = min(12, self.POSITION_LIMIT["CHOCOLATE"] - position["CHOCOLATE"])
            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_asks["CHOCOLATE"], limit_mult))
        elif chocolate_rolling_mean_fast < chocolate_rolling_mean - 1.5:
            limit_mult = max(-12, -self.POSITION_LIMIT["CHOCOLATE"] - position["CHOCOLATE"])
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

        self.rolling_cache["strawberries_returns"].append(prices["STRAWBERRIES"])

        if len(self.rolling_cache["strawberries_returns"]) < 100:
            return orders

        strawberries_rolling_mean = statistics.fmean(self.rolling_cache["strawberries_returns"][-200:])
        strawberries_rolling_mean_fast = statistics.fmean(self.rolling_cache["strawberries_returns"][-100:])

        if strawberries_rolling_mean_fast > strawberries_rolling_mean + 1.5:
            limit_mult = min(18, self.POSITION_LIMIT["STRAWBERRIES"] - position["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_asks["STRAWBERRIES"], limit_mult))

        elif strawberries_rolling_mean_fast < strawberries_rolling_mean - 1.5:
            limit_mult = max(-18, -self.POSITION_LIMIT["STRAWBERRIES"] - position["STRAWBERRIES"])
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

        self.rolling_cache["coconut_returns"].append(prices["COCONUT"])

        if len(self.rolling_cache["coconut_returns"]) < 100:
            return orders

        coconut_rolling_mean = statistics.fmean(self.rolling_cache["coconut_returns"][-200:])
        coconut_rolling_mean_fast = statistics.fmean(self.rolling_cache["coconut_returns"][-100:])

        if coconut_rolling_mean_fast > coconut_rolling_mean + 4:
            limit_mult = min(30, self.POSITION_LIMIT["COCONUT"] - position["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_asks["COCONUT"], limit_mult))
        elif coconut_rolling_mean_fast < coconut_rolling_mean - 4:
            limit_mult = max(-30, -self.POSITION_LIMIT["COCONUT"] - position["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_bids["COCONUT"], limit_mult))

        return orders

    def packageData(self):
        return json.dumps({
            "starfruit_history": self.starfruit_history, "starfruit_spread_cache": self.starfruit_spread_cache, 
            "rolling_cache": self.rolling_cache, "rhianna_buy": self.rhianna_buy, "rhianna_trade": self.rhianna_trade
        })

    def unpackData(self, state):
        if not state.traderData:
            return
        data = json.loads(state.traderData)
        self.starfruit_history = data.get("starfruit_history", [])
        self.starfruit_spread_cache = data.get("starfruit_spread_cache", [])
        self.rolling_cache = data.get("rolling_cache", self.rolling_cache)
        self.rhianna_buy = data.get("rhianna_buy", False)
        self.rhianna_trade = data.get("rhianna_trade", False)