import collections
from datamodel import OrderDepth, TradingState, Order, Observation
from typing import List, Dict
import numpy as np
import math
import pandas as pd
import json
import jsonpickle
import statistics

class MarketData:
    def __init__(self):
        self.amethysts_history = []
        self.starfruit_history = []

class Trader:
    def __init__(self):
        self.INF = int(1e9)
        self.starfruit_cache_size = 35
        self.amethysts_cache_size = 5 # 10
        self.position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0}
        self.POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
        self.amethysts_spread = 1
        self.amethysts_default_price = 10_000
        self.starfruit_spread = 1
        self.orchids_spread = 5
        self.std = 33.389996883675984
        self.mean = 411.9
        self.threshold = 0.6


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
            predicted_price = int(round(historical_prices[-1] * 1.02))
        elif macd_value < 0 or volatility > np.mean(historical_prices) * 0.05:
            predicted_price = int(round(historical_prices[-1] * 0.98))
        else:
            predicted_price = int(round(historical_prices[-1]))

        return predicted_price

    def predict_starfruit_price(self, cache, smoothing_level=0.2):
        # simple exponential smoothing
        data = pd.Series(cache)
        predicted_price = data.ewm(alpha=smoothing_level, adjust=False).mean().iloc[-1]
        return int(round(predicted_price))
    
    def vol_and_best_price(self, orders, buy_order):
        volume = 0
        best = 0 if buy_order else self.INF

        for price, vol in orders.items():
            if buy_order:
                volume += vol
                best = max(best, price)
            else:
                volume -= vol
                best = min(best, price)

        return volume, best


    def compute_orders(self, product, order_depth, acc_bid, acc_ask, orchids = False):
        orders: list[Order] = []
        
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.vol_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.vol_and_best_price(buy_orders, buy_order=True)

        position = self.position[product] if not orchids else 0
        limit = self.POSITION_LIMIT[product]

        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, acc_bid)
        ask_price = max(penny_sell, acc_ask)

        if orchids:
            ask_price = max(best_sell_price - self.orchids_spread, acc_ask)

        for ask, vol in sell_orders.items():
            if position < limit and (ask <= acc_bid or (position < 0 and ask == acc_bid+1)): 
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        position = self.position[product] if not orchids else 0 

        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= acc_ask or (position > 0 and bid+1 == acc_ask)):
                num_orders = max(-vol, -limit-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders 

        return orders

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        conversions = 0
 
        for product in state.order_depths:
            self.position[product] = state.position[product] if product in state.position else 0

        if state.traderData == '':
            data = MarketData()
        else:
            data = jsonpickle.decode(state.traderData)
        
        for product in state.order_depths:
            position = state.position[product] if product in state.position else 0
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product == "AMETHYSTS":
                if len(data.amethysts_history) == self.amethysts_cache_size:
                    data.amethysts_history.pop(0)

                sell_vol, best_sell_price = self.vol_and_best_price(order_depth.sell_orders, buy_order=False)
                buy_vol, best_buy_price = self.vol_and_best_price(order_depth.buy_orders, buy_order=True)

                data.amethysts_history.append((best_sell_price+best_buy_price)/2)

                lower = -self.INF
                upper = self.INF

                if len(data.amethysts_history) == self.amethysts_cache_size:
                    lower = self.predict_amethysts_price(data.amethysts_history)-self.amethysts_spread
                    upper = self.predict_amethysts_price(data.amethysts_history)+self.amethysts_spread

                orders += self.compute_orders(product, order_depth, lower, upper)
                
            elif product == "STARFRUIT":  
                if len(data.starfruit_history) == self.starfruit_cache_size:
                    data.starfruit_history.pop(0)

                sell_vol, best_sell_price = self.vol_and_best_price(order_depth.sell_orders, buy_order=False)
                buy_vol, best_buy_price = self.vol_and_best_price(order_depth.buy_orders, buy_order=True)

                data.starfruit_history.append((best_sell_price+best_buy_price)/2)

                lower = -self.INF
                upper = self.INF

                if len(data.starfruit_history) == self.starfruit_cache_size:
                    lower = self.predict_starfruit_price(data.starfruit_history)-self.starfruit_spread
                    upper = self.predict_starfruit_price(data.starfruit_history)+self.starfruit_spread

                orders += self.compute_orders(product, order_depth, lower, upper)

            elif product == "ORCHIDS":
                shipment = state.observations.conversionObservations["ORCHIDS"].transportFees
                ask = state.observations.conversionObservations["ORCHIDS"].askPrice
                bid = state.observations.conversionObservations["ORCHIDS"].bidPrice
                import_duty = state.observations.conversionObservations["ORCHIDS"].importTariff
                export_duty = state.observations.conversionObservations["ORCHIDS"].exportTariff

                buy_price = ask + shipment + import_duty
                sell_price = bid + shipment + export_duty

                # # Collect price history
                # data.orchids_history.append((buy_price + sell_price) / 2)
                # prices = data.orchids_history
                
                # # Calculate volatility and RSI
                # if len(prices) > 30:
                #     log_returns = np.log(prices / np.roll(prices, 1))[1:]
                #     volatility = np.std(log_returns)
                #     rsi = self.get_rsi(np.array(prices))

                #     # Adjust bid and ask spread based on volatility and RSI
                #     if rsi > 70:  # Overbought
                #         ask_price = sell_price - 2  # More aggressive selling
                #         bid_price = buy_price + 2  # Less aggressive buying
                #     elif rsi < 30:  # Oversold
                #         ask_price = sell_price + 2  # Less aggressive selling
                #         bid_price = buy_price - 2  # More aggressive buying
                #     else:
                #         ask_price = sell_price + 1  # Normal trading conditions
                #         bid_price = buy_price - 1
                # else:
                #     ask_price = sell_price
                #     bid_price = buy_price


                lower = int(round(buy_price)) - 1
                upper = int(round(buy_price)) + 1

                orders += self.compute_orders(product, order_depth, lower, upper, orchids = True)
                conversions = -self.position[product]

            elif product == "GIFT_BASKET":
                _, chocolate_best_sell_price = self.vol_and_best_price(state.order_depths['CHOCOLATE'].sell_orders, buy_order=False)
                _, chocolate_best_buy_price = self.vol_and_best_price(state.order_depths['CHOCOLATE'].buy_orders, buy_order=True)
                _, strawberries_best_sell_price = self.vol_and_best_price(state.order_depths['STRAWBERRIES'].sell_orders, buy_order=False)
                _, strawberries_best_buy_price = self.vol_and_best_price(state.order_depths['STRAWBERRIES'].buy_orders, buy_order=True)
                _, roses_best_sell_price = self.vol_and_best_price(state.order_depths['ROSES'].sell_orders, buy_order=False)
                _, roses_best_buy_price = self.vol_and_best_price(state.order_depths['ROSES'].buy_orders, buy_order=True)

                basket_items = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
                mid_price = {}

                for item in basket_items:
                    sell_vol, best_sell_price = self.vol_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                    buy_vol, best_buy_price = self.vol_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                    mid_price[item] = (best_sell_price+best_buy_price)/2

                difference = mid_price['GIFT_BASKET'] - 4*mid_price['CHOCOLATE'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'] - self.mean

                worst_bid = min(order_depth.buy_orders.keys())
                worst_ask = max(order_depth.sell_orders.keys())

                if difference > self.threshold * self.std:
                    orders += self.compute_orders(product, order_depth, -self.INF, worst_bid)
                
                elif difference < -self.threshold * self.std:
                    orders += self.compute_orders(product, order_depth, worst_ask, self.INF)

            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData