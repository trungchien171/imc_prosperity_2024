import collections
from datamodel import OrderDepth, TradingState, Order, Observation
from typing import List
import numpy as np
import math
import pandas as pd
import json
import jsonpickle

class MarketData:
    def __init__(self):
        self.amethysts_history = []
        self.starfruit_history = []

class Trader:
    INF = int(1e9)
    starfruit_cache_size = 20
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0}
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100}
    amethysts_spread = 1
    amethysts_default_price = 10_000
    starfruit_spread = 1
    orchids_spread = 5

    def predict_starfruit_price(self, cache):
        x = np.array([i for i in range(self.starfruit_cache_size)])
        y = np.array(cache)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        predicted_price = int(round(self.starfruit_cache_size*m + c))
        return predicted_price
    
    # def predict_starfruit_price(self, cache, alpha=1.0, degree = 2):
    #     # Generate polynomial features
    #     x = np.array([i for i in range(len(cache))])
    #     X = np.vstack([x**d for d in range(degree + 1)]).T
    #     # Calculate weights for Ridge Regression 
    #     weights = np.linspace(1,2, len(cache))
    #     W = np.diag(weights)
    #     y = np.array(cache)
    #     # Ridge Regression formula: (X'X + alpha*I)^-1 X'y
    #     XTWX = X.T @ W @ X
    #     Ridge = XTWX + alpha * np.eye(XTWX.shape[0])
    #     XTWy = X.T @ W @ y
    #     beta = np.linalg.inv(Ridge).dot(XTWy)
    #     # Predicting the next value
    #     new_x = np.array([len(cache)**d for d in range(degree + 1)])
    #     predicted_price = new_x.dot(beta)
    #     return int(round(predicted_price))

    # def predict_starfruit_price(self, cache):
    #     if not cache:
    #         return 0  # No data to predict from

    #     # Convert cache to numpy array for easier manipulation
    #     prices = np.array(cache)

    #     # Basic statistics
    #     mean_price = np.mean(prices)
    #     std_price = np.std(prices)
    #     recent_price = prices[-1]  # Last known price
        
    #     # Simple decision rules
    #     if recent_price > mean_price + std_price:
    #         # If the recent price is significantly higher than the average, predict decrease
    #         return int(round(recent_price - std_price))
    #     elif recent_price < mean_price - std_price:
    #         # If the recent price is significantly lower than the average, predict increase
    #         return int(round(recent_price + std_price))
    #     else:
    #         # If prices are stable, predict the last price adjusted slightly by recent trend
    #         trend = np.polyfit(range(len(prices)), prices, 1)[0]  # Linear trend
    #         return int(round(recent_price + trend))
        
    def get_volume_and_best_price(self, orders, buy_order):
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

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

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

            if product == 'AMETHYSTS':
                orders += self.compute_orders(product, order_depth, self.amethysts_default_price - self.amethysts_spread, self.amethysts_default_price + self.amethysts_spread)
                
            elif product == "STARFRUIT":  
                if len(data.starfruit_history) == self.starfruit_cache_size:
                    data.starfruit_history.pop(0)

                _, best_sell_price = self.get_volume_and_best_price(order_depth.sell_orders, buy_order=False)
                _, best_buy_price = self.get_volume_and_best_price(order_depth.buy_orders, buy_order=True)

                data.starfruit_history.append((best_sell_price+best_buy_price)/2)

                lower_bound = -self.INF
                upper_bound = self.INF

                if len(data.starfruit_history) == self.starfruit_cache_size:
                    lower_bound = self.predict_starfruit_price(data.starfruit_history)-self.starfruit_spread
                    upper_bound = self.predict_starfruit_price(data.starfruit_history)+self.starfruit_spread

                orders += self.compute_orders(product, order_depth, lower_bound, upper_bound)

            elif product == "ORCHIDS":
                shipment = state.observations.conversionObservations["ORCHIDS"].transportFees
                import_duty = state.observations.conversionObservations["ORCHIDS"].importTariff
                export_duty = state.observations.conversionObservations["ORCHIDS"].exportTariff
                ask = state.observations.conversionObservations["ORCHIDS"].askPrice
                bid = state.observations.conversionObservations["ORCHIDS"].bidPrice

                buy_price = ask + shipment + import_duty
                sell_price = bid + shipment + export_duty

                lower_bound = int(round(buy_price)) - 1
                upper_bound = int(round(buy_price)) + 1

                orders += self.compute_orders(product, order_depth, lower_bound, upper_bound, orchids = True)
                conversions = -self.position[product]
                
            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData