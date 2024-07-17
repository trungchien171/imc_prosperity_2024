import collections
from datamodel import OrderDepth, TradingState, Order
from typing import List
import numpy as np
import math
import pandas as pd
import json
import jsonpickle

class Data:
    def __init__(self):
        self.starfruit_cache = []

class Trader:
    INF = int(1e9)
    STARFRUIT_CACHE_SIZE = 20
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}

    def predict_starfruit_price(self, cache):
        x = np.array([i for i in range(self.STARFRUIT_CACHE_SIZE)])
        y = np.array(cache)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return int(round(self.STARFRUIT_CACHE_SIZE*m + c))
    
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
    
    def compute_starfruit_orders(self, product, order_depth, our_bid, our_ask):
        orders: list[Order] = []
        
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        position = self.position[product]
        limit = self.POSITION_LIMIT[product]

        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)

        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        position = self.position[product] # RESET position

        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
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

        # update our position 
        for product in state.order_depths:
            self.position[product] = state.position[product] if product in state.position else 0

        if state.traderData == '': # first run, set up data
            data = Data()
        else:
            data = jsonpickle.decode(state.traderData)
        
        global price_history_starfruit
        for product in state.order_depths:
            position = state.position[product] if product in state.position else 0
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == 'AMETHYSTS':
                spread = 1
                open_spread = 3
                start_trading = 0
                position_limit = 20
                position_spread = 15
                current_position = state.position.get(product,0)
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                    
                if state.timestamp >= start_trading:
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        
                        if best_ask <= 10000-spread:
                            best_ask_volume = order_depth.sell_orders[best_ask]
                        else:
                            best_ask_volume = 0
                    else:
                        best_ask_volume = 0
                         
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                    
                        if best_bid >= 10000+spread:
                            best_bid_volume = order_depth.buy_orders[best_bid]
                        else:
                            best_bid_volume = 0 
                    else:
                        best_bid_volume = 0
                    
                    if current_position - best_ask_volume > position_limit:
                        best_ask_volume = current_position - position_limit
                        open_ask_volume = 0
                    else:
                        open_ask_volume = current_position - position_spread - best_ask_volume
                        
                    if current_position - best_bid_volume < -position_limit:
                        best_bid_volume = current_position + position_limit
                        open_bid_volume = 0
                    else:
                        open_bid_volume = current_position + position_spread - best_bid_volume
                        
                    if -open_ask_volume < 0:
                        open_ask_volume = 0         
                    if open_bid_volume < 0:
                        open_bid_volume = 0

                    if -best_ask_volume > 0:
                        orders.append(Order(product, best_ask, -best_ask_volume))
                    if -open_ask_volume > 0:
                        orders.append(Order(product, 10000 - open_spread, -open_ask_volume))

                    if best_bid_volume > 0:
                        orders.append(Order(product, best_bid, -best_bid_volume))
                    if open_bid_volume > 0:
                        orders.append(Order(product, 10000 + open_spread, -open_bid_volume))
                        
                result[product] = orders
                

            if product == "STARFRUIT":  
                if len(data.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                    data.starfruit_cache.pop(0)

                _, best_sell_price = self.get_volume_and_best_price(order_depth.sell_orders, buy_order=False)
                _, best_buy_price = self.get_volume_and_best_price(order_depth.buy_orders, buy_order=True)

                data.starfruit_cache.append((best_sell_price+best_buy_price)/2)

                lower_bound = -self.INF
                upper_bound = self.INF

                if len(data.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                    lower_bound = self.predict_starfruit_price(data.starfruit_cache)-2
                    upper_bound = self.predict_starfruit_price(data.starfruit_cache)+2


                orders += self.compute_starfruit_orders(product, order_depth, lower_bound, upper_bound)

                result[product] = orders

        traderData = jsonpickle.encode(data)
        conversions = 1
        return result, conversions, traderData