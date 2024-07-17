import collections
from datamodel import OrderDepth, TradingState, Order, Observation
from typing import List, Dict
import numpy as np
import math
import pandas as pd
import json
import jsonpickle
import statistics

class RecordedData: 
    def __init__(self):
        self.POSITION = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
        self.LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.starfruit_cache = []
        self.STARFRUIT_CACHE_SIZE = 38
        self.AME_RANGE = 2
        self.ORCHID_MM_RANGE = 5

        self.DIFFERENCE_MEAN = 379.4904833333333
        self.DIFFERENCE_STD = 76.42438217375009
        self.PERCENT_OF_STD_TO_TRADE_AT = 0.5

        self.STRAWBERRY_CACHE = []
        self.CHOCOLATE_CACHE = []
        self.ROSE_CACHE = []

        self.STRAWBERRY_CACHE_SIZE = 4
        self.CHOCOLATE_CACHE_SIZE = 10
        self.ROSE_CACHE_SIZE = 10

        self.ITERS = 30_000
        self.COUPON_DIFFERENCE_STD = 13.381768062409492
        self.COCONUT_SUM = 299997029.5
        self.COUPON_SUM = 19051393.0
        self.COUPON_Z_SCORE = 0.5



class Trader:
    POSITION = {}
    LIMIT = {}
    INF = int(1e9)
    STARFRUIT_CACHE_SIZE = 38
    AME_RANGE = 2
    ORCHID_MM_RANGE = 5
    

    def estimate_lobf_price(self, cache):
        x = np.array([i for i in range(len(cache))])
        y = np.array(cache)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return int(round(len(cache) * m + c))


    # gets the total traded volume of each time stamp and best price
    # best price in buy_orders is the max; best price in sell_orders is the min
    # buy_order indicates orders are buy or sell orders
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
    

    # given estimated bid and ask prices, market take if there are good offers, otherwise market make 
    # by pennying or placing our bid/ask, whichever is more profitable
    def calculate_orders(self, product, order_depth, our_bid, our_ask, orchild=False, mm_at_our_price=False):
        orders: list[Order] = []
        
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        buy_vol, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        position = self.POSITION[product] if not orchild else 0
        limit = self.LIMIT[product]

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)

        if orchild:
            ask_price = max(best_sell_price-self.ORCHID_MM_RANGE, our_ask)
        if mm_at_our_price:
            bid_price = our_bid
            ask_price = our_ask

        if our_bid != -self.INF:
            # MARKET TAKE ASKS (buy items)
            for ask, vol in sell_orders.items():
                if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                    num_orders = min(-vol, limit - position)
                    position += num_orders
                    orders.append(Order(product, ask, num_orders))

            # MARKET MAKE BY PENNYING
            if position < limit:
                num_orders = limit - position
                orders.append(Order(product, bid_price, num_orders))
                position += num_orders

        # RESET POSITION
        position = self.POSITION[product] if not orchild else 0

        if our_ask != self.INF:
            # MARKET TAKE BIDS (sell items)
            for bid, vol in buy_orders.items():
                if position > -limit and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                    num_orders = max(-vol, -limit-position)
                    position += num_orders
                    orders.append(Order(product, bid, num_orders))

            # MARKET MAKE BY PENNYING
            if position > -limit:
                num_orders = -limit - position
                orders.append(Order(product, ask_price, num_orders))
                position += num_orders

        return orders

    
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        if state.traderData == '': # first run, set up data
            data = RecordedData()
        else:
            data = jsonpickle.decode(state.traderData)

        self.LIMIT = data.LIMIT
        self.INF = int(1e9)
        self.STARFRUIT_CACHE_SIZE = data.STARFRUIT_CACHE_SIZE
        self.AME_RANGE = data.AME_RANGE
        self.POSITION = data.POSITION
        self.ORCHID_MM_RANGE = data.ORCHID_MM_RANGE
        self.STRAWBERRY_CACHE_SIZE = data.STRAWBERRY_CACHE_SIZE
        self.CHOCOLATE_CACHE_SIZE = data.CHOCOLATE_CACHE_SIZE
        self.ROSE_CACHE_SIZE = data.ROSE_CACHE_SIZE

        # update our position 
        for product in state.order_depths:
            self.POSITION[product] = state.position[product] if product in state.position else 0


        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []
            
            if product == "AMETHYSTS":
                orders += self.calculate_orders(product, order_depth, 10000-self.AME_RANGE, 10000+self.AME_RANGE)
            
            elif product == "STARFRUIT":
                # keep the length of starfruit cache as STARFRUIT_CACHE_SIZE
                if len(data.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                    data.starfruit_cache.pop(0)

                _, best_sell_price = self.get_volume_and_best_price(order_depth.sell_orders, buy_order=False)
                _, best_buy_price = self.get_volume_and_best_price(order_depth.buy_orders, buy_order=True)

                data.starfruit_cache.append((best_sell_price+best_buy_price)/2)

                # if cache size is maxed, calculate next price and place orders
                lower_bound = -self.INF
                upper_bound = self.INF

                if len(data.starfruit_cache) == self.STARFRUIT_CACHE_SIZE:
                    lower_bound = self.estimate_lobf_price(data.starfruit_cache)-2
                    upper_bound = self.estimate_lobf_price(data.starfruit_cache)+2

                orders += self.calculate_orders(product, order_depth, lower_bound, upper_bound)
            
            elif product == 'ORCHIDS':
                shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
                import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
                export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
                ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
                ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice

                buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
                sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

                lower_bound = int(round(buy_from_ducks_prices))-1
                upper_bound = int(round(buy_from_ducks_prices))+1

                orders += self.calculate_orders(product, order_depth, lower_bound, upper_bound, orchild=True)
                conversions = -self.POSITION[product]
            
            elif product == 'GIFT_BASKET':
                basket_items = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
                mid_price = {}
                worst_bid_price, worst_ask_price, best_bid_price, best_ask_price = {}, {}, {}, {}

                for item in basket_items:
                    _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                    _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                    mid_price[item] = (best_sell_price+best_buy_price)/2

                    worst_bid_price[item] = min(state.order_depths[item].buy_orders.keys())
                    worst_ask_price[item] = max(state.order_depths[item].sell_orders.keys())
                    best_bid_price[item] = max(state.order_depths[item].buy_orders.keys())
                    best_ask_price[item] = min(state.order_depths[item].sell_orders.keys())

                difference = mid_price['GIFT_BASKET'] - 4*mid_price['CHOCOLATE'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'] - data.DIFFERENCE_MEAN

                if difference > data.PERCENT_OF_STD_TO_TRADE_AT * data.DIFFERENCE_STD:
                    # basket overvalued, sell
                    orders += self.calculate_orders(product, order_depth, -self.INF, best_bid_price['GIFT_BASKET'], mm_at_our_price=True)
           
                elif difference < -data.PERCENT_OF_STD_TO_TRADE_AT * data.DIFFERENCE_STD:
                    # basket undervalued, buy
                    orders += self.calculate_orders(product, order_depth, best_ask_price['GIFT_BASKET'], self.INF, mm_at_our_price=True)

            elif product == 'COCONUT_COUPON':
                items = ['COCONUT', 'COCONUT_COUPON']
                mid_price, best_bid_price, best_ask_price = {}, {}, {}

                for item in items:
                    _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                    _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                    mid_price[item] = (best_sell_price+best_buy_price)/2
                    best_bid_price[item] = best_buy_price
                    best_ask_price[item] = best_sell_price

                data.COCONUT_SUM += mid_price['COCONUT']
                data.COUPON_SUM += mid_price['COCONUT_COUPON']
                data.ITERS += 1
                coconut_mean = data.COCONUT_SUM / data.ITERS
                coupon_mean = data.COUPON_SUM / data.ITERS

                predicted_coupon_price = (mid_price['COCONUT'] - coconut_mean) * 0.503 + coupon_mean
                difference = mid_price['COCONUT_COUPON'] - predicted_coupon_price

                if difference > data.COUPON_Z_SCORE * data.COUPON_DIFFERENCE_STD:
                    # coupons overvalued, sell
                    orders += self.calculate_orders(product, order_depth, -self.INF, best_bid_price['COCONUT_COUPON'])
                
                elif difference < -data.COUPON_Z_SCORE * data.COUPON_DIFFERENCE_STD:
                    # coupons undervalued, buy
                    orders += self.calculate_orders(product, order_depth, best_ask_price['COCONUT_COUPON'], self.INF)

            # update orders for current product
            if len(orders) > 0:
                result[product] = orders

        traderData = jsonpickle.encode(data)

        return result, conversions, traderData