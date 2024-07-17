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
        self.position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
        self.POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.amethysts_spread = 1
        self.amethysts_default_price = 10_000
        self.starfruit_spread = 1
        self.orchids_spread = 5
        self.std = 76.42438217375009 #33.389996883675984
        self.mean = 379.4904833333333 #411.9
        self.threshold = 0.5
        self.lower_threshold = 0.3
        self.upper_threshold = 0.8
        self.COCONUT_SUM = 299997029.5
        self.COUPON_SUM = 19051393.0
        self.COUPON_Z_SCORE = 0.5
        self.ITERS = 30_000
        self.COCONUT_STORE = []
        self.COCONUT_STORE_SIZE = 12
        self.COCONUT_BS_STORE = []
        self.delta_signs = 1
        self.time = 0

    def N(self, x):
        return 0.5 * (1+math.erf(x/math.sqrt(2)))

    def black_scholes_estimate(self, S, K, T, r, sigma, mean):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.N(d1) - K * np.exp(-r*T)* self.N(d2)
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


    def compute_orders(self, product, order_depth, acc_bid, acc_ask, orchids = False, market_make = False):
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

        if market_make:
            bid_price = acc_bid
            ask_price = acc_ask

        if acc_bid != -self.INF:
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

        if acc_ask != self.INF:
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
                import_duty = state.observations.conversionObservations["ORCHIDS"].importTariff
                export_duty = state.observations.conversionObservations["ORCHIDS"].exportTariff
                ask = state.observations.conversionObservations["ORCHIDS"].askPrice
                bid = state.observations.conversionObservations["ORCHIDS"].bidPrice

                buy_price = ask + shipment + import_duty
                sell_price = bid + shipment + export_duty

                lower = int(round(buy_price)) - 2
                upper = int(round(buy_price)) + 2

                orders += self.compute_orders(product, order_depth, lower, upper, orchids = True)
                conversions = -self.position[product]

            elif product == "GIFT_BASKET":

                # basket_items = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
                # mid_price = {}
                # worst_bid, worst_ask, best_bid, best_ask = {}, {}, {}, {}

                # for item in basket_items:
                #     sell_vol, best_sell_price = self.vol_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                #     buy_vol, best_buy_price = self.vol_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                #     mid_price[item] = (best_sell_price+best_buy_price)/2

                #     worst_bid[item] = min(state.order_depths[item].buy_orders.keys())
                #     worst_ask[item] = max(state.order_depths[item].sell_orders.keys())
                #     best_bid[item] = max(state.order_depths[item].buy_orders.keys())
                #     best_ask[item] = min(state.order_depths[item].sell_orders.keys())

                # difference = mid_price['GIFT_BASKET'] - 4*mid_price['CHOCOLATE'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'] - self.mean
                # prediction = 4*mid_price['CHOCOLATE'] + 6*mid_price['STRAWBERRIES'] + mid_price['ROSES'] + self.mean
                # orders += self.compute_orders(product, order_depth, int(round(prediction)) - 2, int(round(prediction)) + 2)
                orders = {'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': []}
                prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
                osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

                for p in prods:
                    osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
                    obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

                    best_sell[p] = next(iter(osell[p]))
                    best_buy[p] = next(iter(obuy[p]))

                    worst_sell[p] = next(reversed(osell[p]))
                    worst_buy[p] = next(reversed(obuy[p]))

                    mid_price[p] = (best_sell[p] + best_buy[p]) / 2
                    vol_buy[p], vol_sell[p] = 0, 0
                    for price, vol in obuy[p].items():
                        vol_buy[p] += vol 
                        if vol_buy[p] >= self.POSITION_LIMIT[p] / 10:
                            break
                    for price, vol in osell[p].items():
                        vol_sell[p] += -vol 
                        if vol_sell[p] >= self.POSITION_LIMIT[p] / 10:
                            break

                res_buy = mid_price['GIFT_BASKET'] - sum(mid_price[item] * qty for item, qty in zip(['CHOCOLATE', 'STRAWBERRIES', 'ROSES'], [4, 6, 1])) - 388
                res_sell = res_buy  # Simplified, adjust as necessary

                trade_at = self.basket_std * 0.5

                if res_sell > trade_at:
                    vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
                    if vol > 0:
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))
                elif res_buy < -trade_at:
                    vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
                    if vol > 0:
                        orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))

            elif product == 'COCONUT_COUPON':
                items = ['COCONUT', 'COCONUT_COUPON']
                mid_price, best_bid_price, best_ask_price = {}, {}, {}

                for item in items:
                    _, best_sell_price = self.vol_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                    _, best_buy_price = self.vol_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                    mid_price[item] = (best_sell_price+best_buy_price)/2
                    best_bid_price[item] = best_buy_price
                    best_ask_price[item] = best_sell_price

                self.COCONUT_SUM += mid_price['COCONUT']
                self.COUPON_SUM += mid_price['COCONUT_COUPON']
                self.ITERS += 1
                coconut_mean = self.COCONUT_SUM / self.ITERS
                coupon_mean = self.COUPON_SUM / self.ITERS

                self.COCONUT_STORE.append(mid_price['COCONUT'])

                store = np.array(self.COCONUT_STORE)
                mean, std = np.mean(store), np.std(store)
                curr_bs_est = self.black_scholes_estimate(S=mid_price['COCONUT'], K=10_000, T=250, r=0.001811, sigma=std+0.00000000000001, mean=mean)

                bs_mean = np.mean(self.COCONUT_BS_STORE) if len(self.COCONUT_BS_STORE) > 0 else 0

                modified_bs = (curr_bs_est - bs_mean) * 3.229 + curr_bs_est

                self.COCONUT_BS_STORE.append(modified_bs)

                # 25-50 store
                if len(self.COCONUT_STORE) >= self.COCONUT_STORE_SIZE:

                    curr_bs_est = self.black_scholes_estimate(S=mid_price['COCONUT'], K=10_000, T=250, r=0.001811, sigma=std, mean=mean)
                    prev_bs_est = self.black_scholes_estimate(S=self.PREV_COCONUT_PRICE, K=10_000, T=250, r=0.001811, sigma=std, mean=mean)


                    delta = mid_price['COCONUT_COUPON'] > curr_bs_est

                    if delta == self.delta_signs:
                        if self.time > 25:
                            change = curr_bs_est - prev_bs_est

                            predicted_coupon_price = change + self.PREV_COUPON_PRICE

                            lower_bound = int(round(predicted_coupon_price))-1
                            upper_bound = int(round(predicted_coupon_price))+1

                            orders += self.compute_orders(product, order_depth, lower_bound, upper_bound)

                    else:
                        self.time = 0

                    self.delta_signs = delta
                    self.time += 1


                    self.COCONUT_BS_STORE.pop(0)
                    self.COCONUT_STORE.pop(0)

                self.PREV_COCONUT_PRICE = mid_price['COCONUT']
                self.PREV_COUPON_PRICE = mid_price['COCONUT_COUPON']


            result[product] = orders

        traderData = jsonpickle.encode(data)
        return result, conversions, traderData