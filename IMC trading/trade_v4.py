import collections
from datamodel import OrderDepth, TradingState, Order,ConversionObservation, Observation
from typing import List, Dict
import numpy as np
import math
import pandas as pd
import json
import jsonpickle
import statistics

class Trader:
    def __init__(self):
        self.INF = int(1e9)
        self.amethysts_history = []
        self.starfruit_history = []
        self.starfruit_cache_size = 35
        self.amethysts_cache_size = 5 # 10
        self.position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}
        self.POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
        self.amethysts_spread = 1
        self.amethysts_default_price = 10_000
        self.starfruit_spread = 1
        self.buy_orchids = False
        self.sell_orchids = False
        self.clear_orchids = False
        self.last_orchid = 0
        self.sunlight_value = 0
        self.humidity_value = 0
        self.steps = 0
        self.start_sunlight = 0
        self.last_sunlight = -1
        self.last_humidity = -1
        self.last_export = -1
        self.last_import = -1
        self.std = 25    
        self.basket_std = 50 # 191.1808805 standard deviation

        self.cont_buy_basket_unfill = 0
        self.cont_sell_basket_unfill = 0
        
        self.rate = 0.04669
        self.years = 1
        self.Tstep = 246
        self.stdev_cc = 0.0240898021
        self.strike = 10000
        self.buy_coconut = False
        self.sell_coconut = False
        self.buy_coupon = False
        self.sell_coupon = False

    def values_extract(self, order_dict, buy=0):
        volume = 0
        best = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            volume += vol
            if volume > mxvol:
                mxvol = vol
                best = ask
        
        return volume, best

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
    
    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(osell)
        buy_vol, best_buy_price = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_price + best_buy_price)/2
        mprice_ours = (acc_bid+acc_ask)/2

        undercut_buy = best_buy_price + 1
        undercut_sell = best_sell_price - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(20, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 20):
            num = min(20, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 2), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(20, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-20, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -20):
            num = max(-20, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-20, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def compute_orders_starfruit(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
    def compute_orders_orchids(self,order_depth, convobv, timestamp):    
        orders = {'ORCHIDS' : []}
        prods = ['ORCHIDS']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
       
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            osunlight = convobv[p].sunlight
            ohumidity = convobv[p].humidity
            oshipping = convobv[p].transportFees
            oexport = convobv[p].exportTariff
            oimport = convobv[p].importTariff
            southbid = convobv[p].bidPrice
            southask = convobv[p].askPrice
            
            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
            
            
        if self.last_sunlight != -1 and (osunlight - self.last_sunlight > 1) and (mid_price['ORCHIDS'] - self.last_orchid > 2.33):
            self.buy_orchids = True
        if self.last_export != -1 and ((oexport - self.last_export >= 1.5) or (oexport - self.last_export<= -1.5)):
            self.buy_orchids = True
        if self.last_export != -1 and (oexport - self.last_export == 1):
            self.sell_orchids = True
            
        # export tariff only changes by increments of 1
        # import tariff only by 0.2

        if self.buy_orchids and self.sell_orchids:
            self.buy_orchids = False
            self.sell_orchids = False
            
        if self.position['ORCHIDS'] > 0 and self.sell_orchids == False:
            self.buy_orchids = False
            self.sell_orchids = False
            self.clear_orchids = True
           
        # stop gap so we don't exceed position limit    
        if self.position['ORCHIDS'] == self.POSITION_LIMIT['ORCHIDS']:#self.buy_orchids and self.position['ORCHIDS'] == self.POSITION_LIMIT['ORCHIDS']:
            self.buy_orchids = False
        elif self.position['ORCHIDS'] == -self.POSITION_LIMIT['ORCHIDS']:#self.sell_orchids and self.position['ORCHIDS'] == -self.POSITION_LIMIT['ORCHIDS']:
            self.sell_orchids = False
            
        if self.clear_orchids:
            vol = round(math.sqrt(self.position['ORCHIDS']))
            orders['ORCHIDS'].append(Order('ORCHIDS', worst_buy['ORCHIDS'], -vol))
        if self.buy_orchids:
            vol = self.POSITION_LIMIT['ORCHIDS']  - self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', (best_sell['ORCHIDS']), vol))     
        if self.sell_orchids:
            vol = self.POSITION_LIMIT['ORCHIDS'] + self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', (best_buy['ORCHIDS']), -vol))
                
        self.last_export = convobv['ORCHIDS'].exportTariff
        self.last_sunlight = convobv['ORCHIDS'].sunlight
        self.last_orchid = mid_price['ORCHIDS']

        return orders
    
    def conversion_opp(self, convobv, timestamp):
        conversions = [1]
        prods = ['ORCHIDS']
        
        return sum(conversions)
    
    def compute_orders_basket(self, order_depth):

        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 388
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 388

        trade_at = self.basket_std*0.5
        close_at = self.basket_std*(-1000)

        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']

        uku_pos = self.position['ROSES']
        uku_neg = self.position['ROSES']


        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
                #uku_pos += vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol

        return orders
    
    def calculate_coupon_binomial(self, S, K, T, r, sigma, steps):
        # Precompute constants
        dt = T / steps  # Delta t
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        discount_factor = np.exp(-r * dt)  # Discount factor for each step

        # Initialize asset prices at maturity
        stock_prices = S * d**np.arange(steps, -1, -1) * u**np.arange(0, steps + 1, 1)
        option_values = np.maximum(stock_prices - K, 0)  # Call option payoff

        # Iterate backwards through the tree
        for i in range(steps - 1, -1, -1):
            option_values[:-1] = (p * option_values[1:] + (1 - p) * option_values[:-1]) * discount_factor

        return option_values[0]
        
    def coconuts_and_coupons(self, order_depth):
        orders = {'COCONUT' : [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        S = mid_price['COCONUT']  # Current price of the underlying asset
        K = self.strike  # Strike price of the option
        T = self.years  # Time to expiration in years
        r = self.rate  # Annual risk-free rate
        sigma = self.stdev_cc  # Volatility of the underlying asset
        steps = self.Tstep  # Number of steps in the binomial model

        calculated_coupon = self.calculate_coupon_binomial(S, K, T, r, sigma, steps)
        
        # u = math.exp(self.stdev_cc*math.sqrt(self.years/self.Tstep))
        # d = 1/u
        # probup = (((math.exp((self.rate)*self.years/self.Tstep)) - d) / (u - d))
        # discount_factor = math.exp(self.rate/self.Tstep)
        # duration_of_time_step = (self.years/self.Tstep)
        # payoffs = []
        # for n in range(self.Tstep+1): 
        #     payoffs.append(max(0, (mid_price['COCONUT']*(u**((self.Tstep)-n))*(d**n) - self.strike)))   
        
        # for x in reversed(range(1, self.Tstep+1)):
        #     discounting1 = []
        #     for i in range(0,x):
        #         discounting1.append((((probup)*payoffs[i]) + ((1-probup)*payoffs[i+1])) / (math.exp(discount_factor)))
                
        #     payoffs.clear()
        #     payoffs.extend(discounting1)
        # calculated_coupon = discounting1[0]
# *************************************End of calculations************************************
        
        # coconut logic

        # if mid_price['COCONUT'] > 10000:
        #     self.sell_coconut = True
        # else:
        #     self.buy_coconut = True
        
        # if self.position['COCONUT'] == self.POSITION_LIMIT['COCONUT']:
        #     self.buy_coconut = False
        # if self.position['COCONUT'] == -self.POSITION_LIMIT['COCONUT']:
        #     self.sell_coconut = False
            
        # if self.buy_coconut:
        #     vol = self.POSITION_LIMIT['COCONUT'] - self.position['COCONUT']
        #     orders['COCONUT'].append(Order('COCONUT', best_sell['COCONUT'], vol))
        # if self.sell_coconut:
        #     vol = self.POSITION_LIMIT['COCONUT'] + self.position['COCONUT']
        #     orders['COCONUT'].append(Order('COCONUT', best_buy['COCONUT'], -vol))
            

        # coconut coupon logic

        if mid_price['COCONUT'] > calculated_coupon:
            self.sell_coupon = True
        else:
            self.buy_coupon = True

        if self.position['COCONUT_COUPON'] == self.POSITION_LIMIT['COCONUT_COUPON']:
            self.buy_coupon = False
        if self.position['COCONUT_COUPON'] == -self.POSITION_LIMIT['COCONUT_COUPON']:
            self.sell_coupon = False
            
        if self.buy_coupon:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] - self.position['COCONUT_COUPON']
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_sell['COCONUT_COUPON'], vol))
        if self.sell_coupon:
            vol = self.POSITION_LIMIT['COCONUT_COUPON'] + self.position['COCONUT_COUPON']
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_buy['COCONUT_COUPON'], -vol))

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_starfruit(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])

    def run(self, state: TradingState):
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [],  'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': [], 'COCONUT': [], 'COCONUT_COUPON': []}

        # Iterate over all the keys (the available products) contained in the order depths
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

        timestamp = state.timestamp

        if len(self.starfruit_history) == self.starfruit_cache_size:
            self.starfruit_history.pop(0)

        _, best_sell_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, best_buy_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.starfruit_history.append((best_buy_starfruit+best_sell_starfruit)/2)

        starfruit_lower_bound = -self.INF
        starfruit_upper_bound = self.INF

        if len(self.starfruit_history) == self.starfruit_cache_size:
            starfruit_lower_bound = self.predict_starfruit_price(self.starfruit_history)-self.starfruit_spread
            starfruit_upper_bound = self.predict_starfruit_price(self.starfruit_history)+self.starfruit_spread

        if len(self.amethysts_history) == self.amethysts_cache_size:
            self.amethysts_history.pop(0)

        _, best_sell_amethysts = self.values_extract(collections.OrderedDict(sorted(state.order_depths['AMETHYSTS'].sell_orders.items())))
        _, best_buy_amethysts = self.values_extract(collections.OrderedDict(sorted(state.order_depths['AMETHYSTS'].buy_orders.items(), reverse=True)), 1)

        self.amethysts_history.append((best_buy_amethysts+best_sell_amethysts)/2)

        amethysts_lower_bound = -self.INF
        amethysts_upper_bound = self.INF

        if len(self.amethysts_history) == self.amethysts_cache_size:
            amethysts_lower_bound = self.predict_amethysts_price(self.amethysts_history)-self.amethysts_spread
            amethysts_upper_bound = self.predict_amethysts_price(self.amethysts_history)+self.amethysts_spread

        acc_bid = {'AMETHYSTS' : amethysts_lower_bound, 'STARFRUIT' : starfruit_lower_bound}
        acc_ask = {'AMETHYSTS' : amethysts_upper_bound, 'STARFRUIT' : starfruit_upper_bound}
        self.steps += 1
         
        # orders for the different products
          
        orders = self.compute_orders_orchids(state.order_depths, state.observations.conversionObservations, state.timestamp)
        result['ORCHIDS'] += orders['ORCHIDS']
        orders = self.compute_orders_basket(state.order_depths)
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['ROSES'] += orders['ROSES']
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        orders = self.coconuts_and_coupons(state.order_depths)
        result['COCONUT'] += orders['COCONUT']
        result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
        
        for product in ['AMETHYSTS', 'STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
            result[product] += orders

        traderdata = "random_returns" 
        
        conversions = self.conversion_opp(state.observations.conversionObservations, state.timestamp)

        return result, conversions, traderdata