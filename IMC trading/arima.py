import collections
from datamodel import OrderDepth, TradingState, Order
from typing import List
import numpy as np

class Trader:

    position = {'AMETHYSTS': 0, 'STARFRUIT': 0}
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    
    def compute_orders_amethysts_or_starfruit(self, product, order_depth, acc_bid, acc_ask):
        orders: List[Order] = []

        sell_ord = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_ord = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        cur_pos = self.position[product]

        for ask, vol in sell_ord.items():
            if ask < acc_bid and cur_pos < self.POSITION_LIMIT[product]:
                order_for = min(-vol, self.POSITION_LIMIT[product] - cur_pos)
                cur_pos += order_for
                orders.append(Order(product, ask, order_for))

        for bid, vol in buy_ord.items():
            if bid > acc_ask and cur_pos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product] - cur_pos)
                cur_pos += order_for
                orders.append(Order(product, bid, order_for))

        return orders

    def run(self, state: TradingState):
        result = {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == 'AMETHYSTS':
                acc_bid, acc_ask = 10001, 9999
                
                orders = self.compute_orders_amethysts_or_starfruit(product, order_depth, acc_bid, acc_ask)

                result[product] = orders

            if product == "STARFRUIT":
                acceptable_price = -0.0210 + (-0.7425)*(state.timestamp - 100) + np.sqrt(1.8348)
                acc_bid, acc_ask = acceptable_price, acceptable_price

                orders = self.compute_orders_amethysts_or_starfruit(product, order_depth, acc_bid, acc_ask)

                result[product] = orders

        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData