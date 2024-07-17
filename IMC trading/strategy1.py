from datamodel import Listing, ConversionObservation, Observation, Trade, ProsperityEncoder, OrderDepth, UserId, TradingState, Order
from typing import List
import string
import math

DEFAULT_PRICES = {
    AMETHYSTS: 10_000,
    STARFRUIT: 5_000
}

class Trader:
    def __init__(self) -> None:
        self.round = 0
        self.cash = 0
        self.past_prices = dict()
        for product in Listing.product:
            self.past_prices[product] = []

        self.ema_prices = dict()
        for product in Listing.product:
            self.ema_prices[product] = None

        self.ema_param = 0.5
    
    def get_position(self, product, state: TradingState):
        return state.position.get(product, 0)
    
    def get_mid_price(self, product, state: TradingState):
        default_price = self.ema_prices[product]
        if default_price is None:
            default_price = DEFAULT_PRICES[product]
        if product not in state.order_depths:
            return default_price
        
        market_bids = state.order_depths[product].buy_orders
        if len(market_bids) == 0:
            return default_price
        
        market_asks = state.order_depths[product].sell_orders
        if len(market_asks) == 0:
            return default_price
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def get_value_on_product(self, product, state: TradingState):
        return self.get_position(product, state) * self.get_mid_price(product, state)
    
    def update_pnl(self, state: TradingState):
        def update_cash():
            for product in state.own_trades:
                for trade in state.own_trades[product]:
                    if trade.timestamp != state.timestamp - 100:
                        continue

                    if trade.buyer == "SAMPLE":
                        self.cash -= trade.quantity * trade.price
                    if trade.seller == "SAMPLE":
                        self.cash += trade.quantity * trade.price
        def get_value_on_positions():
            value = 0
            for product in state.position:
                value += self.get_value_on_product(product, state)
            return value
        
        update_cash()
        return self.cash + get_value_on_positions()
    
    def update_ema_prices(self, state : TradingState):
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in Listing.product:
            mid_price = self.get_mid_price(product, state)
            if mid_price is None:
                continue

            if self.ema_prices[product] is None:
                self.ema_prices[product] = mid_price
            else:
                self.ema_prices[product] = self.ema_param * mid_price + (1-self.ema_param) * self.ema_prices[product]
    
    def strategy(self, state : TradingState):

        position_sample = self.get_position("SAMPLE", state)

        bid_volume = self.position_limit["SAMPLE"] - position_sample
        ask_volume = - self.position_limit["SAMPLE"] - position_sample

        orders = []

        if position_sample == 0:
            # Not long nor short
            orders.append(Order("SAMPLE", math.floor(self.ema_prices["SAMPLE"] - 1), bid_volume))
            orders.append(Order("SAMPLE", math.ceil(self.ema_prices["SAMPLE"] + 1), ask_volume))
        
        if position_sample > 0:
            # Long position
            orders.append(Order("SAMPLE", math.floor(self.ema_prices["SAMPLE"] - 2), bid_volume))
            orders.append(Order("SAMPLE", math.ceil(self.ema_prices["SAMPLE"]), ask_volume))

        if position_sample < 0:
            # Short position
            orders.append(Order("SAMPLE", math.floor(self.ema_prices["SAMPLE"]), bid_volume))
            orders.append(Order("SAMPLE", math.ceil(self.ema_prices["SAMPLE"] + 2), ask_volume))

        return orders

    
    def run(self, state: TradingState):
        self.round += 1
        pnl = self.update_pnl(state)
        self.update_ema_prices(state)

        for product in state.own_trades:
            for trade in state.own_trades[product]:
                if trade.timestamp == state.timestamp - 100:
                    print(trade)
        
        result = {}
        result["SAMPLE"] =  self.strategy(state)
    
        traderData = "SAMPLE"
        
        conversions = 1
        return result, conversions, traderData