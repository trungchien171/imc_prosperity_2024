
import collections
from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import List
import numpy as np
import math
import pandas as pd
import json
import jsonpickle

class Vwap_amethysts:
    def __init__(self, bv=0, sv=0, bpv=0, spv=0):
        self.buy_volume = bv
        self.sell_volume = sv
        self.buy_price_volume = bpv
        self.sell_price_volume = spv

class Data:
    def __init__(self):
        self.amethysts_vwap = Vwap_amethysts()
        self.starfruit_cache = []
        self.orchids_data = []

class Trader:
    INF = int(1e9)
    STARFRUIT_CACHE_SIZE = 10
    position = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0}
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100}
    amethysts_spread = 1
    amethysts_default_price = 10_000
    starfruit_spread = 1
    
    def __init__(self):
        self.data = Data()

    def predict_amethysts_price(self, orders):
        total_volume = sum(amount for _, amount in orders.items())
        total_price_volume = sum(price * amount for price, amount in orders.items())
        if total_volume == 0:
            return 0
        return total_price_volume / total_volume

    def calculate_orchids_price_impact(self, sunlight, humidity):
        sunlight_hours = sunlight / 2500
        if sunlight_hours < 7:
            sunlight_decrease = (420 - sunlight_hours * 60) / 10 * 4
        else:
            sunlight_decrease = 0
        if humidity < 60:
            humidity_decrease = (60 - humidity) / 5 * 2
        elif humidity > 80:
            humidity_decrease = (humidity - 80) / 5 * 2
        else:
            humidity_decrease = 0
        return 100 - (sunlight_decrease + humidity_decrease)

    def make_trading_decision(self, observation: ConversionObservation):
        predicted_price = observation.askPrice * self.calculate_orchids_price_impact(observation.sunlight, observation.humidity) / 100
        if predicted_price < observation.askPrice and self.position['ORCHIDS'] < self.POSITION_LIMIT['ORCHIDS']:
            self.buy_orchids(observation.askPrice)
        elif predicted_price > observation.bidPrice and self.position['ORCHIDS'] > 0:
            self.sell_orchids(observation.bidPrice)

    def buy_orchids(self, price):
        # Placeholder for buying logic
        self.position['ORCHIDS'] += 1
        print(f"Bought ORCHIDS at {price}")

    def sell_orchids(self, price):
        # Placeholder for selling logic
        self.position['ORCHIDS'] -= 1
        print(f"Sold ORCHIDS at {price}")

# Sample usage in simulation or actual trading scenario
trader = Trader()
sample_observation = ConversionObservation(1200, 1180, 1.5, 9.5, -2.0, 18000, 70)
trader.make_trading_decision(sample_observation)
