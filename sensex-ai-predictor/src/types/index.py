class StockData:
    def __init__(self, symbol: str, current_value: float, previous_value: float):
        self.symbol = symbol
        self.current_value = current_value
        self.previous_value = previous_value

    def price_change(self) -> float:
        return self.current_value - self.previous_value

    def percentage_change(self) -> float:
        if self.previous_value == 0:
            return 0.0
        return (self.price_change() / self.previous_value) * 100


class Prediction:
    def __init__(self, stock_data: StockData, predicted_value: float):
        self.stock_data = stock_data
        self.predicted_value = predicted_value

    def performance_indicator(self) -> float:
        return (self.predicted_value - self.stock_data.current_value) / self.stock_data.current_value * 100


class MarketAnalysis:
    def __init__(self, predictions: list):
        self.predictions = predictions

    def best_performing_stock(self) -> Prediction:
        return max(self.predictions, key=lambda p: p.performance_indicator()) if self.predictions else None