"""
Backtesting module for FED Watcher models.
Provides event-driven backtesting for trading strategies based on model predictions.
"""

from .model_predictor import ModelPredictor
from .backtest_engine import BacktestEngine
from .strategies import Strategy, BinaryStrategy, MultiClassStrategy

__all__ = [
    'ModelPredictor',
    'BacktestEngine',
    'Strategy',
    'BinaryStrategy',
    'MultiClassStrategy'
]
