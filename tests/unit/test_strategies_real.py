"""Real integration tests for trading strategies"""

import pytest
import numpy as np
from src.backtesting.strategies import (
    BinaryStrategy,
    BinaryLongOnlyStrategy,
    MultiClassStrategy,
    ThresholdStrategy,
    KellyStrategy,
)


class TestBinaryStrategy:
    def test_initialization(self):
        strategy = BinaryStrategy(name="Test")
        assert strategy.name == "Test"

    def test_long_position(self):
        strategy = BinaryStrategy()
        assert strategy.get_position(prediction=1) == 1.0

    def test_short_position(self):
        strategy = BinaryStrategy()
        assert strategy.get_position(prediction=0) == -1.0

    def test_pnl_long_win(self):
        strategy = BinaryStrategy()
        pnl = strategy.calculate_pnl(1.0, 5.0)
        assert pnl == 5.0


class TestMultiClassStrategy:
    def test_strong_rise(self):
        strategy = MultiClassStrategy(scale_positions=True)
        assert strategy.get_position(prediction=2) == 1.0

    def test_modest_rise(self):
        strategy = MultiClassStrategy(scale_positions=True)
        assert strategy.get_position(prediction=1) == 0.5

    def test_neutral(self):
        strategy = MultiClassStrategy(scale_positions=True)
        assert strategy.get_position(prediction=0) == 0.0


pytestmark = pytest.mark.unit
