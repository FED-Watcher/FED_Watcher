"""
Trading strategy definitions for backtesting.
Defines how predictions are converted to positions and P&L.
"""

from abc import ABC, abstractmethod
import numpy as np


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, name="Strategy"):
        """
        Initialize strategy.

        Args:
            name (str): Strategy name for identification
        """
        self.name = name

    @abstractmethod
    def get_position(self, prediction, **kwargs):
        """
        Determine position based on model prediction.

        Args:
            prediction: Model prediction
            **kwargs: Additional strategy parameters

        Returns:
            float: Position size (-1 for short, 0 for neutral, +1 for long)
        """
        pass

    def calculate_pnl(self, position, actual_return):
        """
        Calculate P&L for a single trade.

        Args:
            position (float): Position size (-1 to +1)
            actual_return (float): Actual market return (%)

        Returns:
            float: P&L in percentage terms
        """
        return position * actual_return

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class BinaryStrategy(Strategy):
    """
    Simple binary strategy: Long on Up prediction, Short on Down prediction.

    Assumptions:
    - Up prediction (1) → Go Long (position = +1)
    - Down prediction (0) → Go Short (position = -1)
    - No neutral position
    """

    def __init__(self, name="Binary Long/Short"):
        super().__init__(name)

    def get_position(self, prediction, **kwargs):
        """
        Convert binary prediction to position.

        Args:
            prediction (int): 0 (Down) or 1 (Up)

        Returns:
            float: +1 for long, -1 for short
        """
        if prediction == 1:
            return 1.0  # Long
        else:
            return -1.0  # Short


class BinaryLongOnlyStrategy(Strategy):
    """
    Binary long-only strategy: Long on Up, Flat on Down.

    Assumptions:
    - Up prediction (1) → Go Long (position = +1)
    - Down prediction (0) → Stay Flat (position = 0)
    """

    def __init__(self, name="Binary Long-Only"):
        super().__init__(name)

    def get_position(self, prediction, **kwargs):
        """
        Convert binary prediction to position.

        Args:
            prediction (int): 0 (Down) or 1 (Up)

        Returns:
            float: +1 for long, 0 for flat
        """
        if prediction == 1:
            return 1.0  # Long
        else:
            return 0.0  # Flat


class MultiClassStrategy(Strategy):
    """
    Multi-class strategy with scaled positions based on predicted magnitude.

    Assumptions:
    - Strong Rise (+2) → Full Long (position = +1.0)
    - Modest Rise (+1) → Half Long (position = +0.5)
    - Neutral (0) → Flat (position = 0)
    - Modest Drop (-1) → Half Short (position = -0.5)
    - Strong Drop (-2) → Full Short (position = -1.0)
    """

    def __init__(self, name="Multi-Class Scaled", scale_positions=True):
        """
        Initialize multi-class strategy.

        Args:
            name (str): Strategy name
            scale_positions (bool): If True, scale positions by magnitude.
                                   If False, use binary (+1, 0, -1) positions.
        """
        super().__init__(name)
        self.scale_positions = scale_positions

    def get_position(self, prediction, **kwargs):
        """
        Convert multi-class prediction to position.

        Args:
            prediction (int): -2, -1, 0, 1, 2

        Returns:
            float: Position size scaled by predicted magnitude
        """
        if self.scale_positions:
            # Scale by magnitude: -2→-1.0, -1→-0.5, 0→0, 1→0.5, 2→1.0
            position_map = {-2: -1.0, -1: -0.5, 0: 0.0, 1: 0.5, 2: 1.0}
            return position_map.get(prediction, 0.0)
        else:
            # Binary positions: negative→-1, zero→0, positive→+1
            if prediction > 0:
                return 1.0
            elif prediction < 0:
                return -1.0
            else:
                return 0.0


class ThresholdStrategy(Strategy):
    """
    Strategy that only takes positions when probability exceeds threshold.

    Useful for filtering low-confidence predictions.
    """

    def __init__(self, name="Threshold Strategy", threshold=0.6, base_strategy=None):
        """
        Initialize threshold strategy.

        Args:
            name (str): Strategy name
            threshold (float): Minimum probability to take position (0-1)
            base_strategy (Strategy): Underlying strategy to use when threshold exceeded
        """
        super().__init__(name)
        self.threshold = threshold
        self.base_strategy = base_strategy or BinaryStrategy()

    def get_position(self, prediction, probabilities=None, **kwargs):
        """
        Get position only if probability exceeds threshold.

        Args:
            prediction: Model prediction
            probabilities (np.ndarray): Prediction probabilities (optional)

        Returns:
            float: Position from base strategy if threshold met, else 0
        """
        if probabilities is None:
            # No probability info, use base strategy
            return self.base_strategy.get_position(prediction, **kwargs)

        # Get max probability
        max_prob = np.max(probabilities)

        if max_prob >= self.threshold:
            return self.base_strategy.get_position(prediction, **kwargs)
        else:
            return 0.0  # Flat - low confidence


class KellyStrategy(Strategy):
    """
    Kelly Criterion position sizing based on win rate and odds.

    Adjusts position size based on expected value of the bet.
    """

    def __init__(
        self, name="Kelly Criterion", win_rate=0.6, avg_win=1.0, avg_loss=1.0, kelly_fraction=0.5
    ):
        """
        Initialize Kelly strategy.

        Args:
            name (str): Strategy name
            win_rate (float): Historical win rate (0-1)
            avg_win (float): Average win size (%)
            avg_loss (float): Average loss size (%)
            kelly_fraction (float): Fraction of Kelly bet to use (0-1, default 0.5 for half-Kelly)
        """
        super().__init__(name)
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction

    def get_position(self, prediction, **kwargs):
        """
        Calculate Kelly position size.

        Args:
            prediction: Model prediction (direction)

        Returns:
            float: Position size scaled by Kelly criterion
        """
        # Calculate Kelly fraction: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        p = self.win_rate
        q = 1 - p
        b = self.avg_win / self.avg_loss

        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, min(1, kelly_fraction))  # Clamp to [0, 1]

        # Apply fractional Kelly
        position_size = kelly_fraction * self.kelly_fraction

        # Apply direction from prediction
        if hasattr(prediction, "__iter__"):
            # Multi-class
            if prediction > 0:
                return position_size
            elif prediction < 0:
                return -position_size
            else:
                return 0.0
        else:
            # Binary
            if prediction == 1:
                return position_size
            else:
                return -position_size
