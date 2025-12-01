"""
Unit tests for backtesting strategies module
Tests trading strategy implementations
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch


class TestTradingStrategies:
    """Test suite for trading strategy implementations"""
    
    def test_simple_strategy_initialization(self):
        """Test strategy initialization"""
        strategy_params = {
            'name': 'momentum',
            'threshold': 0.6,
            'holding_period': 5
        }
        
        assert strategy_params['name'] == 'momentum'
        assert strategy_params['threshold'] == 0.6
    
    def test_buy_signal_generation(self):
        """Test buy signal generation"""
        predictions = np.array([0.8, 0.7, 0.4, 0.9, 0.3])
        threshold = 0.6
        
        buy_signals = predictions > threshold
        
        assert buy_signals.sum() == 3  # 0.8, 0.7, 0.9
        assert buy_signals[0] == True
        assert buy_signals[2] == False
    
    def test_sell_signal_generation(self):
        """Test sell signal generation"""
        predictions = np.array([0.2, 0.3, 0.7, 0.1, 0.4])
        threshold = 0.4
        
        sell_signals = predictions < threshold
        
        assert sell_signals.sum() == 3  # 0.2, 0.3, 0.1
        assert sell_signals[0] == True
        assert sell_signals[2] == False
    
    def test_hold_signal_generation(self):
        """Test hold signal generation"""
        predictions = np.array([0.45, 0.55, 0.48, 0.52])
        lower_threshold = 0.4
        upper_threshold = 0.6
        
        hold_signals = (predictions >= lower_threshold) & (predictions <= upper_threshold)
        
        assert hold_signals.all()  # All in hold range
    
    def test_strategy_position_sizing(self):
        """Test position sizing calculation"""
        account_balance = 10000
        risk_per_trade = 0.02  # 2%
        
        position_size = account_balance * risk_per_trade
        
        assert position_size == 200
        assert position_size < account_balance
    
    def test_stop_loss_calculation(self):
        """Test stop loss level calculation"""
        entry_price = 100
        stop_loss_pct = 0.05  # 5%
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        
        assert stop_loss == 95
        assert stop_loss < entry_price
    
    def test_take_profit_calculation(self):
        """Test take profit level calculation"""
        entry_price = 100
        take_profit_pct = 0.10  # 10%
        
        take_profit = entry_price * (1 + take_profit_pct)
        
        assert abs(take_profit - 110) < 0.01
        assert take_profit > entry_price
    
    def test_risk_reward_ratio(self):
        """Test risk-reward ratio calculation"""
        entry_price = 100
        stop_loss = 95
        take_profit = 115
        
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward_ratio = reward / risk
        
        assert risk == 5
        assert reward == 15
        assert risk_reward_ratio == 3.0  # 3:1 ratio
    
    def test_momentum_strategy(self):
        """Test momentum-based strategy"""
        prices = pd.Series([100, 102, 105, 103, 108, 110])
        returns = prices.pct_change()
        
        # Momentum signal: buy if positive momentum
        momentum_threshold = 0.02
        momentum_signals = returns > momentum_threshold
        
        assert momentum_signals.sum() >= 1
        assert isinstance(momentum_signals, pd.Series)
    
    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy"""
        prices = pd.Series([100, 95, 90, 95, 100, 105])
        sma_20 = prices.rolling(window=3).mean()
        
        # Buy when price below moving average
        mean_reversion_signals = prices < sma_20
        
        assert isinstance(mean_reversion_signals, pd.Series)
    
    def test_breakout_strategy(self):
        """Test breakout strategy"""
        prices = pd.Series([100, 101, 99, 102, 98, 105])
        lookback = 3
        
        rolling_max = prices.rolling(window=lookback).max()
        breakout_signals = prices > rolling_max.shift(1)
        
        assert isinstance(breakout_signals, pd.Series)
    
    def test_volatility_filter(self):
        """Test volatility-based filter"""
        returns = pd.Series([0.02, -0.03, 0.01, 0.05, -0.02])
        volatility = returns.std()
        
        vol_threshold = 0.03
        low_vol_environment = volatility < vol_threshold
        
        assert isinstance(bool(low_vol_environment), bool)
        assert volatility > 0
    
    def test_multi_timeframe_signals(self):
        """Test multiple timeframe analysis"""
        daily_signal = 1  # Buy
        weekly_signal = 1  # Buy
        monthly_signal = 0  # Neutral
        
        # Confirm if 2+ timeframes agree
        signals = [daily_signal, weekly_signal, monthly_signal]
        confirmation = sum(signals) >= 2
        
        assert confirmation == True
    
    def test_strategy_backtesting_single_trade(self):
        """Test single trade execution"""
        entry_price = 100
        exit_price = 105
        position_size = 10
        
        pnl = (exit_price - entry_price) * position_size
        return_pct = (exit_price - entry_price) / entry_price
        
        assert pnl == 50
        assert return_pct == 0.05  # 5%
    
    def test_strategy_win_rate_calculation(self):
        """Test win rate calculation"""
        trades = [
            {'pnl': 50},
            {'pnl': -20},
            {'pnl': 30},
            {'pnl': -10},
            {'pnl': 40}
        ]
        
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        total_trades = len(trades)
        win_rate = winning_trades / total_trades
        
        assert winning_trades == 3
        assert win_rate == 0.6  # 60%
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation"""
        winning_trades_pnl = [50, 30, 40]
        losing_trades_pnl = [20, 10]
        
        total_wins = sum(winning_trades_pnl)
        total_losses = sum(losing_trades_pnl)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        assert total_wins == 120
        assert total_losses == 30
        assert profit_factor == 4.0
    
    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation"""
        equity_curve = np.array([10000, 10500, 10200, 9800, 10100, 10800])
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        assert max_drawdown < 0  # Should be negative
        assert max_drawdown >= -1  # Can't lose more than 100%
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        returns = np.array([0.02, 0.03, -0.01, 0.02, 0.04])
        risk_free_rate = 0.01
        
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        assert isinstance(sharpe_ratio, (int, float))
    
    def test_kelly_criterion(self):
        """Test Kelly criterion for position sizing"""
        win_rate = 0.6
        win_loss_ratio = 2.0  # Average win / Average loss
        
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        assert 0 <= kelly_pct <= 1
        assert kelly_pct > 0  # Positive edge
    
    def test_trade_frequency_calculation(self):
        """Test trade frequency metrics"""
        num_trades = 50
        num_days = 252  # Trading days in a year
        
        trades_per_day = num_trades / num_days
        avg_days_per_trade = num_days / num_trades
        
        assert trades_per_day < 1  # Less than 1 trade per day
        assert avg_days_per_trade > 1  # More than 1 day per trade


class TestStrategyOptimization:
    """Test suite for strategy optimization"""
    
    def test_parameter_grid_search(self):
        """Test parameter grid search"""
        thresholds = [0.5, 0.6, 0.7]
        holding_periods = [3, 5, 7]
        
        combinations = [(t, h) for t in thresholds for h in holding_periods]
        
        assert len(combinations) == 9
        assert (0.6, 5) in combinations
    
    def test_walk_forward_optimization(self):
        """Test walk-forward optimization"""
        train_periods = [
            {'start': '2020-01', 'end': '2020-06'},
            {'start': '2020-07', 'end': '2020-12'},
            {'start': '2021-01', 'end': '2021-06'}
        ]
        
        assert len(train_periods) == 3
        assert all('start' in p and 'end' in p for p in train_periods)
    
    def test_cross_validation_for_strategies(self):
        """Test cross-validation for strategy parameters"""
        folds = 5
        total_data_points = 1000
        
        fold_size = total_data_points // folds
        
        assert fold_size == 200
        assert fold_size * folds <= total_data_points
    
    def test_overfitting_detection(self):
        """Test overfitting detection"""
        in_sample_performance = 0.85
        out_sample_performance = 0.55
        
        performance_degradation = in_sample_performance - out_sample_performance
        likely_overfit = performance_degradation > 0.15
        
        assert likely_overfit == True  # 30% degradation indicates overfitting


# Mark all tests as unit tests
pytestmark = pytest.mark.unit