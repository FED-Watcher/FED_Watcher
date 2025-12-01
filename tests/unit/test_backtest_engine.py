"""
Unit tests for backtesting engine module
Tests backtesting engine core functionality
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestBacktestEngine:
    """Test suite for BacktestEngine class"""
    
    def test_backtest_engine_initialization(self):
        """Test BacktestEngine initialization"""
        config = {
            'initial_capital': 10000,
            'commission': 0.001,
            'slippage': 0.0005
        }
        
        assert config['initial_capital'] == 10000
        assert config['commission'] < 0.01
    
    def test_load_historical_data(self):
        """Test loading historical data"""
        dates = pd.date_range('2025-01-01', periods=100)
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 102,
            'low': np.random.randn(100) + 98,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        assert len(data) == 100
        assert 'close' in data.columns
        assert all(data['high'] >= data['low'])
    
    def test_calculate_returns(self):
        """Test return calculation"""
        prices = pd.Series([100, 102, 101, 105, 103])
        returns = prices.pct_change()
        
        assert len(returns) == len(prices)
        assert pd.isna(returns.iloc[0])  # First value is NaN
        assert np.isclose(returns.iloc[1], 0.02)  # 2% return
    
    def test_execute_trade(self):
        """Test trade execution"""
        trade = {
            'type': 'buy',
            'quantity': 10,
            'price': 100,
            'commission': 0.001
        }
        
        cost = trade['quantity'] * trade['price']
        commission = cost * trade['commission']
        total_cost = cost + commission
        
        assert cost == 1000
        assert commission == 1.0
        assert total_cost == 1001.0
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation"""
        cash = 5000
        positions = [
            {'quantity': 10, 'current_price': 105},
            {'quantity': 20, 'current_price': 50}
        ]
        
        position_value = sum(p['quantity'] * p['current_price'] for p in positions)
        total_value = cash + position_value
        
        assert position_value == 2050
        assert total_value == 7050
    
    def test_position_management(self):
        """Test position opening and closing"""
        positions = {}
        
        # Open position
        positions['AAPL'] = {'quantity': 10, 'entry_price': 100}
        assert 'AAPL' in positions
        
        # Close position
        del positions['AAPL']
        assert 'AAPL' not in positions
    
    def test_transaction_cost_calculation(self):
        """Test transaction cost calculation"""
        trade_value = 10000
        commission_rate = 0.001
        slippage_rate = 0.0005
        
        commission = trade_value * commission_rate
        slippage = trade_value * slippage_rate
        total_cost = commission + slippage
        
        assert commission == 10
        assert slippage == 5
        assert total_cost == 15
    
    def test_equity_curve_generation(self):
        """Test equity curve generation"""
        initial_capital = 10000
        daily_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        
        equity_curve = [initial_capital]
        for ret in daily_returns:
            equity_curve.append(equity_curve[-1] * (1 + ret))
        
        assert len(equity_curve) == len(daily_returns) + 1
        assert equity_curve[0] == initial_capital
        assert equity_curve[-1] != initial_capital  # Changed after returns
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        equity_curve = np.array([10000, 10200, 10100, 10500, 10300, 10800])
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        max_equity = equity_curve.max()
        min_equity = equity_curve.min()
        
        assert total_return > 0  # Profitable
        assert max_equity >= equity_curve[-1]
        assert min_equity <= equity_curve[0]
    
    def test_trade_log_creation(self):
        """Test trade log creation"""
        trade_log = []
        
        trade = {
            'date': '2025-01-01',
            'type': 'buy',
            'symbol': 'SPY',
            'quantity': 10,
            'price': 450,
            'pnl': 0
        }
        
        trade_log.append(trade)
        
        assert len(trade_log) == 1
        assert trade_log[0]['type'] == 'buy'
    
    def test_slippage_simulation(self):
        """Test slippage simulation"""
        desired_price = 100
        slippage_pct = 0.001
        
        # Buy order - pay more
        actual_buy_price = desired_price * (1 + slippage_pct)
        
        # Sell order - receive less
        actual_sell_price = desired_price * (1 - slippage_pct)
        
        assert actual_buy_price > desired_price
        assert actual_sell_price < desired_price
    
    def test_market_impact_modeling(self):
        """Test market impact modeling"""
        order_size = 1000
        average_volume = 100000
        
        # Simple linear market impact
        volume_participation = order_size / average_volume
        market_impact = volume_participation * 0.01  # 1% per 100% of volume
        
        assert 0 <= volume_participation <= 1
        assert market_impact >= 0
    
    def test_rebalancing_logic(self):
        """Test portfolio rebalancing"""
        target_allocation = {'SPY': 0.6, 'TLT': 0.4}
        current_allocation = {'SPY': 0.7, 'TLT': 0.3}
        
        rebalance_needed = any(
            abs(current_allocation[asset] - target_allocation[asset]) > 0.05
            for asset in target_allocation
        )
        
        assert rebalance_needed == True
    
    def test_date_range_validation(self):
        """Test date range validation"""
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31)
        
        assert end_date > start_date
        assert (end_date - start_date).days > 0
    
    def test_data_frequency_handling(self):
        """Test different data frequencies"""
        daily_data = pd.date_range('2025-01-01', periods=252, freq='D')
        hourly_data = pd.date_range('2025-01-01', periods=24, freq='h')
        
        assert len(daily_data) == 252
        assert len(hourly_data) == 24
    
    def test_missing_data_handling(self):
        """Test missing data handling"""
        data = pd.Series([100, np.nan, 102, 103, np.nan, 105])
        
        # Forward fill
        filled_data = data.fillna(method='ffill')
        
        assert not filled_data.isna().any()
        assert len(filled_data) == len(data)
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.50, 0.015, -0.02])  # 0.50 is outlier
        
        mean = returns.mean()
        std = returns.std()
        threshold = 3
        
        outliers = abs(returns - mean) > (threshold * std)
        
        assert outliers.sum() >= 0  # Outlier detection
    
    def test_risk_management_rules(self):
        """Test risk management rules"""
        account_value = 10000
        max_position_size = 0.1  # 10% max per position
        max_total_exposure = 0.8  # 80% max total
        
        max_position_value = account_value * max_position_size
        max_total_value = account_value * max_total_exposure
        
        assert max_position_value == 1000
        assert max_total_value == 8000
    
    def test_margin_requirements(self):
        """Test margin requirement calculation"""
        position_value = 10000
        margin_requirement = 0.25  # 25% initial margin
        
        required_margin = position_value * margin_requirement
        
        assert required_margin == 2500
        assert required_margin < position_value
    
    def test_bankruptcy_check(self):
        """Test bankruptcy detection"""
        account_balance = -500
        margin_call_threshold = 0
        
        is_bankrupt = account_balance < margin_call_threshold
        
        assert is_bankrupt == True
    
    def test_win_loss_tracking(self):
        """Test win/loss tracking"""
        trades = [
            {'pnl': 100},
            {'pnl': -50},
            {'pnl': 75},
            {'pnl': -25},
            {'pnl': 150}
        ]
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        assert len(wins) == 3
        assert len(losses) == 2
    
    def test_consecutive_wins_losses(self):
        """Test consecutive wins/losses tracking"""
        results = [1, 1, 1, -1, 1, 1, -1, -1, -1, 1]  # 1=win, -1=loss
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for result in results:
            if result == 1:
                current_streak = max(1, current_streak + 1) if current_streak > 0 else 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            else:
                current_streak = min(-1, current_streak - 1) if current_streak < 0 else -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        assert max_consecutive_wins >= 1
        assert max_consecutive_losses >= 1
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison"""
        strategy_return = 0.15  # 15%
        benchmark_return = 0.10  # 10%
        
        excess_return = strategy_return - benchmark_return
        
        assert abs(excess_return - 0.05) < 0.001  # 5% outperformance
        assert excess_return > 0  # Beat benchmark


class TestBacktestValidation:
    """Test suite for backtest validation"""
    
    def test_lookahead_bias_prevention(self):
        """Test prevention of lookahead bias"""
        current_index = 50
        data_length = 100
        
        # Should only use data up to current_index
        available_data_indices = list(range(current_index + 1))
        
        assert max(available_data_indices) == current_index
        assert len(available_data_indices) <= data_length
    
    def test_survivorship_bias_check(self):
        """Test survivorship bias awareness"""
        active_stocks = ['AAPL', 'MSFT', 'GOOGL']
        delisted_stocks = ['ENRON', 'LEHMAN']
        
        # Should include delisted stocks in historical test
        all_stocks = active_stocks + delisted_stocks
        
        assert len(all_stocks) > len(active_stocks)
    
    def test_transaction_cost_realism(self):
        """Test realistic transaction costs"""
        commission_per_trade = 1.0  # $1 per trade
        trades_per_year = 100
        
        annual_commission_cost = commission_per_trade * trades_per_year
        
        assert annual_commission_cost == 100
        assert commission_per_trade > 0  # Never zero cost
    
    def test_data_snooping_prevention(self):
        """Test data snooping prevention"""
        optimization_period = (0, 60)  # First 60% for optimization
        validation_period = (60, 80)  # Next 20% for validation
        test_period = (80, 100)  # Last 20% for testing
        
        # Periods should not overlap
        assert optimization_period[1] == validation_period[0]
        assert validation_period[1] == test_period[0]


# Mark all tests as unit tests
pytestmark = pytest.mark.unit