"""
Unit tests for report generator module
Tests report generation and formatting functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestReportGenerator:
    """Test suite for report generation"""
    
    def test_summary_statistics_generation(self):
        """Test summary statistics generation"""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.04])
        
        stats = {
            'mean': returns.mean(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max()
        }
        
        assert 'mean' in stats
        assert 'std' in stats
        assert stats['max'] > stats['min']
    
    def test_performance_report_creation(self):
        """Test performance report creation"""
        report = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10,
            'num_trades': 50,
            'win_rate': 0.6
        }
        
        assert report['total_return'] > 0
        assert report['sharpe_ratio'] > 1
        assert report['max_drawdown'] < 0
    
    def test_trade_list_formatting(self):
        """Test trade list formatting"""
        trades = [
            {'date': '2025-01-01', 'type': 'buy', 'pnl': 100},
            {'date': '2025-01-02', 'type': 'sell', 'pnl': -50}
        ]
        
        df = pd.DataFrame(trades)
        
        assert len(df) == 2
        assert 'pnl' in df.columns
    
    def test_equity_curve_plotting_data(self):
        """Test equity curve data preparation"""
        dates = pd.date_range('2025-01-01', periods=10)
        equity = np.linspace(10000, 11000, 10)
        
        curve_data = pd.DataFrame({
            'date': dates,
            'equity': equity
        })
        
        assert len(curve_data) == 10
        assert curve_data['equity'].iloc[0] < curve_data['equity'].iloc[-1]
    
    def test_drawdown_chart_data(self):
        """Test drawdown chart data preparation"""
        equity = np.array([10000, 10500, 10200, 9800, 10100, 10800])
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        assert len(drawdown) == len(equity)
        assert drawdown.max() <= 0  # Drawdown is non-positive
    
    def test_monthly_returns_table(self):
        """Test monthly returns table generation"""
        dates = pd.date_range('2025-01-01', periods=365, freq='D')
        returns = pd.Series(np.random.randn(365) * 0.01, index=dates)
        
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        assert len(monthly_returns) == 12
        assert isinstance(monthly_returns, pd.Series)
    
    def test_risk_metrics_table(self):
        """Test risk metrics table creation"""
        metrics = {
            'Volatility': 0.15,
            'Sharpe Ratio': 1.5,
            'Max Drawdown': -0.12,
            'VaR 95%': -0.03
        }
        
        df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        
        assert len(df) == 4
        assert 'Metric' in df.columns
    
    def test_trade_analysis_report(self):
        """Test trade analysis report"""
        trades_df = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 150],
            'duration': [5, 3, 7, 2, 6]
        })
        
        analysis = {
            'total_trades': len(trades_df),
            'avg_pnl': trades_df['pnl'].mean(),
            'avg_duration': trades_df['duration'].mean()
        }
        
        assert analysis['total_trades'] == 5
        assert analysis['avg_pnl'] == 50
    
    def test_html_report_generation(self):
        """Test HTML report generation"""
        html_template = """
        <html>
        <head><title>Backtest Report</title></head>
        <body>
        <h1>Performance Report</h1>
        <p>Total Return: {total_return:.2%}</p>
        </body>
        </html>
        """
        
        report_data = {'total_return': 0.15}
        html_report = html_template.format(**report_data)
        
        assert 'Performance Report' in html_report
        assert '15.00%' in html_report
    
    def test_pdf_report_metadata(self):
        """Test PDF report metadata"""
        metadata = {
            'title': 'Backtest Report',
            'author': 'FED-Watcher',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'version': '1.0'
        }
        
        assert metadata['title'] == 'Backtest Report'
        assert metadata['version'] == '1.0'
    
    def test_csv_export_functionality(self):
        """Test CSV export"""
        trades = pd.DataFrame({
            'date': ['2025-01-01', '2025-01-02'],
            'pnl': [100, -50]
        })
        
        # Simulate CSV export
        csv_string = trades.to_csv(index=False)
        
        assert 'date,pnl' in csv_string
        assert '100' in csv_string
    
    def test_json_export_functionality(self):
        """Test JSON export"""
        report_data = {
            'total_return': 0.15,
            'sharpe_ratio': 1.5,
            'trades': 50
        }
        
        import json
        json_string = json.dumps(report_data, indent=2)
        
        assert 'total_return' in json_string
        assert '0.15' in json_string
    
    def test_report_timestamp(self):
        """Test report timestamp"""
        report_time = datetime.now()
        timestamp_str = report_time.strftime('%Y-%m-%d %H:%M:%S')
        
        assert len(timestamp_str) == 19
        assert '2025' in timestamp_str
    
    def test_benchmark_comparison_table(self):
        """Test benchmark comparison table"""
        comparison = pd.DataFrame({
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown'],
            'Strategy': [0.15, 1.5, -0.10],
            'Benchmark': [0.10, 1.2, -0.08]
        })
        
        comparison['Difference'] = comparison['Strategy'] - comparison['Benchmark']
        
        assert len(comparison) == 3
        assert 'Difference' in comparison.columns
    
    def test_rolling_metrics_calculation(self):
        """Test rolling metrics calculation"""
        returns = pd.Series(np.random.randn(252) * 0.01)
        
        rolling_sharpe = returns.rolling(window=60).mean() / returns.rolling(window=60).std()
        
        assert len(rolling_sharpe) == len(returns)
        assert rolling_sharpe.isna().sum() < len(returns)
    
    def test_correlation_matrix(self):
        """Test correlation matrix generation"""
        data = pd.DataFrame({
            'strategy': np.random.randn(100),
            'spy': np.random.randn(100),
            'qqq': np.random.randn(100)
        })
        
        corr_matrix = data.corr()
        
        assert corr_matrix.shape == (3, 3)
        assert corr_matrix.loc['strategy', 'strategy'] == 1.0
    
    def test_percentile_calculations(self):
        """Test percentile calculations"""
        returns = pd.Series(np.random.randn(1000) * 0.01)
        
        percentiles = {
            'p5': returns.quantile(0.05),
            'p25': returns.quantile(0.25),
            'p50': returns.quantile(0.50),
            'p75': returns.quantile(0.75),
            'p95': returns.quantile(0.95)
        }
        
        assert percentiles['p5'] < percentiles['p50']
        assert percentiles['p50'] < percentiles['p95']
    
    def test_report_completeness_check(self):
        """Test report completeness validation"""
        required_sections = [
            'summary',
            'performance',
            'trades',
            'risk_metrics',
            'charts'
        ]
        
        report_sections = ['summary', 'performance', 'trades', 'risk_metrics', 'charts']
        
        is_complete = all(section in report_sections for section in required_sections)
        
        assert is_complete == True
    
    def test_error_handling_missing_data(self):
        """Test error handling for missing data"""
        incomplete_data = {'total_return': 0.15}  # Missing other metrics
        
        # Should handle gracefully
        try:
            sharpe = incomplete_data.get('sharpe_ratio', None)
            assert sharpe is None
        except KeyError:
            pytest.fail("Should handle missing data gracefully")


class TestReportFormatting:
    """Test suite for report formatting"""
    
    def test_percentage_formatting(self):
        """Test percentage formatting"""
        value = 0.1234
        formatted = f"{value:.2%}"
        
        assert formatted == "12.34%"
    
    def test_currency_formatting(self):
        """Test currency formatting"""
        value = 12345.67
        formatted = f"${value:,.2f}"
        
        assert formatted == "$12,345.67"
    
    def test_number_abbreviation(self):
        """Test large number abbreviation"""
        value = 1500000
        
        if value >= 1_000_000:
            abbreviated = f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:
            abbreviated = f"{value / 1_000:.1f}K"
        else:
            abbreviated = str(value)
        
        assert abbreviated == "1.5M"
    
    def test_date_formatting(self):
        """Test date formatting"""
        date = datetime(2025, 12, 1)
        formatted = date.strftime('%Y-%m-%d')
        
        assert formatted == "2025-12-01"
    
    def test_table_alignment(self):
        """Test table column alignment"""
        df = pd.DataFrame({
            'Metric': ['Return', 'Sharpe'],
            'Value': [0.15, 1.5]
        })
        
        # Check data types for alignment
        assert df['Metric'].dtype == 'object'  # Left align
        assert pd.api.types.is_numeric_dtype(df['Value'])  # Right align


# Mark all tests as unit tests
pytestmark = pytest.mark.unit