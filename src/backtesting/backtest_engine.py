"""
Event-driven backtesting engine for FED Watcher trading strategies.

The engine simulates a trading strategy by:
1. Iterating through test data chronologically
2. Using model predictions to determine positions
3. Calculating P&L based on actual market returns
4. Tracking portfolio equity over time
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class BacktestEngine:
    """
    Event-driven backtesting engine for trading strategies.

    Key Assumptions (explicitly stated):
    - Zero transaction costs (no commissions or fees)
    - No slippage (trade executed at exact closing price)
    - Full liquidity (can always enter/exit at market price)
    - No position size limits
    - Positions held for exactly horizon period (default 1 day)
    - Returns calculated using close-to-close prices
    """

    def __init__(
        self,
        predictor,
        strategy,
        initial_capital=100000,
        horizon=1,
        commission_pct=0.0,
        slippage_pct=0.0,
    ):
        """
        Initialize backtesting engine.

        Args:
            predictor (ModelPredictor): Trained model predictor
            strategy (Strategy): Trading strategy to backtest
            initial_capital (float): Starting portfolio value ($)
            horizon (int): Holding period in days (default 1 for 24h)
            commission_pct (float): Commission per trade as percentage (e.g., 0.1 for 0.1%)
            slippage_pct (float): Slippage per trade as percentage (e.g., 0.05 for 0.05%)
        """
        self.predictor = predictor
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.horizon = horizon
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # Results storage
        self.trades = []
        self.equity_curve = []
        self.metrics = {}

        print(f"\n{'='*60}")
        print(f"BACKTESTING ENGINE INITIALIZED")
        print(f"{'='*60}")
        print(f"Model Type: {self.predictor.model_type}")
        print(f"Strategy: {self.strategy.name}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Holding Period: {self.horizon} day(s)")
        print(f"Commission: {self.commission_pct}%")
        print(f"Slippage: {self.slippage_pct}%")
        print(f"{'='*60}\n")

    def run(self, test_data, date_col="Date", return_col="return", verbose=True):
        """
        Run backtest on test data.

        Args:
            test_data (pd.DataFrame): Test dataset with features and actual returns
            date_col (str): Name of date column (default 'Date')
            return_col (str): Name of actual return column (default 'return')
            verbose (bool): Print progress during backtest

        Returns:
            pd.DataFrame: DataFrame with trade-by-trade results
        """
        if verbose:
            print(f"Starting backtest on {len(test_data)} events...")
            print(f"{'='*60}\n")

        # Reset results
        self.trades = []
        self.equity_curve = []
        current_equity = self.initial_capital

        # Ensure we have a date index or column
        if date_col in test_data.columns:
            dates = test_data[date_col]
        elif test_data.index.name == date_col or isinstance(test_data.index, pd.DatetimeIndex):
            dates = test_data.index
        else:
            raise ValueError(f"Date column '{date_col}' not found in test data")

        # Get predictions for all test data
        predictions = self.predictor.predict(test_data)

        # Get probabilities if available
        try:
            probabilities = self.predictor.predict_proba(test_data)
        except (AttributeError, Exception):
            probabilities = None

        # Iterate through test set chronologically (event-driven)
        for idx, (date, pred) in enumerate(zip(dates, predictions)):
            # Get actual return for this event
            if return_col in test_data.columns:
                actual_return = test_data[return_col].iloc[idx]
            elif hasattr(test_data.index, "get_loc"):
                actual_return = test_data.loc[date, return_col]
            else:
                actual_return = test_data[return_col].iloc[idx]

            # Get probabilities for this event if available
            event_probs = probabilities[idx] if probabilities is not None else None

            # Determine position based on prediction and strategy
            position = self.strategy.get_position(pred, probabilities=event_probs)

            # Calculate P&L for this trade
            pnl_pct = self.strategy.calculate_pnl(position, actual_return * 100)  # Convert to %

            # Apply transaction costs (commission + slippage)
            # Costs apply only when position != 0 (actual trade executed)
            if position != 0:
                transaction_cost_pct = self.commission_pct + self.slippage_pct
                pnl_pct -= transaction_cost_pct

            pnl_dollars = current_equity * (pnl_pct / 100)

            # Update equity
            current_equity += pnl_dollars

            # Get prediction label
            pred_label = self.predictor.get_prediction_label(pred)

            # Store trade details
            trade = {
                "date": date,
                "prediction": pred,
                "prediction_label": pred_label,
                "position": position,
                "actual_return_pct": actual_return * 100,
                "pnl_pct": pnl_pct,
                "pnl_dollars": pnl_dollars,
                "equity": current_equity,
            }

            if event_probs is not None:
                trade["max_probability"] = np.max(event_probs)

            self.trades.append(trade)

            # Store equity point
            self.equity_curve.append({"date": date, "equity": current_equity})

            if verbose and (idx + 1) % 5 == 0:
                print(
                    f"  Event {idx + 1}/{len(test_data)}: "
                    f"{date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date} | "
                    f"Pred: {pred_label:12s} | "
                    f"Position: {position:+.2f} | "
                    f"P&L: ${pnl_dollars:+,.2f} | "
                    f"Equity: ${current_equity:,.2f}"
                )

        # Convert to DataFrames
        self.trades_df = pd.DataFrame(self.trades)
        self.equity_df = pd.DataFrame(self.equity_curve)

        # Calculate metrics
        self._calculate_metrics()

        if verbose:
            print(f"\n{'='*60}")
            print(f"BACKTEST COMPLETE")
            print(f"{'='*60}\n")
            self.print_summary()

        return self.trades_df

    def _calculate_metrics(self):
        """Calculate performance metrics from backtest results."""
        trades_df = self.trades_df

        # Basic stats
        total_trades = len(trades_df)
        total_return = (self.equity_df["equity"].iloc[-1] / self.initial_capital - 1) * 100

        # Win/loss statistics
        winning_trades = trades_df[trades_df["pnl_dollars"] > 0]
        losing_trades = trades_df[trades_df["pnl_dollars"] < 0]
        neutral_trades = trades_df[trades_df["pnl_dollars"] == 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0

        avg_win = winning_trades["pnl_pct"].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades["pnl_pct"].mean() if len(losing_trades) > 0 else 0

        # Risk metrics
        returns_series = trades_df["pnl_pct"]
        sharpe_ratio = self._calculate_sharpe(returns_series)
        sortino_ratio = self._calculate_sortino(returns_series)
        max_drawdown = self._calculate_max_drawdown(self.equity_df["equity"])
        calmar_ratio = self._calculate_calmar(total_return, max_drawdown)

        # Profit factor
        total_wins = winning_trades["pnl_dollars"].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades["pnl_dollars"].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

        # Win/Loss streaks
        win_streak, loss_streak = self._calculate_streaks(trades_df)

        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Recovery factor
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else np.inf

        self.metrics = {
            "total_trades": total_trades,
            "total_return_pct": total_return,
            "annualized_return_pct": self._annualize_return(total_return, len(trades_df)),
            "final_equity": self.equity_df["equity"].iloc[-1],
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "neutral_trades": len(neutral_trades),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "best_trade_pct": trades_df["pnl_pct"].max(),
            "worst_trade_pct": trades_df["pnl_pct"].min(),
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_drawdown,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "recovery_factor": recovery_factor,
            "max_win_streak": win_streak,
            "max_loss_streak": loss_streak,
            "total_pnl": self.equity_df["equity"].iloc[-1] - self.initial_capital,
            "total_commission_cost": self.commission_pct
            * len(trades_df[trades_df["position"] != 0]),
        }

    def _calculate_sortino(self, returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculate Sortino ratio (uses downside deviation only).

        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Risk-free rate (annualized)
            periods_per_year (int): Trading periods per year

        Returns:
            float: Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
        return sortino

    def _calculate_calmar(self, total_return, max_drawdown):
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            total_return (float): Total return percentage
            max_drawdown (float): Maximum drawdown percentage

        Returns:
            float: Calmar ratio
        """
        if max_drawdown == 0 or max_drawdown >= 0:
            return 0.0
        return abs(total_return / max_drawdown)

    def _calculate_streaks(self, trades_df):
        """
        Calculate maximum winning and losing streaks.

        Args:
            trades_df (pd.DataFrame): DataFrame of trades

        Returns:
            tuple: (max_win_streak, max_loss_streak)
        """
        if len(trades_df) == 0:
            return 0, 0

        is_win = (trades_df["pnl_dollars"] > 0).astype(int)
        is_loss = (trades_df["pnl_dollars"] < 0).astype(int)

        # Calculate streaks
        win_streaks = is_win * (is_win.groupby((is_win != is_win.shift()).cumsum()).cumcount() + 1)
        loss_streaks = is_loss * (
            is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1
        )

        max_win_streak = win_streaks.max() if len(win_streaks) > 0 else 0
        max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0

        return int(max_win_streak), int(max_loss_streak)

    def _annualize_return(self, total_return_pct, num_trades, trades_per_year=12):
        """
        Annualize the return based on number of trades.

        Args:
            total_return_pct (float): Total return percentage
            num_trades (int): Number of trades executed
            trades_per_year (int): Estimated trades per year (default 12 for FOMC)

        Returns:
            float: Annualized return percentage
        """
        if num_trades == 0:
            return 0.0

        years = num_trades / trades_per_year
        if years <= 0:
            return 0.0

        annualized = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
        return annualized

    def _calculate_sharpe(self, returns, risk_free_rate=0.0, periods_per_year=252):
        """
        Calculate Sharpe ratio.

        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Risk-free rate (annualized)
            periods_per_year (int): Trading periods per year (252 for daily)

        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return sharpe

    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown.

        Args:
            equity_curve (pd.Series): Series of equity values

        Returns:
            float: Maximum drawdown as percentage
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = ((equity_curve - running_max) / running_max) * 100

        # Return maximum drawdown (most negative value)
        max_dd = drawdown.min()
        return max_dd

    def print_summary(self):
        """Print summary statistics of backtest."""
        m = self.metrics

        print("PERFORMANCE SUMMARY")
        print(f"{'-'*60}")
        print(f"Total Trades:          {m['total_trades']}")
        print(f"Winning Trades:        {m['winning_trades']} ({m['win_rate']*100:.1f}%)")
        print(f"Losing Trades:         {m['losing_trades']} ({m['loss_rate']*100:.1f}%)")
        print(f"Neutral Trades:        {m['neutral_trades']}")
        print(f"Max Win Streak:        {m['max_win_streak']}")
        print(f"Max Loss Streak:       {m['max_loss_streak']}")
        print("\nRETURNS")
        print(f"{'-'*60}")
        print(f"Initial Capital:       ${self.initial_capital:,.2f}")
        print(f"Final Equity:          ${m['final_equity']:,.2f}")
        print(f"Total P&L:             ${m['total_pnl']:+,.2f}")
        print(f"Total Return:          {m['total_return_pct']:+.2f}%")
        print(f"Annualized Return:     {m['annualized_return_pct']:+.2f}%")
        print("\nPER-TRADE STATISTICS")
        print(f"{'-'*60}")
        print(f"Average Win:           {m['avg_win_pct']:+.2f}%")
        print(f"Average Loss:          {m['avg_loss_pct']:+.2f}%")
        print(f"Best Trade:            {m['best_trade_pct']:+.2f}%")
        print(f"Worst Trade:           {m['worst_trade_pct']:+.2f}%")
        print(f"Expectancy:            {m['expectancy']:+.2f}%")
        print(f"Profit Factor:         {m['profit_factor']:.2f}")
        print("\nRISK METRICS")
        print(f"{'-'*60}")
        print(f"Sharpe Ratio:          {m['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:         {m['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:          {m['calmar_ratio']:.2f}")
        print(f"Max Drawdown:          {m['max_drawdown_pct']:.2f}%")
        print(f"Recovery Factor:       {m['recovery_factor']:.2f}")
        print("\nCOSTS")
        print(f"{'-'*60}")
        print(f"Commission Rate:       {self.commission_pct}%")
        print(f"Slippage Rate:         {self.slippage_pct}%")
        print(f"Total Commission:      {m['total_commission_cost']:.2f}%")
        print(f"{'='*60}\n")

    def plot_equity_curve(self, save_path=None, show_trades=True):
        """
        Plot equity curve over time.

        Args:
            save_path (str): Path to save plot (optional)
            show_trades (bool): Show individual trade markers
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot equity curve
        equity_df = self.equity_df.copy()
        if isinstance(equity_df["date"].iloc[0], str):
            equity_df["date"] = pd.to_datetime(equity_df["date"])

        ax.plot(
            equity_df["date"],
            equity_df["equity"],
            linewidth=2.5,
            color="steelblue",
            label="Portfolio Equity",
        )

        # Add horizontal line at initial capital
        ax.axhline(
            y=self.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital"
        )

        # Mark winning and losing trades
        if show_trades:
            trades_df = self.trades_df.copy()
            if isinstance(trades_df["date"].iloc[0], str):
                trades_df["date"] = pd.to_datetime(trades_df["date"])

            wins = trades_df[trades_df["pnl_dollars"] > 0]
            losses = trades_df[trades_df["pnl_dollars"] < 0]

            ax.scatter(
                wins["date"],
                wins["equity"],
                color="green",
                marker="^",
                s=100,
                alpha=0.6,
                label="Winning Trades",
                zorder=5,
            )
            ax.scatter(
                losses["date"],
                losses["equity"],
                color="red",
                marker="v",
                s=100,
                alpha=0.6,
                label="Losing Trades",
                zorder=5,
            )

        # Formatting
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Backtesting Results: {self.strategy.name}\n"
            f'Total Return: {self.metrics["total_return_pct"]:+.2f}% | '
            f'Sharpe: {self.metrics["sharpe_ratio"]:.2f} | '
            f'Max DD: {self.metrics["max_drawdown_pct"]:.2f}%',
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Equity curve saved to: {save_path}")

        plt.show()

    def plot_returns_distribution(self, save_path=None):
        """
        Plot distribution of trade returns.

        Args:
            save_path (str): Path to save plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        returns = self.trades_df["pnl_pct"]

        # Histogram
        ax1.hist(returns, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax1.axvline(
            x=returns.mean(),
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean: {returns.mean():.2f}%",
        )
        ax1.set_xlabel("Return (%)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax1.set_title("Distribution of Trade Returns", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot by prediction
        trades_df = self.trades_df.copy()
        sns.boxplot(data=trades_df, x="prediction_label", y="pnl_pct", ax=ax2, palette="Set2")
        ax2.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax2.set_xlabel("Prediction", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Return (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Returns by Prediction Type", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Returns distribution saved to: {save_path}")

        plt.show()

    def plot_drawdown(self, save_path=None):
        """
        Plot underwater (drawdown) chart.

        Args:
            save_path (str): Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        equity_df = self.equity_df.copy()
        if isinstance(equity_df["date"].iloc[0], str):
            equity_df["date"] = pd.to_datetime(equity_df["date"])

        # Calculate drawdown
        running_max = equity_df["equity"].expanding().max()
        drawdown = ((equity_df["equity"] - running_max) / running_max) * 100

        # Plot
        ax.fill_between(equity_df["date"], drawdown, 0, color="red", alpha=0.3)
        ax.plot(equity_df["date"], drawdown, color="darkred", linewidth=2)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_date = equity_df.loc[max_dd_idx, "date"]
        max_dd_val = drawdown[max_dd_idx]
        ax.scatter([max_dd_date], [max_dd_val], color="red", s=150, zorder=5, marker="v")
        ax.annotate(
            f"Max DD: {max_dd_val:.2f}%",
            xy=(max_dd_date, max_dd_val),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        # Formatting
        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Underwater Plot: {self.strategy.name}", fontsize=14, fontweight="bold", pad=20
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Drawdown plot saved to: {save_path}")

        plt.show()

    def plot_monthly_returns(self, save_path=None):
        """
        Plot monthly returns heatmap.

        Args:
            save_path (str): Path to save plot (optional)
        """
        trades_df = self.trades_df.copy()
        if isinstance(trades_df["date"].iloc[0], str):
            trades_df["date"] = pd.to_datetime(trades_df["date"])

        # Extract year and month
        trades_df["year"] = trades_df["date"].dt.year
        trades_df["month"] = trades_df["date"].dt.month

        # Group by year and month
        monthly_returns = trades_df.groupby(["year", "month"])["pnl_pct"].sum().reset_index()

        # Pivot for heatmap
        pivot_data = monthly_returns.pivot(index="year", columns="month", values="pnl_pct")

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))

        # Custom colormap (red for negative, green for positive)
        cmap = sns.diverging_palette(10, 130, as_cmap=True)

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            cbar_kws={"label": "Return (%)"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax.set_ylabel("Year", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Monthly Returns Heatmap: {self.strategy.name}", fontsize=14, fontweight="bold", pad=20
        )

        # Month names - map to actual columns present
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        # Only set labels for months that exist in the data
        present_months = [month_names[i - 1] for i in pivot_data.columns if 1 <= i <= 12]
        ax.set_xticklabels(present_months, rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Monthly returns heatmap saved to: {save_path}")

        plt.show()

    def plot_trade_analysis(self, save_path=None):
        """
        Plot comprehensive trade analysis (4-panel).

        Args:
            save_path (str): Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        trades_df = self.trades_df.copy()
        if isinstance(trades_df["date"].iloc[0], str):
            trades_df["date"] = pd.to_datetime(trades_df["date"])

        # Panel 1: Cumulative P&L
        ax1 = axes[0, 0]
        cumulative_pnl = trades_df["pnl_dollars"].cumsum()
        ax1.plot(trades_df["date"], cumulative_pnl, linewidth=2.5, color="steelblue")
        ax1.fill_between(trades_df["date"], cumulative_pnl, 0, alpha=0.3, color="steelblue")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Date", fontweight="bold")
        ax1.set_ylabel("Cumulative P&L ($)", fontweight="bold")
        ax1.set_title("Cumulative Profit & Loss", fontweight="bold", pad=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Panel 2: Trade P&L by Position Type
        ax2 = axes[0, 1]
        long_trades = trades_df[trades_df["position"] > 0]["pnl_pct"]
        short_trades = trades_df[trades_df["position"] < 0]["pnl_pct"]
        flat_trades = trades_df[trades_df["position"] == 0]["pnl_pct"]

        box_data = []
        labels = []
        if len(long_trades) > 0:
            box_data.append(long_trades)
            labels.append(f"Long ({len(long_trades)})")
        if len(short_trades) > 0:
            box_data.append(short_trades)
            labels.append(f"Short ({len(short_trades)})")
        if len(flat_trades) > 0:
            box_data.append(flat_trades)
            labels.append(f"Flat ({len(flat_trades)})")

        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True, meanline=True)
        for patch, color in zip(bp["boxes"], ["green", "red", "gray"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax2.set_ylabel("Return (%)", fontweight="bold")
        ax2.set_title("Returns by Position Type", fontweight="bold", pad=10)
        ax2.grid(True, alpha=0.3, axis="y")

        # Panel 3: Rolling Win Rate
        ax3 = axes[1, 0]
        window = min(5, len(trades_df))
        trades_df["is_win"] = (trades_df["pnl_dollars"] > 0).astype(int)
        rolling_win_rate = trades_df["is_win"].rolling(window=window, min_periods=1).mean() * 100

        ax3.plot(trades_df["date"], rolling_win_rate, linewidth=2, color="purple")
        ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% Break-even")
        ax3.axhline(
            y=self.metrics["win_rate"] * 100,
            color="green",
            linestyle="-",
            alpha=0.7,
            label=f'Avg: {self.metrics["win_rate"]*100:.1f}%',
        )
        ax3.fill_between(
            trades_df["date"],
            rolling_win_rate,
            50,
            alpha=0.3,
            where=(rolling_win_rate >= 50),
            color="green",
            interpolate=True,
        )
        ax3.fill_between(
            trades_df["date"],
            rolling_win_rate,
            50,
            alpha=0.3,
            where=(rolling_win_rate < 50),
            color="red",
            interpolate=True,
        )
        ax3.set_xlabel("Date", fontweight="bold")
        ax3.set_ylabel("Win Rate (%)", fontweight="bold")
        ax3.set_title(f"Rolling Win Rate ({window}-Trade Window)", fontweight="bold", pad=10)
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        # Panel 4: Trade Size Distribution
        ax4 = axes[1, 1]
        ax4.scatter(
            range(len(trades_df)),
            trades_df["pnl_pct"],
            c=trades_df["pnl_pct"],
            cmap="RdYlGn",
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax4.set_xlabel("Trade Number", fontweight="bold")
        ax4.set_ylabel("P&L (%)", fontweight="bold")
        ax4.set_title("Individual Trade Performance", fontweight="bold", pad=10)
        ax4.grid(True, alpha=0.3)

        plt.suptitle(
            f"Trade Analysis: {self.strategy.name}", fontsize=16, fontweight="bold", y=0.995
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[OK] Trade analysis plot saved to: {save_path}")

        plt.show()

    def export_results(self, output_dir="backtesting_results"):
        """
        Export backtest results to CSV files.

        Args:
            output_dir (str): Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save trades
        trades_file = output_path / "trades.csv"
        self.trades_df.to_csv(trades_file, index=False)
        print(f"[OK] Trades saved to: {trades_file}")

        # Save equity curve
        equity_file = output_path / "equity_curve.csv"
        self.equity_df.to_csv(equity_file, index=False)
        print(f"[OK] Equity curve saved to: {equity_file}")

        # Save metrics
        import json

        metrics_file = output_path / "metrics.json"
        with open(metrics_file, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            metrics_json = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in self.metrics.items()
            }
            json.dump(metrics_json, f, indent=4)
        print(f"[OK] Metrics saved to: {metrics_file}")
