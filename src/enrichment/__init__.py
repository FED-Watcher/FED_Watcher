"""Enrichment module for adding market context data to the master dataset.

This module enriches the master dataset with additional market indicators:
- VIX (Volatility Index)
- DXY (US Dollar Index)
- US02Y (2-Year Treasury Yield)
- US10Y (10-Year Treasury Yield)
- Yield Curve (10Y-2Y spread)
"""

__all__ = ["enrich_dataset"]


# Lazy import to avoid import errors during testing
def __getattr__(name):
    if name == "enrich_dataset":
        from src.enrichment.enrich_with_market_data import enrich_dataset

        return enrich_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
