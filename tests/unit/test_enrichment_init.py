"""Unit tests for enrichment package __init__.py"""

import pytest
import sys
from unittest.mock import Mock, patch


class TestEnrichmentInit:
    """Test enrichment package initialization"""

    def test_all_exports(self):
        """Test __all__ contains expected exports"""
        import src.enrichment

        assert hasattr(src.enrichment, "__all__")
        assert "enrich_dataset" in src.enrichment.__all__

    def test_lazy_import_enrich_dataset(self):
        """Test lazy import of enrich_dataset"""
        import src.enrichment

        # Access the function through __getattr__
        func = src.enrichment.enrich_dataset

        assert callable(func)

    def test_getattr_invalid_name(self):
        """Test __getattr__ raises AttributeError for invalid names"""
        import src.enrichment

        with pytest.raises(AttributeError, match="has no attribute 'invalid_function'"):
            _ = src.enrichment.invalid_function


pytestmark = pytest.mark.unit
