"""
Tests for XGBoostImpactModel and the updated ImpactCalculator.

These tests run without a real model file; they verify:
  - Graceful load failure when the pkl doesn't exist
  - Rule-based fallback behaviour when model is absent
  - Feature vector structure and completeness
  - Impact score always in [1.0, 10.0]
  - ImpactCalculator respects MODEL_IMPACT env var
  - Missing article fields are handled with safe defaults

A mock pkl is synthesised using a trivial sklearn DummyRegressor so we can
test the XGBoost happy-path without requiring a real trained model.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_model(feature_columns: list[str], predict_value: float = 6.0):
    """
    Return a (model_path, features_path) pair pointing to temp files that
    contain a sklearn DummyRegressor pkl and a matching features JSON.

    The DummyRegressor always returns `predict_value` regardless of input,
    which lets us test the loading and scoring plumbing without XGBoost.
    """
    from sklearn.dummy import DummyRegressor
    import numpy as np

    dummy = DummyRegressor(strategy="constant", constant=predict_value)
    dummy.fit(
        [[0.0] * len(feature_columns)],
        [predict_value],
    )

    tmp_dir = tempfile.mkdtemp()
    model_path = Path(tmp_dir) / "impact_xgb.pkl"
    features_path = Path(tmp_dir) / "impact_xgb_features.json"

    with open(model_path, "wb") as fh:
        pickle.dump(dummy, fh)

    features_meta = {
        "feature_columns": feature_columns,
        "event_types": [
            "Earnings", "M&A", "Regulatory", "Macro", "Geopolitical",
            "Credit", "Commodity", "Cyber", "Leadership", "Product",
        ],
        "sectors": [
            "Technology", "Communication Services", "Consumer Discretionary",
            "Financials", "Health Care", "Energy", "Industrials", "Consumer Staples",
        ],
        "n_train": 1000,
        "n_val": 200,
        "n_test": 200,
        "best_iteration": 150,
        "test_mae": 1.0,
        "test_rmse": 1.2,
        "test_spearman_rho": 0.6,
        "baseline_mae": 1.5,
        "baseline_spearman_rho": 0.4,
    }
    features_path.write_text(json.dumps(features_meta))

    return model_path, features_path


def _minimal_feature_columns() -> list[str]:
    """Return the expected feature column list."""
    from app.engine.xgboost_model import EVENT_TYPES, SECTORS

    cols = [
        "sentiment", "abs_sentiment", "tone_positive", "tone_negative",
        "source_count", "is_after_hours", "days_since_last",
    ]
    for et in EVENT_TYPES:
        cols.append(f"event_type_{et.lower().replace('&', 'and').replace(' ', '_')}")
    for sec in SECTORS:
        cols.append(f"sector_{sec.lower().replace(' ', '_')}")
    return cols


# ---------------------------------------------------------------------------
# XGBoostImpactModel tests
# ---------------------------------------------------------------------------


class TestXGBoostImpactModelLoad:
    def test_load_returns_false_when_pkl_missing(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        result = model.load(
            model_path=Path("/tmp/definitely_does_not_exist_impact.pkl"),
            features_path=Path("/tmp/definitely_does_not_exist_features.json"),
        )
        assert result is False, "Expected load() to return False for missing file"

    def test_load_does_not_raise_when_pkl_missing(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        # Must not raise any exception
        try:
            model.load(
                model_path=Path("/tmp/no_such_model.pkl"),
                features_path=Path("/tmp/no_such_features.json"),
            )
        except Exception as exc:
            pytest.fail(f"load() raised unexpectedly: {exc}")

    def test_load_returns_true_with_valid_files(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        cols = _minimal_feature_columns()
        model_path, features_path = _make_dummy_model(cols)

        model = XGBoostImpactModel()
        result = model.load(model_path=model_path, features_path=features_path)
        assert result is True

    def test_load_returns_false_when_features_missing(self):
        """Model file exists but features JSON is absent → load fails gracefully."""
        import tempfile, pickle
        from sklearn.dummy import DummyRegressor
        from app.engine.xgboost_model import XGBoostImpactModel

        tmp = Path(tempfile.mkdtemp())
        model_path = tmp / "model.pkl"
        with open(model_path, "wb") as fh:
            dummy = DummyRegressor()
            dummy.fit([[0]], [5])
            pickle.dump(dummy, fh)

        model = XGBoostImpactModel()
        result = model.load(
            model_path=model_path,
            features_path=tmp / "nonexistent_features.json",
        )
        assert result is False


class TestXGBoostImpactModelAvailability:
    def test_is_available_false_before_load(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        assert model.is_available() is False

    def test_is_available_false_after_failed_load(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        model.load(
            model_path=Path("/tmp/ghost.pkl"),
            features_path=Path("/tmp/ghost_features.json"),
        )
        assert model.is_available() is False

    def test_is_available_true_after_successful_load(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        cols = _minimal_feature_columns()
        model_path, features_path = _make_dummy_model(cols)

        model = XGBoostImpactModel()
        model.load(model_path=model_path, features_path=features_path)
        assert model.is_available() is True


class TestXGBoostImpactModelFeatureVector:
    def _get_all_required_keys(self) -> set[str]:
        from app.engine.xgboost_model import EVENT_TYPES, SECTORS

        required = {
            "sentiment", "abs_sentiment", "tone_positive", "tone_negative",
            "source_count", "is_after_hours", "days_since_last",
        }
        for et in EVENT_TYPES:
            required.add(f"event_type_{et.lower().replace('&', 'and').replace(' ', '_')}")
        for sec in SECTORS:
            required.add(f"sector_{sec.lower().replace(' ', '_')}")
        return required

    def test_get_feature_vector_has_all_required_keys(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        article = {
            "ticker_symbols": ["AAPL"],
            "published_at": "2024-01-15T14:00:00+00:00",
        }
        features = model.get_feature_vector(
            article_dict=article,
            sentiment=0.5,
            event_type="Earnings",
            source_count=2,
        )
        required = self._get_all_required_keys()
        missing = required - set(features.keys())
        assert not missing, f"Missing feature keys: {missing}"

    def test_get_feature_vector_handles_empty_article(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        # Completely empty article dict — must not raise
        features = model.get_feature_vector(
            article_dict={},
            sentiment=0.0,
            event_type="Macro",
            source_count=1,
        )
        required = self._get_all_required_keys()
        missing = required - set(features.keys())
        assert not missing, f"Missing feature keys with empty article: {missing}"

    def test_get_feature_vector_handles_none_published_at(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector(
            article_dict={"published_at": None, "ticker_symbols": []},
            sentiment=-0.3,
            event_type="Credit",
        )
        # is_after_hours should default to 0.0 when published_at is None
        assert features["is_after_hours"] == 0.0

    def test_event_type_one_hot_is_correct(self):
        from app.engine.xgboost_model import XGBoostImpactModel, EVENT_TYPES

        model = XGBoostImpactModel()
        for event_type in EVENT_TYPES:
            features = model.get_feature_vector({}, 0.0, event_type)
            col = f"event_type_{event_type.lower().replace('&', 'and').replace(' ', '_')}"
            assert features[col] == 1.0, f"{col} should be 1.0 for event_type={event_type}"
            # All other event type columns should be 0
            for other in EVENT_TYPES:
                if other == event_type:
                    continue
                other_col = f"event_type_{other.lower().replace('&', 'and').replace(' ', '_')}"
                assert features[other_col] == 0.0, f"{other_col} should be 0.0"

    def test_sector_one_hot_matches_ticker(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        # AAPL → Technology
        features = model.get_feature_vector(
            {"ticker_symbols": ["AAPL"]}, 0.0, "Earnings"
        )
        assert features["sector_technology"] == 1.0
        assert features["sector_financials"] == 0.0

    def test_abs_sentiment_always_nonnegative(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        for sent in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            features = model.get_feature_vector({}, sent, "Earnings")
            assert features["abs_sentiment"] >= 0.0

    def test_days_since_last_defaults_to_30(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector({}, 0.0, "Earnings")
        assert features["days_since_last"] == 30.0

    def test_days_since_last_capped_at_90(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector(
            {"days_since_last_event": 999.0}, 0.0, "Earnings"
        )
        assert features["days_since_last"] == 90.0


class TestXGBoostImpactModelPredict:
    def test_predict_falls_back_when_model_not_loaded(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector({}, 0.5, "Earnings")
        score = model.predict(features)
        # Must not raise and must be in valid range
        assert 1.0 <= score <= 10.0

    def test_predict_returns_value_in_range_with_loaded_model(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        cols = _minimal_feature_columns()
        model_path, features_path = _make_dummy_model(cols, predict_value=6.0)

        model = XGBoostImpactModel()
        model.load(model_path=model_path, features_path=features_path)

        features = model.get_feature_vector(
            {"ticker_symbols": ["MSFT"], "published_at": "2024-03-01T10:00:00Z"},
            sentiment=0.7,
            event_type="Earnings",
            source_count=3,
        )
        score = model.predict(features)
        assert 1.0 <= score <= 10.0

    def test_predict_clamps_extreme_model_outputs(self):
        """Even if the underlying model returns out-of-range values, predict() clamps them."""
        from app.engine.xgboost_model import XGBoostImpactModel

        cols = _minimal_feature_columns()
        # DummyRegressor returning extreme value
        model_path, features_path = _make_dummy_model(cols, predict_value=99.0)

        model = XGBoostImpactModel()
        model.load(model_path=model_path, features_path=features_path)

        features = model.get_feature_vector({}, 0.0, "Geopolitical")
        score = model.predict(features)
        assert score <= 10.0, f"Expected clamped to 10.0, got {score}"

    def test_predict_clamps_low_model_outputs(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        cols = _minimal_feature_columns()
        model_path, features_path = _make_dummy_model(cols, predict_value=-5.0)

        model = XGBoostImpactModel()
        model.load(model_path=model_path, features_path=features_path)

        features = model.get_feature_vector({}, 0.0, "Product")
        score = model.predict(features)
        assert score >= 1.0, f"Expected clamped to 1.0, got {score}"


# ---------------------------------------------------------------------------
# ImpactCalculator integration tests
# ---------------------------------------------------------------------------


class TestImpactCalculatorRulesBased:
    def test_uses_rule_based_by_default(self):
        """Without MODEL_IMPACT=xgboost, ImpactCalculator must use rule-based."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MODEL_IMPACT", None)
            from app.engine.impact_calculator import ImpactCalculator
            calc = ImpactCalculator()
            assert calc._use_xgb is False

    def test_uses_rule_based_when_env_is_rules(self):
        with patch.dict(os.environ, {"MODEL_IMPACT": "rules"}):
            from app.engine.impact_calculator import ImpactCalculator
            calc = ImpactCalculator()
            assert calc._use_xgb is False

    def test_rule_based_output_in_range(self):
        from app.engine.impact_calculator import ImpactCalculator, BASE_IMPACT

        with patch.dict(os.environ, {"MODEL_IMPACT": "rules"}):
            calc = ImpactCalculator()
            for event_type in BASE_IMPACT.keys():
                for sentiment in [-1.0, 0.0, 0.5, 1.0]:
                    for novelty in [0.1, 0.5, 1.0]:
                        impact, (low, high) = calc.calculate(
                            event_type=event_type,
                            sentiment=sentiment,
                            novelty=novelty,
                        )
                        assert 1.0 <= impact <= 10.0, (
                            f"impact={impact} out of range for {event_type}, s={sentiment}, n={novelty}"
                        )
                        assert 1.0 <= low <= impact <= high <= 10.0


class TestImpactCalculatorXGBoost:
    def test_loads_xgboost_when_model_exists(self):
        """ImpactCalculator loads the XGBoost model when MODEL_IMPACT=xgboost and file exists."""
        cols = _minimal_feature_columns()
        model_path, features_path = _make_dummy_model(cols, predict_value=7.0)

        # Patch the features path constant so ImpactCalculator finds it
        with patch.dict(os.environ, {"MODEL_IMPACT": "xgboost"}):
            import app.engine.impact_calculator as _ic_module
            import app.engine.xgboost_model as _xgb_module

            original_features_path = _ic_module._XGB_FEATURES_PATH
            _ic_module._XGB_FEATURES_PATH = features_path
            try:
                from app.engine.impact_calculator import ImpactCalculator
                calc = ImpactCalculator(model_path=model_path)
                assert calc._use_xgb is True
                assert calc._xgb is not None
                assert calc._xgb.is_available() is True
            finally:
                _ic_module._XGB_FEATURES_PATH = original_features_path

    def test_falls_back_to_rule_based_when_model_missing(self):
        """MODEL_IMPACT=xgboost but pkl doesn't exist → silently use rule-based."""
        with patch.dict(os.environ, {"MODEL_IMPACT": "xgboost"}):
            from app.engine.impact_calculator import ImpactCalculator
            calc = ImpactCalculator(model_path=Path("/tmp/no_such_model_xyz.pkl"))
            assert calc._use_xgb is False

    def test_xgboost_output_in_range(self):
        """XGBoost path also produces impact in [1.0, 10.0]."""
        cols = _minimal_feature_columns()
        model_path, features_path = _make_dummy_model(cols, predict_value=6.5)

        with patch.dict(os.environ, {"MODEL_IMPACT": "xgboost"}):
            import app.engine.impact_calculator as _ic_module

            original_features_path = _ic_module._XGB_FEATURES_PATH
            _ic_module._XGB_FEATURES_PATH = features_path
            try:
                from app.engine.impact_calculator import ImpactCalculator
                calc = ImpactCalculator(model_path=model_path)

                article = {
                    "ticker_symbols": ["AAPL"],
                    "published_at": "2024-06-15T15:00:00Z",
                }
                for sentiment in [-0.8, 0.0, 0.8]:
                    for novelty in [0.3, 0.7, 1.0]:
                        impact, (low, high) = calc.calculate(
                            event_type="Earnings",
                            sentiment=sentiment,
                            novelty=novelty,
                            diffusion_score=0.6,
                            article_dict=article,
                            source_count=2,
                        )
                        assert 1.0 <= impact <= 10.0, f"impact={impact} out of range"
                        assert 1.0 <= low <= impact <= high <= 10.0
            finally:
                _ic_module._XGB_FEATURES_PATH = original_features_path


class TestImpactAlwaysInRange:
    """Exhaustive range check across both modes."""

    def test_all_event_types_rule_based_always_in_range(self):
        from app.engine.impact_calculator import ImpactCalculator, BASE_IMPACT

        with patch.dict(os.environ, {"MODEL_IMPACT": "rules"}):
            calc = ImpactCalculator()
            for et in list(BASE_IMPACT.keys()) + ["UnknownType", ""]:
                impact, (lo, hi) = calc.calculate(
                    event_type=et,
                    sentiment=0.5,
                    novelty=0.5,
                )
                assert 1.0 <= impact <= 10.0
                assert lo <= impact <= hi

    def test_extreme_sentiment_values_rule_based(self):
        from app.engine.impact_calculator import ImpactCalculator

        with patch.dict(os.environ, {"MODEL_IMPACT": "rules"}):
            calc = ImpactCalculator()
            for sent in [-10.0, -1.0, 0.0, 1.0, 10.0]:
                impact, _ = calc.calculate("Geopolitical", sent, 0.9)
                assert 1.0 <= impact <= 10.0, f"Out of range for sentiment={sent}"


class TestFeatureExtractionMissingFields:
    """get_feature_vector must never raise on partial article dicts."""

    def test_handles_missing_ticker(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector(
            {"published_at": "2024-01-01T12:00:00Z"},
            sentiment=0.1,
            event_type="Macro",
        )
        # Should default to Technology sector
        assert features.get("sector_technology") == 1.0

    def test_handles_none_days_since_last_event(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector(
            {"days_since_last_event": None},
            sentiment=0.0,
            event_type="Earnings",
        )
        assert features["days_since_last"] == 30.0

    def test_handles_unknown_event_type_gracefully(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        # Unknown event type: all event_type_* one-hot columns should be 0
        features = model.get_feature_vector(
            {}, 0.0, "UnknownEventType"
        )
        from app.engine.xgboost_model import EVENT_TYPES
        for et in EVENT_TYPES:
            col = f"event_type_{et.lower().replace('&', 'and').replace(' ', '_')}"
            assert features[col] == 0.0

    def test_handles_invalid_published_at(self):
        from app.engine.xgboost_model import XGBoostImpactModel

        model = XGBoostImpactModel()
        features = model.get_feature_vector(
            {"published_at": "not-a-date"},
            sentiment=0.0,
            event_type="Earnings",
        )
        # Should not raise; is_after_hours defaults to 0.0 on parse failure
        assert "is_after_hours" in features
