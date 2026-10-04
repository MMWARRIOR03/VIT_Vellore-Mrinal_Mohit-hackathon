"""
evaluate.py — Standalone evaluation script for all FinRisk NLP components.

Sections:
  1. Sentiment model (FinBERT) on Financial PhraseBank
  2. Event classifier (RuleBasedClassifier, and DeBERTa if available)
  3. Impact model (XGBoost vs rule-based baseline, if training data exists)

Run from the finrisk/ root:
    python scripts/evaluate.py

Output: data/results/evaluation.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow running from finrisk/ root without package install
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("evaluate")

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "evaluation.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_rule(char: str = "━", width: int = 53) -> None:
    print(char * width)


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def _macro_f1(per_class: dict[str, float]) -> float:
    values = list(per_class.values())
    return sum(values) / len(values) if values else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Sentiment evaluation
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_SENTIMENT_EXAMPLES: list[tuple[str, str]] = [
    # (text, label)  label ∈ {positive, negative, neutral}
    ("Company beats quarterly earnings estimates by 15 percent", "positive"),
    ("Revenue grew strongly, raising full-year guidance", "positive"),
    ("Record profit driven by robust consumer demand", "positive"),
    ("Acquisition deal completed at premium to market price", "positive"),
    ("FDA approves new drug candidate ahead of schedule", "positive"),
    ("Stock upgraded to buy on improving fundamentals", "positive"),
    ("Share buyback programme signals board confidence", "positive"),
    ("Operating margins expanded on cost discipline", "positive"),
    ("Partnership with key technology firm announced", "positive"),
    ("Dividend raised for the twelfth consecutive quarter", "positive"),
    ("Sales growth accelerated despite market headwinds", "positive"),
    ("Quarterly revenue significantly exceeded expectations", "positive"),
    ("Analyst raises price target citing strong order backlog", "positive"),
    ("New product launch drives outperformance versus peers", "positive"),
    ("Capital return programme extended by additional billion", "positive"),
    ("Company missed earnings expectations for the third quarter", "negative"),
    ("Revenue declined fifteen percent year over year", "negative"),
    ("CEO resigned amid accounting investigation", "negative"),
    ("Major data breach exposed customer records", "negative"),
    ("Regulator imposed substantial fine for compliance failure", "negative"),
    ("Credit rating downgraded to junk by Moody's", "negative"),
    ("Bankruptcy filing expected after debt restructuring failed", "negative"),
    ("Ransomware attack disrupted operations across facilities", "negative"),
    ("Lawsuit filed over alleged product safety violations", "negative"),
    ("Guidance cut for second consecutive quarter", "negative"),
    ("Supply chain disruption causes production halt", "negative"),
    ("Company faces antitrust investigation from FTC", "negative"),
    ("Layoffs announced affecting ten thousand employees", "negative"),
    ("Quarterly loss widens on rising costs", "negative"),
    ("Export ban removes key revenue stream", "negative"),
    ("Federal Reserve held interest rates unchanged", "neutral"),
    ("Company to present at investor conference next month", "neutral"),
    ("Annual report filed with Securities and Exchange Commission", "neutral"),
    ("Management change at subsidiary level announced", "neutral"),
    ("Scheduled maintenance reduces output temporarily", "neutral"),
    ("Board approves routine capital expenditure plan", "neutral"),
    ("Quarterly filing submitted ahead of deadline", "neutral"),
    ("Company updates investor relations website", "neutral"),
    ("Analyst reiterates hold rating with unchanged price target", "neutral"),
    ("Product roadmap presentation scheduled for next quarter", "neutral"),
]


def evaluate_sentiment() -> dict:
    """Evaluate FinBERT (or rule-based fallback) on Financial PhraseBank."""
    logger.info("=== Section 1: Sentiment evaluation ===")

    # Attempt to load Financial PhraseBank from HuggingFace datasets
    texts: list[str] = []
    labels: list[str] = []
    dataset_source = "synthetic"

    try:
        from datasets import load_dataset  # type: ignore[import]
        logger.info("Loading Financial PhraseBank from HuggingFace …")
        ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
        split = ds["train"]
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        for row in split:
            texts.append(str(row["sentence"]))
            labels.append(label_map[int(row["label"])])
        dataset_source = "financial_phrasebank"
        logger.info("Loaded %d samples from Financial PhraseBank", len(texts))
    except Exception as exc:
        logger.warning(
            "Financial PhraseBank unavailable (%s) — using %d synthetic samples",
            exc,
            len(SYNTHETIC_SENTIMENT_EXAMPLES),
        )
        texts, labels = zip(*SYNTHETIC_SENTIMENT_EXAMPLES)
        texts = list(texts)
        labels = list(labels)

    # Load model
    from app.engine.sentiment import FinBERTSentimentModel
    model = FinBERTSentimentModel()
    logger.info("Running sentiment predictions on %d samples …", len(texts))

    preds: list[str] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        results = model.predict_batch(batch)
        for score, _conf in results:
            if score > 0.05:
                preds.append("positive")
            elif score < -0.05:
                preds.append("negative")
            else:
                preds.append("neutral")
        if (i // batch_size + 1) % 10 == 0:
            logger.info("  Processed %d/%d", min(i + batch_size, len(texts)), len(texts))

    # Compute metrics
    class_names = ["positive", "negative", "neutral"]
    tp: dict[str, int] = {c: 0 for c in class_names}
    fp: dict[str, int] = {c: 0 for c in class_names}
    fn: dict[str, int] = {c: 0 for c in class_names}
    correct = 0

    for pred, true in zip(preds, labels):
        if pred == true:
            correct += 1
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    accuracy = correct / len(labels) if labels else 0.0
    per_class_f1 = {c: _safe_f1(tp[c], fp[c], fn[c]) for c in class_names}
    macro_f1 = _macro_f1(per_class_f1)

    # Confusion matrix: rows=true, cols=pred
    confusion: dict[str, dict[str, int]] = {
        t: {p: 0 for p in class_names} for t in class_names
    }
    for pred, true in zip(preds, labels):
        confusion[true][pred] += 1

    result = {
        "dataset": dataset_source,
        "n_samples": len(labels),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
        "confusion_matrix": confusion,
    }
    logger.info(
        "Sentiment: accuracy=%.1f%%  macro_f1=%.3f",
        accuracy * 100,
        macro_f1,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Event classifier evaluation
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFIER_EVENT_TYPES = [
    "Geopolitical", "Regulatory", "M&A", "Earnings", "Credit",
    "Macro", "Product", "Cyber", "Leadership", "Commodity",
]

# Alias used throughout this file
EVENT_TYPES = _CLASSIFIER_EVENT_TYPES

# 10 headlines per event type — representative but varied phrasing
SYNTHETIC_EVENT_EXAMPLES: dict[str, list[str]] = {
    "Earnings": [
        "Company beats Q3 earnings estimates by 15 percent",
        "Revenue fell short of analyst expectations in fiscal Q2",
        "Quarterly profit surged on strong consumer spending",
        "EPS guidance raised for full year after record sales",
        "Operating income declined despite revenue growth",
        "Management cut full-year outlook citing rising costs",
        "Third quarter results showed accelerating sales growth",
        "Annual earnings beat consensus for fifth consecutive year",
        "Loss widened in Q4 due to one-time restructuring charges",
        "Revenue per share hit record high in latest quarter",
    ],
    "M&A": [
        "Tech giant acquires startup for two billion dollars",
        "Merger agreement signed in all-cash deal",
        "Takeover bid rejected by target company board",
        "Buyout firm completes leveraged acquisition of retailer",
        "Strategic combination creates industry leader",
        "Spinoff of consumer division expected next quarter",
        "All-stock deal values target at premium to close",
        "Tender offer launched at thirty percent premium",
        "Divests healthcare unit to private equity group",
        "Carve-out transaction creates standalone business",
    ],
    "Regulatory": [
        "SEC launches investigation into accounting practices",
        "Company fined for antitrust violations",
        "Regulator bans product pending safety review",
        "DOJ probe focuses on pricing practices",
        "FTC challenges proposed merger on competition grounds",
        "Compliance failures lead to consent decree",
        "Penalty imposed for data privacy violations",
        "Subpoena issued as part of securities enforcement",
        "Settlement reached in long-running regulatory dispute",
        "License revoked following repeated violations",
    ],
    "Macro": [
        "Federal Reserve raises interest rates by fifty basis points",
        "Inflation data exceeds forecasts raising rate hike fears",
        "GDP growth slowed more than expected in third quarter",
        "Unemployment rate falls to lowest level since pandemic",
        "FOMC signals pause in tightening cycle",
        "Treasury yields surge on strong economic data",
        "Central bank cuts rates to support slowing economy",
        "CPI print surprises to upside stoking inflation concerns",
        "Quantitative tightening accelerated by Federal Reserve",
        "Yield curve inverts signalling recession risk",
    ],
    "Geopolitical": [
        "New sanctions imposed on country over military actions",
        "Trade war escalates with fresh tariff round",
        "Conflict disrupts supply chains across region",
        "Export ban restricts semiconductor shipments",
        "NATO allies increase defence spending commitments",
        "Diplomatic talks collapse between major economies",
        "Military tensions raise commodity price concerns",
        "Treaty signed reducing trade barriers between blocs",
        "Invasion triggers flight to safety in global markets",
        "Embargo extended for second consecutive year",
    ],
    "Credit": [
        "Moody's downgrades company debt to junk status",
        "Bankruptcy filing expected after restructuring fails",
        "Default on bond payment triggers credit event",
        "Fitch places rating on negative creditwatch",
        "High-yield spreads widen on rising default fears",
        "Covenant breach triggers lender acceleration",
        "S&P downgrades sovereign credit outlook",
        "Leveraged loan market tightens on credit concerns",
        "Debt restructuring agreement reached with bondholders",
        "Credit watch negative placed ahead of refinancing",
    ],
    "Product": [
        "FDA approves new cancer drug for commercial launch",
        "Product recall issued over safety concerns",
        "Clinical trial results show strong efficacy data",
        "New platform version releases with major features",
        "Patent granted for breakthrough technology",
        "Drug pipeline milestone accelerates approval timeline",
        "Device approved ahead of schedule by regulators",
        "Service launch expands addressable market significantly",
        "Product liability claims increase after recall",
        "Version two of software platform exceeds user targets",
    ],
    "Cyber": [
        "Major data breach exposes millions of customer records",
        "Ransomware attack disrupts hospital operations",
        "Zero-day vulnerability exploited in widely used software",
        "Cybersecurity incident leads to service outage",
        "Hackers steal intellectual property from manufacturer",
        "Phishing campaign targets employee credentials",
        "Data theft incident triggers regulatory investigation",
        "Malware detected in critical infrastructure systems",
        "Intrusion discovered months after initial breach",
        "Cyber attack disrupts payment processing networks",
    ],
    "Leadership": [
        "Chief executive officer resigns unexpectedly",
        "Company names new CEO from outside the industry",
        "CFO departure raises questions about strategy",
        "Board appoints interim leader during search",
        "Chairman steps down following activist pressure",
        "C-suite restructuring follows poor quarterly results",
        "Director promoted to chief operating officer role",
        "Executive team overhaul signals strategic pivot",
        "New chief financial officer named effective immediately",
        "Succession plan activated after founder departure",
    ],
    "Commodity": [
        "Oil prices surge as OPEC cuts production quota",
        "Gold hits record high on safe haven demand",
        "Copper supply chain disruption raises prices",
        "Natural gas shortage drives energy price spike",
        "Wheat prices jump on drought in major producer",
        "Lithium prices fall on weakening battery demand",
        "Crude oil drops on weak demand outlook",
        "Rare earth export restrictions announced",
        "Energy price volatility affects downstream margins",
        "Commodity market tightness persists into next quarter",
    ],
}


def evaluate_event_classifier() -> dict:
    """Evaluate RuleBasedEventClassifier on labeled headlines."""
    logger.info("=== Section 2: Event classifier evaluation ===")

    from app.engine.event_classifier import RuleBasedEventClassifier

    # Try loading held-out data from data/training/
    texts: list[str] = []
    true_labels: list[str] = []
    data_source = "synthetic"

    training_path = Path("data/training/labeled_events.parquet")
    if training_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(training_path)
            if "headline" in df.columns and "event_type" in df.columns:
                texts = df["headline"].tolist()
                true_labels = df["event_type"].tolist()
                data_source = str(training_path)
                logger.info("Loaded %d labeled headlines from %s", len(texts), training_path)
        except Exception as exc:
            logger.warning("Could not load training data (%s) — using synthetic", exc)

    if not texts:
        for event_type, headlines in SYNTHETIC_EVENT_EXAMPLES.items():
            for h in headlines:
                texts.append(h)
                true_labels.append(event_type)
        data_source = "synthetic"
        logger.info("Using %d synthetic labeled headlines", len(texts))

    # Rule-based classifier
    clf = RuleBasedEventClassifier()
    rb_preds: list[str] = []
    for text in texts:
        primary, _ = clf.get_primary(text)
        rb_preds.append(primary)

    # Compute metrics
    rb_metrics = _compute_classifier_metrics(true_labels, rb_preds, EVENT_TYPES)

    result: dict = {
        "data_source": data_source,
        "n_samples": len(texts),
        "rule_based": rb_metrics,
    }

    # DeBERTa classifier (if available)
    deberta_path = Path("data/models/event_deberta")
    if deberta_path.exists():
        try:
            from app.engine.deberta_classifier import DeBERTaEventClassifier
            logger.info("DeBERTa model found — running DeBERTa evaluation …")
            deberta_clf = DeBERTaEventClassifier()
            db_preds: list[str] = []
            for text in texts:
                pred = deberta_clf.predict(text)
                # predict() returns dict; get top key
                if isinstance(pred, dict):
                    primary = max(pred, key=lambda k: pred[k])
                else:
                    primary = str(pred)
                db_preds.append(primary)
            db_metrics = _compute_classifier_metrics(true_labels, db_preds, EVENT_TYPES)
            result["deberta"] = db_metrics
        except Exception as exc:
            logger.warning("DeBERTa evaluation failed: %s", exc)
    else:
        logger.info("DeBERTa model not found at %s — skipping", deberta_path)

    logger.info(
        "Event (rule-based): accuracy=%.1f%%  macro_f1=%.3f",
        rb_metrics["accuracy"] * 100,
        rb_metrics["macro_f1"],
    )
    return result


def _compute_classifier_metrics(
    true_labels: list[str],
    pred_labels: list[str],
    class_names: list[str],
) -> dict:
    tp: dict[str, int] = {c: 0 for c in class_names}
    fp: dict[str, int] = {c: 0 for c in class_names}
    fn: dict[str, int] = {c: 0 for c in class_names}
    correct = 0

    for pred, true in zip(pred_labels, true_labels):
        if pred == true:
            correct += 1
            if true in tp:
                tp[true] += 1
        else:
            if pred in fp:
                fp[pred] += 1
            if true in fn:
                fn[true] += 1

    accuracy = correct / len(true_labels) if true_labels else 0.0
    per_class_f1 = {c: _safe_f1(tp[c], fp[c], fn[c]) for c in class_names}
    macro_f1 = _macro_f1(per_class_f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Impact model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_impact_model() -> dict:
    """Evaluate impact model (XGBoost vs rule-based) on labeled_events.parquet."""
    logger.info("=== Section 3: Impact model evaluation ===")

    import math

    from app.engine.xgboost_model import XGBoostImpactModel, BASE_IMPACT, EVENT_TYPES as XGB_EVENT_TYPES

    training_path = Path("data/training/labeled_events.parquet")
    model_path = Path("data/models/impact_xgb.pkl")

    # Build synthetic fallback test set
    def _make_synthetic_test_set() -> tuple[list[dict], list[float]]:
        import random
        rng = random.Random(999)
        samples: list[dict] = []
        targets: list[float] = []
        for _ in range(150):
            et = rng.choice(XGB_EVENT_TYPES)
            sent = round(rng.uniform(-1.0, 1.0), 3)
            base = BASE_IMPACT.get(et, 5.0)
            # Ground truth: rule-based formula + small noise
            true_impact = max(1.0, min(10.0, base * (0.5 + 0.5 * abs(sent)) + rng.gauss(0, 0.8)))
            samples.append({
                "event_type": et,
                "sentiment": sent,
                "source_count": rng.randint(1, 20),
                "days_since_last_event": rng.uniform(0, 90),
            })
            targets.append(round(true_impact, 2))
        return samples, targets

    # Load real data if available
    sample_dicts: list[dict] = []
    true_impacts: list[float] = []
    data_source = "synthetic"

    if training_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(training_path)
            required = {"event_type", "sentiment", "abnormal_return"}
            if required.issubset(set(df.columns)):
                df = df.dropna(subset=list(required))
                # Use |abnormal_return| as the target for Spearman correlation
                for _, row in df.iterrows():
                    sample_dicts.append({
                        "event_type": str(row["event_type"]),
                        "sentiment": float(row["sentiment"]),
                        "source_count": int(row.get("source_count", 1)),
                        "days_since_last_event": float(row.get("days_since_last", 30.0)),
                    })
                    true_impacts.append(abs(float(row["abnormal_return"])) * 10.0)
                data_source = str(training_path)
                logger.info("Loaded %d samples from %s", len(sample_dicts), training_path)
        except Exception as exc:
            logger.warning("Could not load training data (%s) — using synthetic", exc)

    if not sample_dicts:
        sample_dicts, true_impacts = _make_synthetic_test_set()
        data_source = "synthetic"
        logger.info("Using %d synthetic impact test samples", len(sample_dicts))

    # Rule-based predictions
    rb_preds: list[float] = []
    for s in sample_dicts:
        et = s["event_type"]
        sent = float(s["sentiment"])
        base = BASE_IMPACT.get(et, 5.0)
        rb_pred = max(1.0, min(10.0, base * (0.5 + 0.5 * abs(sent))))
        rb_preds.append(rb_pred)

    rb_mae = sum(abs(p - t) for p, t in zip(rb_preds, true_impacts)) / len(true_impacts)
    rb_rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(rb_preds, true_impacts)) / len(true_impacts))

    result: dict = {
        "data_source": data_source,
        "n_samples": len(sample_dicts),
        "rule_based": {
            "mae": round(rb_mae, 4),
            "rmse": round(rb_rmse, 4),
        },
    }

    # XGBoost predictions (if model exists)
    xgb_model = XGBoostImpactModel()
    if xgb_model.load(model_path):
        xgb_preds: list[float] = []
        for s in sample_dicts:
            features = xgb_model.get_feature_vector(
                article_dict=s,
                sentiment=float(s["sentiment"]),
                event_type=str(s["event_type"]),
                source_count=int(s.get("source_count", 1)),
            )
            pred = xgb_model.predict(features)
            xgb_preds.append(pred)

        xgb_mae = sum(abs(p - t) for p, t in zip(xgb_preds, true_impacts)) / len(true_impacts)
        xgb_rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(xgb_preds, true_impacts)) / len(true_impacts))

        # Spearman correlation between predictions and true_impacts
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(xgb_preds, true_impacts)
            spearman_rho = float(rho)
        except ImportError:
            # Manual Spearman rank correlation
            def _rank(lst: list[float]) -> list[float]:
                sorted_idx = sorted(range(len(lst)), key=lambda i: lst[i])
                ranks = [0.0] * len(lst)
                for rank, idx in enumerate(sorted_idx):
                    ranks[idx] = float(rank + 1)
                return ranks
            rx = _rank(xgb_preds)
            ry = _rank(true_impacts)
            n = len(rx)
            d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
            spearman_rho = 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1)) if n > 2 else 0.0

        result["xgboost"] = {
            "mae": round(xgb_mae, 4),
            "rmse": round(xgb_rmse, 4),
            "spearman_rho": round(spearman_rho, 4),
        }
        logger.info(
            "Impact: RB-MAE=%.3f  XGB-MAE=%.3f  Spearman=%.3f",
            rb_mae, xgb_mae, spearman_rho,
        )
    else:
        logger.info(
            "XGBoost model not found at %s — rule-based metrics only", model_path
        )
        logger.info("Impact: RB-MAE=%.3f  RB-RMSE=%.3f", rb_mae, rb_rmse)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(sentiment: dict, event: dict, impact: dict) -> None:
    _print_rule()
    print("FinRisk Evaluation Summary")
    _print_rule()

    # Sentiment
    print(f"SENTIMENT (FinBERT on {sentiment['dataset']})")
    print(f"  Samples:    {sentiment['n_samples']}")
    print(f"  Accuracy:   {sentiment['accuracy'] * 100:.1f}%")
    print(f"  Macro F1:   {sentiment['macro_f1']:.3f}")
    pf = sentiment["per_class_f1"]
    print(f"  F1 pos/neg/neu: {pf['positive']:.3f} / {pf['negative']:.3f} / {pf['neutral']:.3f}")
    print()

    # Event
    rb = event["rule_based"]
    print(f"EVENT CLASSIFICATION (RuleBasedClassifier, {event['data_source']})")
    print(f"  Samples:    {event['n_samples']}")
    print(f"  Accuracy:   {rb['accuracy'] * 100:.1f}%")
    print(f"  Macro F1:   {rb['macro_f1']:.3f}")
    if "deberta" in event:
        db = event["deberta"]
        print(f"  DeBERTa F1: {db['macro_f1']:.3f}  (Accuracy: {db['accuracy'] * 100:.1f}%)")
    print()

    # Impact
    irb = impact["rule_based"]
    print(f"IMPACT MODEL ({impact['data_source']})")
    print(f"  Samples:         {impact['n_samples']}")
    print(f"  Rule-based MAE:  {irb['mae']:.3f}")
    print(f"  Rule-based RMSE: {irb['rmse']:.3f}")
    if "xgboost" in impact:
        xgb = impact["xgboost"]
        print(f"  XGBoost MAE:     {xgb['mae']:.3f}")
        print(f"  XGBoost RMSE:    {xgb['rmse']:.3f}")
        print(f"  XGBoost Spearman: {xgb['spearman_rho']:.3f}")

    _print_rule()
    print(f"Results saved to {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    results: dict = {}

    logger.info("Starting FinRisk evaluation …")

    results["sentiment"] = evaluate_sentiment()
    results["event_classification"] = evaluate_event_classifier()
    results["impact_model"] = evaluate_impact_model()

    # Save to JSON
    with OUTPUT_PATH.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Saved full results to %s", OUTPUT_PATH)

    print()
    print_summary(
        results["sentiment"],
        results["event_classification"],
        results["impact_model"],
    )


if __name__ == "__main__":
    main()
