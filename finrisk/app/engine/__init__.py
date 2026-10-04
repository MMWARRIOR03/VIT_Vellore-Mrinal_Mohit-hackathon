"""
NLP engine package — Task 3 of FinRisk.

Exports the main worker entry point and the individual component classes for
use in tests and other modules.
"""

from app.engine.sentiment import FinBERTSentimentModel
from app.engine.event_classifier import RuleBasedEventClassifier
from app.engine.novelty import NoveltyScorer
from app.engine.impact_calculator import ImpactCalculator
from app.engine.nlp_worker import NLPWorker

__all__ = [
    "FinBERTSentimentModel",
    "RuleBasedEventClassifier",
    "NoveltyScorer",
    "ImpactCalculator",
    "NLPWorker",
]
