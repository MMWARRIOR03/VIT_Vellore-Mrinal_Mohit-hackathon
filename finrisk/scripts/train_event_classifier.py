#!/usr/bin/env python3
"""
Fine-tune microsoft/deberta-v3-base for 10-class financial event classification.

Usage:
    python scripts/train_event_classifier.py
    python scripts/train_event_classifier.py --synthetic-only
    python scripts/train_event_classifier.py --epochs 5 --batch-size 8
    python scripts/train_event_classifier.py --output-dir data/models/my_classifier
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Allow running from the project root (finrisk/) without pip-installing the package
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train_event_classifier")

# ── Constants ─────────────────────────────────────────────────────────────────

EVENT_TYPES: list[str] = [
    "Earnings",
    "M&A",
    "Regulatory",
    "Macro",
    "Geopolitical",
    "Credit",
    "Commodity",
    "Cyber",
    "Leadership",
    "Product",
]

_BASE_MODEL_ID = "microsoft/deberta-v3-base"
_GDELT_PARQUET_PATH = Path("data/training/gdelt_gkg_filtered.parquet")

# ── Synthetic headlines ───────────────────────────────────────────────────────

SYNTHETIC_HEADLINES: dict[str, list[str]] = {
    "Earnings": [
        "Apple reports record quarterly earnings, beats analyst estimates",
        "Tesla misses Q3 revenue expectations, shares fall after hours",
        "Amazon raises full-year guidance following strong Prime Day sales",
        "Microsoft reports earnings per share of $2.94, above $2.78 consensus",
        "Nvidia Q2 profit surges 150% on AI chip demand",
        "Goldman Sachs reports 40% decline in investment banking revenue",
        "Meta Q4 revenue beats expectations as ad spending rebounds",
        "Netflix subscriber growth exceeds forecasts in Q1 earnings report",
        "JPMorgan posts record annual profit driven by rising interest rates",
        "Boeing reports wider-than-expected loss on 737 MAX charges",
        "Alphabet third-quarter earnings beat consensus on YouTube ad recovery",
        "Ford cuts full-year earnings guidance citing chip shortage costs",
        "Disney reports streaming losses narrow as Parks revenue hits record",
        "Pfizer misses revenue estimates as COVID vaccine demand wanes",
        "Salesforce raises fiscal-year outlook after beating Q2 expectations",
        "Intel Q1 earnings miss estimates on weak PC demand",
        "Uber reports first full-year operating profit in company history",
        "Shopify quarterly revenue grows 25%, tops Wall Street forecasts",
        "General Electric narrows annual loss, reaffirms turnaround plan",
        "Starbucks same-store sales fall in China, weighing on quarterly results",
    ],
    "M&A": [
        "Microsoft announces $69 billion acquisition of Activision Blizzard",
        "Pfizer agrees to acquire oncology firm for $43 billion in all-cash deal",
        "Amazon completes $3.9 billion buyout of One Medical primary care chain",
        "ExxonMobil to acquire Pioneer Natural Resources in $60 billion deal",
        "Broadcom agrees to purchase VMware for $61 billion",
        "Adobe calls off $20 billion Figma acquisition amid regulatory scrutiny",
        "Chevron acquires Hess Corporation for $53 billion in all-stock deal",
        "Sony acquires game developer Bungie for $3.6 billion",
        "Kroger and Albertsons agree to $25 billion merger to compete with Walmart",
        "LVMH acquires Tiffany for $15.8 billion after lengthy legal dispute",
        "Salesforce buys Slack for $27.7 billion in largest deal in its history",
        "Nvidia abandons $40 billion ARM acquisition after regulatory opposition",
        "AT&T spins off WarnerMedia in $43 billion deal with Discovery",
        "Banco Santander acquires Openbank to expand digital banking footprint",
        "Merger between Spirit Airlines and Frontier Airlines collapses",
        "UnitedHealth acquires Change Healthcare for $13 billion",
        "Thermo Fisher to buy Binding Site Group for $2.6 billion",
        "IBM divests infrastructure services unit into new company Kyndryl",
        "Square acquires Australian buy-now-pay-later firm Afterpay for $29 billion",
        "Hyundai Motor acquires Boston Dynamics from SoftBank for $1.1 billion",
    ],
    "Regulatory": [
        "FTC sues Amazon over alleged monopolistic practices in online retail",
        "SEC charges Binance and Coinbase with unregistered securities violations",
        "DOJ files antitrust lawsuit against Google over search advertising",
        "EU regulators fine Meta 1.2 billion euros for GDPR data transfer violations",
        "Federal Reserve orders Wells Fargo to cap asset growth over compliance failures",
        "CFPB issues rule limiting credit card late fees to $8",
        "FDA places clinical hold on Biogen Alzheimer drug trial after safety concerns",
        "CFTC reaches $200 million settlement with JPMorgan over recordkeeping lapses",
        "European Commission opens formal antitrust probe into Apple App Store",
        "FCC votes to restore net neutrality rules for broadband providers",
        "Congress passes bill requiring TikTok parent ByteDance to divest US operations",
        "FINRA fines Robinhood $70 million for service outages and misleading customers",
        "IRS launches compliance campaign targeting large partnership tax shelters",
        "NHTSA expands Tesla Autopilot recall to cover 2 million vehicles",
        "EPA levies $14 million fine on 3M for PFAS contamination violations",
        "Pfizer reaches $75 million settlement with DOJ over off-label drug marketing",
        "FTC challenges Illumina acquisition of Grail on anticompetitive grounds",
        "Bank of America pays $250 million to settle junk fee and fake account charges",
        "FAA grounds Boeing 737 MAX 9 fleet after door plug blowout incident",
        "Visa and Mastercard reach $30 billion settlement in merchant fee lawsuit",
    ],
    "Macro": [
        "Federal Reserve raises interest rates by 25 basis points, signals pause",
        "US inflation falls to 3.1% in November, lowest since early 2021",
        "GDP growth slows to 1.9% annualized in third quarter on consumer slowdown",
        "Unemployment rate rises to 4.1% as layoffs accelerate in tech sector",
        "Federal Reserve holds rates steady amid mixed economic signals",
        "US Treasury yields hit 16-year high above 5% on resilient jobs data",
        "Bank of Japan abandons yield curve control policy, sending yen higher",
        "ECB raises deposit rate to 4% as eurozone inflation remains elevated",
        "US consumer price index rises 0.4% in March, above economist forecasts",
        "China central bank cuts reserve requirement ratio to support growth",
        "PCE inflation index rises 3.5% year-over-year, remaining above Fed target",
        "US nonfarm payrolls add 256,000 jobs in December, crushing expectations",
        "Fed Chair Powell signals rate cuts coming in 2024 at Jackson Hole",
        "FOMC meeting minutes reveal concerns about persistent services inflation",
        "UK enters technical recession after GDP contracts for second straight quarter",
        "IMF upgrades US growth forecast to 2.7% for 2024, cuts Europe outlook",
        "US 2-year Treasury yield briefly inverts with 10-year by record 107 basis points",
        "Federal Open Market Committee votes to hold federal funds rate at 5.25–5.5%",
        "Consumer confidence index drops to six-month low amid tariff uncertainty",
        "Money supply M2 contracts for first time since Great Depression",
    ],
    "Geopolitical": [
        "US imposes sweeping new sanctions on Russia targeting energy exports",
        "China announces export controls on gallium and germanium over trade war",
        "NATO allies pledge additional military aid to Ukraine after summit",
        "Taiwan Strait tensions escalate as China conducts live-fire military drills",
        "Iran nuclear deal negotiations collapse over ballistic missile program",
        "North Korea fires ballistic missile into waters off Japan coast",
        "US bans advanced AI chip exports to China and Russia",
        "Saudi Arabia and Iran restore diplomatic ties brokered by China",
        "EU agrees on 13th sanctions package targeting Russian oil revenue",
        "Israel-Gaza conflict escalates, raising Red Sea shipping disruption fears",
        "South Korea declares martial law, then reverses decision within hours",
        "Houthi attacks on commercial shipping in Red Sea disrupt global trade",
        "US and EU launch joint tariff investigation into Chinese electric vehicles",
        "Russia suspends participation in New START nuclear arms treaty",
        "OPEC+ agrees to extend voluntary oil production cuts through mid-2024",
        "G7 imposes price cap of $60 per barrel on Russian crude oil",
        "Myanmar military seizes power in coup, triggering US sanctions",
        "Pakistan defaults on sovereign debt as foreign currency reserves collapse",
        "US and Philippines agree to expanded military base access in South China Sea",
        "Venezuela election disputed internationally as opposition claims fraud",
    ],
    "Credit": [
        "Moody's downgrades US credit rating to Aa1 on rising debt concerns",
        "Silicon Valley Bank collapses after bank run, FDIC takes receivership",
        "S&P places AT&T on CreditWatch Negative over debt load concerns",
        "Evergrande liquidation ordered by Hong Kong court after debt default",
        "Fitch downgrades France sovereign debt citing political instability",
        "Argentina defaults on $65 billion in sovereign bonds for ninth time",
        "WeWork files for Chapter 11 bankruptcy protection after cash burn",
        "Credit Suisse AT1 bondholders sue Switzerland over $17 billion writedown",
        "Bed Bath & Beyond files for Chapter 11 after failing to raise cash",
        "Rite Aid files for bankruptcy protection amid opioid lawsuit liabilities",
        "Moody's cuts rating on 10 US regional banks citing funding stress",
        "Yellow Corporation files for bankruptcy, laying off 30,000 workers",
        "Evergrande creditors approve restructuring plan for offshore debt",
        "Hertz emerges from bankruptcy with $5.9 billion restructuring completed",
        "S&P upgrades Greece sovereign debt to investment grade for first time",
        "Sri Lanka reaches $2.9 billion IMF bailout agreement after 2022 default",
        "Puerto Rico exits bankruptcy after six-year debt restructuring process",
        "Lebanon unable to service $31 billion foreign debt, credit rating collapses",
        "Fitch cuts Italy sovereign credit outlook to negative on fiscal slippage",
        "Diamond Sports Group files for bankruptcy as RSN model collapses",
    ],
    "Commodity": [
        "Oil prices surge 4% as OPEC+ announces surprise production cut of 1 million bpd",
        "Natural gas futures fall to three-year low on mild winter demand",
        "Gold hits all-time high above $2,400 per ounce on safe-haven demand",
        "Copper prices drop on weak Chinese manufacturing data",
        "Wheat prices spike 20% after Russia suspends grain deal with Ukraine",
        "Lithium carbonate prices plunge 80% from peak on EV demand slowdown",
        "Brent crude falls below $70 as US shale output hits record highs",
        "Silver prices rally on industrial demand from solar panel manufacturing",
        "Corn and soybean futures tumble on record South American harvest forecast",
        "LNG spot prices collapse as European gas storage reaches 100% capacity",
        "Iron ore drops 15% in a week on China property sector weakness",
        "Nickel market chaos after London Metal Exchange halts trading mid-session",
        "Oil supply tightens as Saudi Arabia extends voluntary 1 million bpd cut",
        "Cocoa futures hit 46-year high on West African crop disease outbreak",
        "Rare earth prices fall as China eases export controls following trade talks",
        "Uranium spot price rises to 15-year high on nuclear energy revival",
        "US Gulf Coast benchmark natural gas turns negative for first time ever",
        "Palladium falls below platinum for first time as EV adoption reduces auto use",
        "Agricultural commodity index rises as El Niño threatens Asian rice harvest",
        "Coffee prices surge on Brazil drought cutting arabica output forecasts",
    ],
    "Cyber": [
        "MGM Resorts hit by ransomware attack, casino systems offline for days",
        "MOVEit zero-day exploit exposes data of 200 organizations worldwide",
        "SolarWinds hackers breach Treasury and Commerce Department networks",
        "LastPass discloses breach exposing encrypted customer password vaults",
        "Uber confirms data breach after hacker uses social engineering to gain access",
        "T-Mobile agrees to $350 million settlement for data breach affecting 76 million",
        "Okta discloses customer support system breach affecting hundreds of clients",
        "Microsoft Exchange vulnerabilities exploited in mass cyberattack campaign",
        "Log4Shell vulnerability triggers global emergency patching across enterprises",
        "Latitude Financial discloses cyberattack exposing 14 million customer records",
        "Change Healthcare ransomware attack disrupts US pharmacy systems for weeks",
        "CloudFlare discloses nation-state intrusion into internal Atlassian systems",
        "SEC charges SolarWinds CISO with fraud over cybersecurity disclosure failures",
        "23andMe breach exposes genetic data of 6.9 million users via credential stuffing",
        "UnitedHealth Group pays $22 million ransom to restore Change Healthcare systems",
        "Snowflake customer data stolen via credential theft campaign hitting 165 companies",
        "US Treasury sanctions Lazarus Group following $615 million Ronin bridge hack",
        "British Library ransomware attack destroys large portions of internal systems",
        "Fortinet discloses exploitation of critical VPN vulnerability in government networks",
        "Boeing data leaked by LockBit after refusing ransom demand of $200 million",
    ],
    "Leadership": [
        "Bob Iger returns as Disney CEO, replacing Bob Chapek amid studio losses",
        "Sam Altman fired as OpenAI CEO, then reinstated after board reversal",
        "Twitter CEO Parag Agrawal ousted after Elon Musk completes $44 billion takeover",
        "Starbucks names Brian Niccol CEO, shares surge on turnaround hopes",
        "Intel names Lip-Bu Tan as CEO to accelerate semiconductor turnaround",
        "Goldman Sachs CEO David Solomon steps back from DJ career after investor pressure",
        "Salesforce founder Marc Benioff resumes sole CEO role after co-CEO departs",
        "Boeing CEO Dave Calhoun to step down following 737 MAX safety crisis",
        "Walt Disney board rejects Nelson Peltz bid for two seats in proxy fight",
        "McKinsey CEO Dominic Barton steps down after a decade leading firm",
        "Rivian appoints Volkswagen executive as new chief financial officer",
        "WeWork founder Adam Neumann attempts to repurchase bankrupt company",
        "Nike CEO John Donahoe steps down, Elliott Hill named as replacement",
        "Ford CFO John Lawler promoted to vice chairman as leadership shakeup",
        "Warner Bros. CEO David Zaslav extends contract despite box office struggles",
        "Alphabet CFO Ruth Porat moves to new role as President of Alphabet",
        "Carlos Ghosn escapes Japan while awaiting financial misconduct trial",
        "Apple COO Jeff Williams elevated to broader operational responsibilities",
        "Peloton names Barry McCarthy as CEO to replace founder John Foley",
        "BP CEO Bernard Looney resigns over undisclosed relationships with colleagues",
    ],
    "Product": [
        "Apple launches Vision Pro spatial computing headset at $3,499",
        "FDA approves Novo Nordisk weight-loss drug Wegovy for cardiovascular risk",
        "Tesla recalls 2 million vehicles to fix Autopilot monitoring deficiencies",
        "OpenAI releases GPT-4 with multimodal capabilities for ChatGPT Plus users",
        "Boeing receives FAA approval to resume 737 MAX 10 test flights",
        "FDA grants breakthrough therapy designation to Eli Lilly Alzheimer drug",
        "Google launches Gemini Ultra AI model to compete with OpenAI GPT-4",
        "Apple delays Apple Intelligence AI features to later iOS 18 update",
        "Samsung recalls Galaxy Note 7 smartphones over battery explosion risk",
        "Pfizer halts Phase 3 trial of oral GLP-1 obesity drug on liver concerns",
        "Amazon launches Alexa Plus subscription with generative AI capabilities",
        "Cruise autonomous vehicle unit suspended after pedestrian collision incident",
        "Meta releases open-source Llama 3 model challenging closed AI systems",
        "Boeing 737 MAX 9 door plug blowout prompts immediate fleet grounding",
        "SpaceX Starship completes first successful orbital test flight with splashdown",
        "Moderna cancer mRNA vaccine shows 44% reduction in melanoma recurrence",
        "Apple granted patent for foldable iPhone display mechanism",
        "Waymo receives permit to charge for driverless rides across San Francisco",
        "FDA clears Neuralink to begin human brain-computer interface clinical trials",
        "Sony launches PlayStation 5 Pro with enhanced GPU for 4K gaming at $699",
    ],
}

# ── Augmentation helpers ───────────────────────────────────────────────────────

_PREFIXES = [
    "BREAKING: ",
    "EXCLUSIVE: ",
    "UPDATE: ",
    "REPORT: ",
    "",
    "",
    "",
]

_SUFFIXES = [
    " — sources say",
    " amid market uncertainty",
    ", analysts say",
    ", filing shows",
    " according to Reuters",
    "",
    "",
    "",
]


def _augment_headline(headline: str, rng: random.Random) -> str:
    """Apply random prefix/suffix paraphrasing to a headline."""
    prefix = rng.choice(_PREFIXES)
    suffix = rng.choice(_SUFFIXES)
    # Randomly lowercase first char for variety
    if rng.random() < 0.3 and headline:
        headline = headline[0].lower() + headline[1:]
    return f"{prefix}{headline}{suffix}"


def _oversample_to_target(
    texts: list[str], target: int, rng: random.Random
) -> list[str]:
    """Oversample ``texts`` to reach ``target`` count using augmentation."""
    result = list(texts)
    seed_texts = texts
    while len(result) < target:
        original = rng.choice(seed_texts)
        result.append(_augment_headline(original, rng))
    return result[:target]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_training_data(
    synthetic_only: bool,
) -> tuple[list[str], list[int]]:
    """
    Build training corpus.

    When ``synthetic_only`` is False and the GDELT parquet file exists at
    ``data/training/gdelt_gkg_filtered.parquet``, it is loaded and expected
    to have columns ``headline`` (str) and ``event_type`` (str).  Only rows
    whose ``event_type`` is in EVENT_TYPES are kept.  The synthetic headlines
    are always included and the corpus is oversampled to ~500 examples per
    class.

    Returns (texts, labels) where labels are integer indices into EVENT_TYPES.
    """
    label2id = {label: idx for idx, label in enumerate(EVENT_TYPES)}
    rng = random.Random(42)
    target_per_class = 500

    texts_by_class: dict[str, list[str]] = {et: [] for et in EVENT_TYPES}

    # Seed with synthetic headlines
    for event_type, headlines in SYNTHETIC_HEADLINES.items():
        texts_by_class[event_type].extend(headlines)

    # Optionally load GDELT data
    if not synthetic_only and _GDELT_PARQUET_PATH.exists():
        try:
            import pandas as pd  # noqa: F401

            logger.info("Loading GDELT data from %s", _GDELT_PARQUET_PATH)
            df = pd.read_parquet(_GDELT_PARQUET_PATH)

            if "headline" not in df.columns or "event_type" not in df.columns:
                logger.warning(
                    "GDELT parquet missing required columns (headline, event_type) — "
                    "falling back to synthetic only"
                )
            else:
                df = df[df["event_type"].isin(EVENT_TYPES)].dropna(subset=["headline"])
                for event_type in EVENT_TYPES:
                    subset = df[df["event_type"] == event_type]["headline"].tolist()
                    texts_by_class[event_type].extend(subset)
                logger.info(
                    "Loaded %d GDELT examples across %d classes",
                    len(df),
                    len(EVENT_TYPES),
                )
        except Exception as exc:
            logger.warning("Failed to load GDELT data: %s — using synthetic only", exc)
    elif not synthetic_only:
        logger.info(
            "GDELT parquet not found at %s — using synthetic only", _GDELT_PARQUET_PATH
        )

    # Oversample to target_per_class
    all_texts: list[str] = []
    all_labels: list[int] = []

    for event_type in EVENT_TYPES:
        class_texts = texts_by_class[event_type]
        if not class_texts:
            logger.warning("No examples for class %s — skipping", event_type)
            continue

        oversampled = _oversample_to_target(class_texts, target_per_class, rng)
        rng.shuffle(oversampled)
        all_texts.extend(oversampled)
        all_labels.extend([label2id[event_type]] * len(oversampled))

    logger.info(
        "Training corpus: %d examples across %d classes (~%d per class)",
        len(all_texts),
        len(EVENT_TYPES),
        target_per_class,
    )
    return all_texts, all_labels


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    texts: list[str],
    labels: list[int],
    output_dir: Path,
    epochs: int,
    batch_size: int,
) -> None:
    """
    Fine-tune deberta-v3-base on the financial event classification task.

    Uses HuggingFace Trainer with an 80/20 train/val split.  Evaluation
    strategy is epoch-based; the best checkpoint (by macro F1) is saved to
    ``output_dir``.  Per-class and macro F1 are reported at the end.
    """
    try:
        import numpy as np
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            DataCollatorWithPadding,
            EvalPrediction,
        )
        from sklearn.metrics import f1_score, classification_report
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Dataset
        import torch
    except ImportError as exc:
        logger.error("Missing dependency: %s — install transformers, torch, sklearn", exc)
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    num_labels = len(EVENT_TYPES)
    label2id = {label: i for i, label in enumerate(EVENT_TYPES)}
    id2label = {i: label for i, label in enumerate(EVENT_TYPES)}

    logger.info("Loading tokenizer and model: %s", _BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL_ID,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # ── Train/val split ────────────────────────────────────────────────────────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(
        "Split: %d train / %d val", len(train_texts), len(val_texts)
    )

    # ── Dataset wrapper ────────────────────────────────────────────────────────
    class FinEventDataset(Dataset):
        def __init__(self, encodings, label_list):
            self.encodings = encodings
            self.labels = label_list

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_enc = tokenizer(
        train_texts, truncation=True, max_length=256, padding=False
    )
    val_enc = tokenizer(
        val_texts, truncation=True, max_length=256, padding=False
    )

    train_dataset = FinEventDataset(train_enc, train_labels)
    val_dataset = FinEventDataset(val_enc, val_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Metrics ────────────────────────────────────────────────────────────────
    def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
        preds = np.argmax(pred.predictions, axis=1)
        macro_f1 = float(f1_score(pred.label_ids, preds, average="macro", zero_division=0))
        return {"f1_macro": macro_f1}

    # ── TrainingArguments ──────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=False,  # DeBERTa-v3 requires bf16 or fp32; fp16 can diverge
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    logger.info("Starting fine-tuning — %d epochs, batch_size=%d", epochs, batch_size)
    trainer.train()

    # ── Per-class F1 report ────────────────────────────────────────────────────
    logger.info("Evaluating on validation set…")
    eval_result = trainer.evaluate()
    logger.info("Validation results: %s", eval_result)

    raw_preds = trainer.predict(val_dataset)
    pred_ids = np.argmax(raw_preds.predictions, axis=1)
    report = classification_report(
        val_labels, pred_ids, target_names=EVENT_TYPES, zero_division=0
    )
    print("\nPer-class classification report (validation set):\n")
    print(report)

    macro_f1 = f1_score(val_labels, pred_ids, average="macro", zero_division=0)
    logger.info("Macro F1 on validation set: %.4f", macro_f1)

    # ── Save final model and tokenizer ────────────────────────────────────────
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model and tokenizer saved to %s", output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DeBERTa-v3-base for financial event classification"
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        default=False,
        help="Use only the built-in synthetic headlines (skip GDELT data).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device training batch size (default: 16).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/models/event_deberta"),
        help="Directory to save the trained model (default: data/models/event_deberta).",
    )
    args = parser.parse_args()

    logger.info(
        "Training config: synthetic_only=%s, epochs=%d, batch_size=%d, output_dir=%s",
        args.synthetic_only,
        args.epochs,
        args.batch_size,
        args.output_dir,
    )

    texts, labels = load_training_data(synthetic_only=args.synthetic_only)
    train(
        texts=texts,
        labels=labels,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
