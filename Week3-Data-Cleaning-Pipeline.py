"""
Beauty & Wellness Data Science Internship — Week 3 Reference Implementation
Task: End-to-End Reproducible Data Cleaning & Preprocessing Pipeline
Aligned with Weeks 1 & 2: Skincare Ingredient Compatibility & Aspect Sentiment Modeling

Public Datasets & Standards Referenced:
- Kaggle Sephora Products & Skincare Reviews: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
- EU CosIng Ingredient Nomenclature Database: https://single-market-economy.ec.europa.eu/sectors/cosmetics/cosmetic-ingredient-database_en
- U.S. FDA CAERS / openFDA Cosmetic Adverse Events API: https://open.fda.gov/apis/cosmetic/event/
- Open Beauty Facts API: https://wiki.openfoodfacts.org/API/OpenBeautyFacts
"""

import re
import unicodedata
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PipelineConfig:
    """Configuration parameters for deterministic, reproducible preprocessing."""
    random_seed: int = 42
    iqr_multiplier: float = 1.5
    winsor_limits: tuple[float, float] = (0.05, 0.95)
    min_review_tokens: int = 3
    inci_synonym_dict: dict = field(default_factory=lambda: {
        "water": "AQUA",
        "water/aqua": "AQUA",
        "water/aqua/eau": "AQUA",
        "aqua": "AQUA",
        "vitamin b3": "NIACINAMIDE",
        "niacinamide": "NIACINAMIDE",
        "cocos nucifera (coconut) oil": "COCOS NUCIFERA OIL",
        "coconut oil": "COCOS NUCIFERA OIL",
        "parfum/fragrance": "PARFUM",
        "fragrance": "PARFUM",
        "parfum": "PARFUM",
        "salicylic acid (bha)": "SALICYLIC ACID",
        "salicylic acid": "SALICYLIC ACID",
        "sodium hyaluronate": "HYALURONIC ACID",
        "hyaluronic acid": "HYALURONIC ACID",
    })


def generate_dirty_beauty_dataset() -> pd.DataFrame:
    """Simulate raw beauty e-commerce records containing typical data quality defects."""
    return pd.DataFrame([
        {
            "review_id": "REV-1001",
            "product_id": "SKU-01",
            "brand_name": "DermPure",
            "category": "Face Serums",
            "size_str": "1.0 oz",
            "price_usd": 14.00,
            "raw_ingredients": "Water/Aqua/Eau, Vitamin B3 (10%), Zinc PCA, Panthenol",
            "target_skin_type": "Oily",
            "user_skin_type": "oily ",
            "rating": 5.0,
            "helpfulness_votes": 18,
            "review_text": "<p>Lightweight serum! Did NOT make my skin greasy &amp; cleared breakouts fast.</p>",
        },
        {
            "review_id": "REV-1001",  # Exact duplicate record
            "product_id": "SKU-01",
            "brand_name": "DermPure",
            "category": "Face Serums",
            "size_str": "1.0 oz",
            "price_usd": 14.00,
            "raw_ingredients": "Water/Aqua/Eau, Vitamin B3 (10%), Zinc PCA, Panthenol",
            "target_skin_type": "Oily",
            "user_skin_type": "oily ",
            "rating": 5.0,
            "helpfulness_votes": 18,
            "review_text": "<p>Lightweight serum! Did NOT make my skin greasy &amp; cleared breakouts fast.</p>",
        },
        {
            "review_id": "REV-1002",
            "product_id": "SKU-01",
            "brand_name": "DermPure",
            "category": "Face Serums",
            "size_str": "30 ml",
            "price_usd": 14.00,
            "raw_ingredients": "Water/Aqua/Eau, Vitamin B3 (10%), Zinc PCA, Panthenol",
            "target_skin_type": "Oily",
            "user_skin_type": None,  # Missing user skin type
            "rating": 4.0,
            "helpfulness_votes": 6,
            "review_text": "No redness or irritation at all, absorbs quickly into pores.",
        },
        {
            "review_id": "REV-1003",
            "product_id": "SKU-02",
            "brand_name": "LuxeGold",
            "category": "Moisturizers",
            "size_str": "50 ml",
            "price_usd": 420.00,  # Extreme luxury price outlier
            "raw_ingredients": "Coconut Oil, Butyrospermum Parkii, Parfum/Fragrance, Limonene",
            "target_skin_type": "Dry",
            "user_skin_type": "Oily",
            "rating": 1.0,
            "helpfulness_votes": 95,
            "review_text": "<b>Terrible!</b> Heavy greasy balm clogged pores and caused severe breakouts.",
        },
        {
            "review_id": "REV-1004",
            "product_id": "SKU-03",
            "brand_name": "HydraLab",
            "category": "Moisturizers",
            "size_str": "50 ml",
            "price_usd": 36.00,
            "raw_ingredients": "Aqua, Glycerin, Ceramide NP, Sodium Hyaluronate, Squalane",
            "target_skin_type": "Dry",
            "user_skin_type": "DRY",
            "rating": 5.0,
            "helpfulness_votes": 12,
            "review_text": "Deeply hydrating barrier cream, not sticky and zero fragrance.",
        },
        {
            "review_id": "REV-1005",
            "product_id": "SKU-04",
            "brand_name": "ClearSkin",
            "category": "Toners",
            "size_str": "4.0 oz",
            "price_usd": 28.00,
            "raw_ingredients": "Water, Salicylic Acid (BHA), Niacinamide, Panthenol",
            "target_skin_type": "Combination",
            "user_skin_type": None,  # Missing user skin type
            "rating": None,  # Missing rating
            "helpfulness_votes": 4,
            "review_text": "Unclogged my pores within one week without drying my cheeks.",
        },
        {
            "review_id": "REV-1006",
            "product_id": "SKU-05",
            "brand_name": "CicaCalm",
            "category": "Face Serums",
            "size_str": "30 ml",
            "price_usd": 32.00,
            "raw_ingredients": "Aqua, Centella Asiatica, Hyaluronic Acid, Madecassoside",
            "target_skin_type": "Sensitive",
            "user_skin_type": "Sensitive",
            "rating": 5.0,
            "helpfulness_votes": 11,
            "review_text": "Calmed redness overnight! Never irritated my sensitive skin.",
        },
    ])


class BeautyDataCleaningPipeline:
    """Reproducible multi-stage data cleaning and normalization pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        np.random.seed(self.config.random_seed)
        self.audit_log = {}

    def _standardize_volume_ml(self, size_str: str) -> float:
        """Convert mixed fluid ounce ('oz') and milliliter ('ml') strings to numeric mL."""
        if not isinstance(size_str, str):
            return 30.0
        s = size_str.lower().strip()
        match = re.search(r"(\d+(?:\.\d+)?)", s)
        val = float(match.group(1)) if match else 30.0
        if "oz" in s:
            return round(val * 29.5735, 1)
        return round(val, 1)

    def _canonicalize_inci(self, raw_ing: str) -> str:
        """Standardize raw chemical lists to EU CosIng INCI tokens and strip concentration %."""
        if not isinstance(raw_ing, str):
            return ""
        tokens = [t.strip().lower() for t in raw_ing.split(",")]
        cleaned_tokens = []
        for tok in tokens:
            tok_no_pct = re.sub(r"\(\d+%?\)", "", tok).strip()
            canonical = self.config.inci_synonym_dict.get(tok_no_pct, tok_no_pct.upper())
            if canonical not in cleaned_tokens:
                cleaned_tokens.append(canonical)
        return ", ".join(cleaned_tokens)

    def _clean_review_text_negation_aware(self, text: str) -> str:
        """Strip HTML tags/entities and bind negation words ('not_greasy', 'no_redness')."""
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("&amp;", " and ").lower()
        text = re.sub(r"\b(not|no|never|without|zero)\s+([a-z]+)", r"\1_\2", text)
        tokens = re.findall(r"\b[a-z_]+\b", text)
        return " ".join(tokens)

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Execute the 6-stage cleaning, imputation, outlier treatment, and scaling pipeline."""
        df = raw_df.copy()
        self.audit_log["raw_row_count"] = len(df)
        self.audit_log["raw_missing_cells"] = int(df.isna().sum().sum())
        self.audit_log["duplicate_rows_removed"] = int(df.duplicated(subset=["review_id"]).sum())

        # Stage 1: Deduplication
        df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

        # Stage 2: String Casing & Unit Standardization (oz -> mL)
        df["target_skin_type"] = df["target_skin_type"].astype(str).str.strip().str.title()
        df["user_skin_type"] = df["user_skin_type"].apply(
            lambda x: x.strip().title() if isinstance(x, str) and x.strip() else np.nan
        )
        df["volume_ml"] = df["size_str"].apply(self._standardize_volume_ml)

        # Stage 3: Missing Value Imputation (Cohort Mode + Category/Global Median Rating + Flags)
        df["skin_type_imputed_flag"] = df["user_skin_type"].isna().astype(int)
        df["user_skin_type"] = df["user_skin_type"].fillna(df["target_skin_type"])
        df["rating_imputed_flag"] = df["rating"].isna().astype(int)
        df["rating"] = df["rating"].fillna(df.groupby("category")["rating"].transform("median")).fillna(df["rating"].median())

        # Stage 4: Outlier Detection (IQR Rule) & Winsorization + Log Transformation
        q1, q3 = df["price_usd"].quantile(0.25), df["price_usd"].quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + self.config.iqr_multiplier * iqr
        df["price_outlier_flag"] = (df["price_usd"] > upper_fence).astype(int)
        self.audit_log["price_outliers_detected"] = int(df["price_outlier_flag"].sum())

        p_low, _ = df["price_usd"].quantile(self.config.winsor_limits)
        df["price_winsorized"] = df["price_usd"].clip(lower=p_low, upper=upper_fence)
        df["price_log1p"] = np.round(np.log1p(df["price_winsorized"]), 4)

        # Stage 5: Feature Scaling (Min-Max [0,1] & Z-Score Standardization)
        p_min, p_max = df["price_winsorized"].min(), df["price_winsorized"].max()
        df["price_minmax"] = np.round((df["price_winsorized"] - p_min) / max(p_max - p_min, 1e-6), 4)
        h_mean, h_std = df["helpfulness_votes"].mean(), max(df["helpfulness_votes"].std(), 1e-6)
        df["helpfulness_zscore"] = np.round((df["helpfulness_votes"] - h_mean) / h_std, 4)

        # Stage 6: INCI Chemical Standardization & Negation-Aware NLP Cleaning
        df["standardized_inci"] = df["raw_ingredients"].apply(self._canonicalize_inci)
        df["clean_review_tokens"] = df["review_text"].apply(self._clean_review_text_negation_aware)

        self.audit_log["clean_row_count"] = len(df)
        self.audit_log["clean_missing_cells"] = int(df.isna().sum().sum())
        return df


if __name__ == "__main__":
    raw_data = generate_dirty_beauty_dataset()
    pipeline = BeautyDataCleaningPipeline(PipelineConfig())
    cleaned_data = pipeline.fit_transform(raw_data)

    print("=== 1. Pipeline Reproducibility & Quality Audit Log ===")
    for k, v in pipeline.audit_log.items():
        print(f"  {k}: {v}")

    print("\n=== 2. Cleaned & Normalized Beauty Dataset Preview ===")
    preview_cols = [
        "review_id", "user_skin_type", "skin_type_imputed_flag", "volume_ml",
        "price_usd", "price_winsorized", "price_minmax", "standardized_inci"
    ]
    print(cleaned_data[preview_cols].to_string(index=False))

    print("\n=== 3. Negation-Aware NLP Tokenization Sample ===")
    for _, row in cleaned_data[["review_id", "clean_review_tokens"]].iterrows():
        print(f"  [{row['review_id']}] -> {row['clean_review_tokens']}")