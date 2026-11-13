"""
Beauty & Wellness Data Science Internship — Week 2 Reference Implementation
Task: Data Identification, Multi-Source Ingestion, Preprocessing & Preliminary EDA
Aligned with Week 1 Problem: Reducing Skincare Mismatch via Ingredient & Sentiment Analysis

Public Data Sources Referenced:
1. Kaggle Sephora Products & Skincare Reviews: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
2. U.S. FDA CAERS / openFDA Cosmetic Adverse Events API: https://open.fda.gov/apis/cosmetic/event/
3. EU CosIng Cosmetic Ingredient Database: https://single-market-economy.ec.europa.eu/sectors/cosmetics/cosmetic-ingredient-database_en
4. Open Beauty Facts REST API: https://wiki.openfoodfacts.org/API/OpenBeautyFacts
5. UCSD McAuley Lab Amazon Beauty Reviews 2023: https://amazon-reviews-2023.github.io/
"""

import re
import numpy as np
import pandas as pd


def load_multi_source_sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate harmonized records from (1) E-Commerce Reviews (Sephora/Amazon),
    (2) EU CosIng Ingredient Function Registry, and (3) FDA CAERS Safety Logs.
    """
    reviews_df = pd.DataFrame([
        {"review_id": "R01", "product_id": "P101", "product_name": "Niacinamide 10% + Zinc 1% Serum", "price_usd": 12.5,
         "raw_ingredients": "Water/Aqua, Niacinamide, Pentylene Glycol, Zinc PCA, Panthenol",
         "target_skin_type": "Oily", "user_skin_type": "Oily", "rating": 5, "is_recommended": 1,
         "review_text": "Lightweight texture, cleared breakouts fast and zero redness!"},
        {"review_id": "R02", "product_id": "P101", "product_name": "Niacinamide 10% + Zinc 1% Serum", "price_usd": 12.5,
         "raw_ingredients": "Water/Aqua, Niacinamide, Pentylene Glycol, Zinc PCA, Panthenol",
         "target_skin_type": "Oily", "user_skin_type": None, "rating": 4, "is_recommended": 1,
         "review_text": "Great oil control for summer, absorbs well, dropper packaging is okay."},
        {"review_id": "R03", "product_id": "P102", "product_name": "Ceramide Barrier Repair Rich Cream", "price_usd": 38.0,
         "raw_ingredients": "Aqua, Glycerin, Ceramide NP, Hyaluronic Acid, Squalane",
         "target_skin_type": "Dry", "user_skin_type": "Dry", "rating": 5, "is_recommended": 1,
         "review_text": "Deeply hydrating for flaky dry skin, soothing and fragrance-free."},
        {"review_id": "R04", "product_id": "P103", "product_name": "Ultra-Rich Coconut & Shea Glow Balm", "price_usd": 45.0,
         "raw_ingredients": "Cocos Nucifera (Coconut) Oil, Butyrospermum Parkii, Parfum/Fragrance, Limonene",
         "target_skin_type": "Dry", "user_skin_type": "Oily", "rating": 1, "is_recommended": 0,
         "review_text": "Greasy heavy balm clogged my pores and caused severe acne breakouts and rash."},
        {"review_id": "R05", "product_id": "P103", "product_name": "Ultra-Rich Coconut & Shea Glow Balm", "price_usd": 45.0,
         "raw_ingredients": "Cocos Nucifera (Coconut) Oil, Butyrospermum Parkii, Parfum/Fragrance, Limonene",
         "target_skin_type": "Dry", "user_skin_type": "Sensitive", "rating": 2, "is_recommended": 0,
         "review_text": "Strong parfum scent triggered redness and itching on my cheeks."},
        {"review_id": "R05", "product_id": "P103", "product_name": "Ultra-Rich Coconut & Shea Glow Balm", "price_usd": 45.0,
         "raw_ingredients": "Cocos Nucifera (Coconut) Oil, Butyrospermum Parkii, Parfum/Fragrance, Limonene",
         "target_skin_type": "Dry", "user_skin_type": "Sensitive", "rating": 2, "is_recommended": 0,
         "review_text": "Strong parfum scent triggered redness and itching on my cheeks."},
        {"review_id": "R06", "product_id": "P104", "product_name": "Salicylic Acid 2% BHA Clarifying Toner", "price_usd": 29.0,
         "raw_ingredients": "Aqua, Salicylic Acid, Niacinamide, Camellia Sinensis Leaf Extract, Panthenol",
         "target_skin_type": "Combination", "user_skin_type": "Combination", "rating": 5, "is_recommended": 1,
         "review_text": "Unclogged pores and smoothed skin texture in ten days."},
        {"review_id": "R07", "product_id": "P105", "product_name": "Centella Calming Recovery Gel", "price_usd": 32.0,
         "raw_ingredients": "Water, Centella Asiatica Extract, Sodium Hyaluronate, Madecassoside, Allantoin",
         "target_skin_type": "Sensitive", "user_skin_type": "Sensitive", "rating": 5, "is_recommended": 1,
         "review_text": "Calmed my irritation and redness overnight, super clean lightweight gel."},
    ])

    cosing_registry = pd.DataFrame([
        {"inci_standard": "AQUA", "synonyms": ["water", "water/aqua", "aqua"], "function": "Solvent", "comedogenic_risk": 0},
        {"inci_standard": "NIACINAMIDE", "synonyms": ["niacinamide"], "function": "Skin Conditioning", "comedogenic_risk": 0},
        {"inci_standard": "SALICYLIC ACID", "synonyms": ["salicylic acid"], "function": "Keratolytic / Anti-Acne", "comedogenic_risk": 0},
        {"inci_standard": "CERAMIDE NP", "synonyms": ["ceramide np"], "function": "Barrier Conditioning", "comedogenic_risk": 0},
        {"inci_standard": "COCOS NUCIFERA OIL", "synonyms": ["cocos nucifera (coconut) oil", "cocos nucifera oil"], "function": "Occlusive Emollient", "comedogenic_risk": 4},
        {"inci_standard": "PARFUM", "synonyms": ["parfum", "parfum/fragrance", "fragrance"], "function": "Perfuming", "comedogenic_risk": 2},
        {"inci_standard": "LIMONENE", "synonyms": ["limonene"], "function": "Fragrance Allergen", "comedogenic_risk": 2},
    ])

    fda_caers_summary = pd.DataFrame([
        {"product_category": "Face Serums (Niacinamide/Zinc)", "adverse_event_rate_per_10k": 1.2, "top_symptom": "Mild Dryness"},
        {"product_category": "Barrier Moisturizers (Ceramide)", "adverse_event_rate_per_10k": 0.8, "top_symptom": "None"},
        {"product_category": "Fragranced Heavy Balms (Coconut/Parfum)", "adverse_event_rate_per_10k": 14.6, "top_symptom": "Contact Dermatitis / Acneiform Eruption"},
    ])

    return reviews_df, cosing_registry, fda_caers_summary


def audit_and_clean_dataset(df: pd.DataFrame, cosing_df: pd.DataFrame) -> pd.DataFrame:
    """Execute Week 2 Data Quality Audit and Preprocessing Pipeline."""
    print("=== 1. Data Quality Audit (Before Cleaning) ===")
    print(f"Total Raw Rows: {len(df)}")
    print(f"Exact Duplicate Rows: {df.duplicated(subset=['review_id']).sum()}")
    print(f"Missing 'user_skin_type' Count: {df['user_skin_type'].isna().sum()} ({df['user_skin_type'].isna().mean()*100:.1f}%)")

    cleaned = df.drop_duplicates(subset=["review_id"]).copy()
    cleaned["user_skin_type"] = cleaned["user_skin_type"].fillna(cleaned["target_skin_type"])

    synonym_map = {}
    risk_map = {}
    for _, row in cosing_df.iterrows():
        risk_map[row["inci_standard"]] = row["comedogenic_risk"]
        for syn in row["synonyms"]:
            synonym_map[syn.lower()] = row["inci_standard"]

    def normalize_inci_list(raw_str: str) -> tuple[str, int]:
        tokens = [t.strip().lower() for t in raw_str.split(",")]
        std_tokens = [synonym_map.get(t, t.upper()) for t in tokens]
        total_risk = sum(risk_map.get(std_t, 0) for std_t in std_tokens)
        return ", ".join(std_tokens), total_risk

    norm_results = cleaned["raw_ingredients"].apply(normalize_inci_list)
    cleaned["standardized_inci"] = [r[0] for r in norm_results]
    cleaned["comedogenic_risk_score"] = [r[1] for r in norm_results]
    cleaned["skin_type_match"] = (cleaned["target_skin_type"] == cleaned["user_skin_type"]).astype(int)
    cleaned["review_word_count"] = cleaned["review_text"].apply(lambda x: len(re.findall(r"\b\w+\b", x)))

    print("\n=== 2. Data Quality Audit (After Cleaning & INCI Normalization) ===")
    print(f"Cleaned Rows: {len(cleaned)} | Missing Values Remaining: {int(cleaned.isna().sum().sum())}")
    return cleaned


def run_preliminary_eda(cleaned_df: pd.DataFrame, fda_df: pd.DataFrame):
    """Compute descriptive statistics and hypothesis-testing indicators for Week 2."""
    print("\n=== 3. Preliminary Exploratory Data Analysis (Descriptive Statistics) ===")
    numeric_cols = ["price_usd", "rating", "comedogenic_risk_score", "review_word_count"]
    sub = cleaned_df[numeric_cols]
    stats = pd.DataFrame({
        "Mean": sub.mean(),
        "Std_Dev": sub.std(),
        "Median": sub.median(),
        "IQR": sub.quantile(0.75) - sub.quantile(0.25),
    }).T
    print(stats.round(2).to_string())

    print("\n=== 4. Hypothesis Testing Readiness: Skin-Type Match vs. Mismatch (Testing H1) ===")
    cohort_summary = cleaned_df.groupby("skin_type_match").agg(
        sample_count=("review_id", "count"),
        mean_rating=("rating", "mean"),
        recommendation_rate=("is_recommended", "mean"),
        mean_comedogenic_risk=("comedogenic_risk_score", "mean"),
    )
    print(cohort_summary.round(3).to_string())

    print("\n=== 5. Cross-Referenced FDA CAERS Safety Benchmarks (Testing H2) ===")
    print(fda_df.to_string(index=False))


if __name__ == "__main__":
    raw_reviews, cosing_ref, fda_ref = load_multi_source_sample_data()
    clean_reviews = audit_and_clean_dataset(raw_reviews, cosing_ref)
    run_preliminary_eda(clean_reviews, fda_ref)