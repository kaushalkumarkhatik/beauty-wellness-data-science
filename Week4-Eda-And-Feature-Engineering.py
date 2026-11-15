"""
Beauty & Wellness Data Science Internship — Week 4 Reference Implementation
Task: Multi-Dimensional Exploratory Data Analysis (EDA) & Domain Feature Engineering
Aligned with Weeks 1–3: Skincare Ingredient Compatibility & Aspect Sentiment Modeling

Public Datasets & Standards Referenced:
- Kaggle Sephora Products & Skincare Reviews: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
- EU CosIng Ingredient Nomenclature Database: https://single-market-economy.ec.europa.eu/sectors/cosmetics/cosmetic-ingredient-database_en
- U.S. FDA CAERS / openFDA Cosmetic Adverse Events API: https://open.fda.gov/apis/cosmetic/event/
- Open Beauty Facts API: https://wiki.openfoodfacts.org/API/OpenBeautyFacts
"""

import numpy as np
import pandas as pd


def load_preprocessed_week3_dataset() -> pd.DataFrame:
    """Load preprocessed skincare records representing the output of the Week 3 pipeline."""
    return pd.DataFrame([
        {
            "review_id": "REV-1001", "product_id": "SKU-01", "brand_name": "DermPure",
            "category": "Face Serums", "volume_ml": 30.0, "price_winsorized": 14.0,
            "user_skin_type": "Oily", "target_skin_type": "Oily", "rating": 5.0,
            "is_recommended": 1, "helpfulness_votes": 24, "active_efficacy_count": 3,
            "comedogenic_risk_score": 0, "pos_aspect_hits": 4, "neg_aspect_hits": 0,
            "fda_category_adverse_rate": 1.2,
        },
        {
            "review_id": "REV-1002", "product_id": "SKU-01", "brand_name": "DermPure",
            "category": "Face Serums", "volume_ml": 30.0, "price_winsorized": 14.0,
            "user_skin_type": "Oily", "target_skin_type": "Oily", "rating": 4.0,
            "is_recommended": 1, "helpfulness_votes": 11, "active_efficacy_count": 3,
            "comedogenic_risk_score": 0, "pos_aspect_hits": 3, "neg_aspect_hits": 0,
            "fda_category_adverse_rate": 1.2,
        },
        {
            "review_id": "REV-1003", "product_id": "SKU-02", "brand_name": "LuxeGold",
            "category": "Moisturizers", "volume_ml": 50.0, "price_winsorized": 65.0,
            "user_skin_type": "Oily", "target_skin_type": "Dry", "rating": 1.0,
            "is_recommended": 0, "helpfulness_votes": 92, "active_efficacy_count": 1,
            "comedogenic_risk_score": 6, "pos_aspect_hits": 0, "neg_aspect_hits": 4,
            "fda_category_adverse_rate": 14.6,
        },
        {
            "review_id": "REV-1004", "product_id": "SKU-03", "brand_name": "HydraLab",
            "category": "Moisturizers", "volume_ml": 50.0, "price_winsorized": 36.0,
            "user_skin_type": "Dry", "target_skin_type": "Dry", "rating": 5.0,
            "is_recommended": 1, "helpfulness_votes": 19, "active_efficacy_count": 3,
            "comedogenic_risk_score": 0, "pos_aspect_hits": 4, "neg_aspect_hits": 0,
            "fda_category_adverse_rate": 0.8,
        },
        {
            "review_id": "REV-1005", "product_id": "SKU-04", "brand_name": "ClearSkin",
            "category": "Toners", "volume_ml": 118.3, "price_winsorized": 28.0,
            "user_skin_type": "Combination", "target_skin_type": "Combination", "rating": 4.5,
            "is_recommended": 1, "helpfulness_votes": 8, "active_efficacy_count": 2,
            "comedogenic_risk_score": 0, "pos_aspect_hits": 3, "neg_aspect_hits": 1,
            "fda_category_adverse_rate": 1.5,
        },
        {
            "review_id": "REV-1006", "product_id": "SKU-05", "brand_name": "CicaCalm",
            "category": "Face Serums", "volume_ml": 30.0, "price_winsorized": 32.0,
            "user_skin_type": "Sensitive", "target_skin_type": "Sensitive", "rating": 5.0,
            "is_recommended": 1, "helpfulness_votes": 15, "active_efficacy_count": 3,
            "comedogenic_risk_score": 0, "pos_aspect_hits": 4, "neg_aspect_hits": 0,
            "fda_category_adverse_rate": 0.9,
        },
        {
            "review_id": "REV-1007", "product_id": "SKU-02", "brand_name": "LuxeGold",
            "category": "Moisturizers", "volume_ml": 50.0, "price_winsorized": 65.0,
            "user_skin_type": "Sensitive", "target_skin_type": "Dry", "rating": 2.0,
            "is_recommended": 0, "helpfulness_votes": 44, "active_efficacy_count": 1,
            "comedogenic_risk_score": 6, "pos_aspect_hits": 1, "neg_aspect_hits": 3,
            "fda_category_adverse_rate": 14.6,
        },
        {
            "review_id": "REV-1008", "product_id": "SKU-06", "brand_name": "BotaniGlow",
            "category": "Face Serums", "volume_ml": 30.0, "price_winsorized": 48.0,
            "user_skin_type": "Combination", "target_skin_type": "Oily", "rating": 3.0,
            "is_recommended": 0, "helpfulness_votes": 12, "active_efficacy_count": 2,
            "comedogenic_risk_score": 3, "pos_aspect_hits": 1, "neg_aspect_hits": 2,
            "fda_category_adverse_rate": 5.4,
        },
    ])


def engineer_beauty_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific interaction, ratio, NLP-weighted, and target-encoded features."""
    feat_df = df.copy()

    # 1. Binary Skin-Type Match Indicator
    feat_df["skin_type_match"] = (feat_df["user_skin_type"] == feat_df["target_skin_type"]).astype(int)

    # 2. Unit Value & Relative Category Price Tier Ratio
    feat_df["price_per_ml"] = np.round(feat_df["price_winsorized"] / feat_df["volume_ml"], 4)
    cat_median_ppml = feat_df.groupby("category")["price_per_ml"].transform("median")
    feat_df["relative_price_tier_ratio"] = np.round(feat_df["price_per_ml"] / cat_median_ppml, 3)

    # 3. Interaction Feature: Skin-Formula Compatibility Index (SFCI)
    feat_df["skin_formula_compatibility_index"] = np.round(
        (feat_df["skin_type_match"] * 2.0 + feat_df["active_efficacy_count"])
        - (0.8 * feat_df["comedogenic_risk_score"]),
        3,
    )

    # 4. Helpfulness-Weighted Net Aspect Sentiment Polarity (WASP)
    raw_polarity = (feat_df["pos_aspect_hits"] - feat_df["neg_aspect_hits"]) / np.maximum(
        feat_df["pos_aspect_hits"] + feat_df["neg_aspect_hits"], 1
    )
    feat_df["weighted_aspect_sentiment"] = np.round(raw_polarity * np.log1p(feat_df["helpfulness_votes"]), 3)

    # 5. Regulatory Clean Safety Composite Score (0 to 1 scale)
    feat_df["clean_safety_score"] = np.round(1.0 / (1.0 + 0.15 * feat_df["fda_category_adverse_rate"]), 3)

    # 6. Smoothed Target Encoding for Brand Reputation (m-estimate smoothing, m=2)
    global_mean = feat_df["rating"].mean()
    m = 2.0
    brand_stats = feat_df.groupby("brand_name")["rating"].agg(["count", "mean"])
    smoothed_brand = (brand_stats["count"] * brand_stats["mean"] + m * global_mean) / (brand_stats["count"] + m)
    feat_df["brand_reputation_encoded"] = feat_df["brand_name"].map(smoothed_brand).round(3)

    # 7. One-Hot Encoding for User Skin Type Cohort
    skin_dummies = pd.get_dummies(feat_df["user_skin_type"], prefix="skin", dtype=int)
    feat_df = pd.concat([feat_df, skin_dummies], axis=1)

    return feat_df


def run_multidimensional_eda(feat_df: pd.DataFrame):
    """Run statistical correlation analysis and cohort comparisons on engineered features."""
    print("=== 1. Engineered Features Preview ===")
    cols = [
        "review_id", "product_id", "price_per_ml", "relative_price_tier_ratio",
        "skin_formula_compatibility_index", "weighted_aspect_sentiment",
        "clean_safety_score", "brand_reputation_encoded",
    ]
    print(feat_df[cols].to_string(index=False))

    print("\n=== 2. Pearson & Spearman Correlation with Customer Rating ===")
    predictor_cols = [
        "skin_formula_compatibility_index",
        "weighted_aspect_sentiment",
        "clean_safety_score",
        "brand_reputation_encoded",
        "relative_price_tier_ratio",
    ]
    corr_table = pd.DataFrame({
        "Pearson_r_with_Rating": [feat_df[c].corr(feat_df["rating"], method="pearson") for c in predictor_cols],
        "Spearman_rho_with_Rating": [feat_df[c].corr(feat_df["rating"], method="spearman") for c in predictor_cols],
    }, index=predictor_cols)
    print(corr_table.round(3).to_string())

    print("\n=== 3. Cohort Segmentation: Matched vs. Mismatched Skin Compatibility ===")
    cohort = feat_df.groupby("skin_type_match")[
        ["rating", "skin_formula_compatibility_index", "weighted_aspect_sentiment", "clean_safety_score"]
    ].mean()
    print(cohort.round(3).to_string())


if __name__ == "__main__":
    base_df = load_preprocessed_week3_dataset()
    engineered_df = engineer_beauty_features(base_df)
    run_multidimensional_eda(engineered_df)