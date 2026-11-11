"""
Beauty & Wellness Data Science Internship — Task 1 Reference Implementation
Project: SkinMatch AI — Hybrid Ingredient-Compatibility & Aspect Sentiment Recommender
Dataset Schema: Modeled after Kaggle Sephora Products and Skincare Reviews Dataset
URL: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

This self-contained Python script (requiring only pandas and numpy) demonstrates
the exploratory data analysis (EDA), aspect-based sentiment extraction (ABSA),
and TF-IDF cosine-similarity recommendation pipeline proposed in the Task 1 Report.
"""

import math
import re
from collections import Counter
import numpy as np
import pandas as pd


def create_sample_beauty_dataset() -> pd.DataFrame:
    """Create a representative sample dataset matching the Sephora Skincare schema."""
    data = [
        {
            "product_id": "P101",
            "product_name": "Niacinamide 10% + Zinc 1% Oil Control Serum",
            "brand_name": "DermPure",
            "category": "Face Serums",
            "price_usd": 12.50,
            "ingredients": "Aqua, Niacinamide, Pentylene Glycol, Zinc PCA, Panthenol, Hyaluronic Acid",
            "target_skin_type": "Oily",
            "user_skin_type": "Oily",
            "rating": 5,
            "is_recommended": 1,
            "review_text": "Lightweight texture and cleared my acne breakouts within two weeks without irritation. Great price!",
        },
        {
            "product_id": "P102",
            "product_name": "Ceramide Barrier Repair Rich Moisture Cream",
            "brand_name": "HydraLab",
            "category": "Moisturizers",
            "price_usd": 38.00,
            "ingredients": "Aqua, Glycerin, Ceramide NP, Ceramide AP, Cholesterol, Hyaluronic Acid, Squalane",
            "target_skin_type": "Dry",
            "user_skin_type": "Dry",
            "rating": 5,
            "is_recommended": 1,
            "review_text": "Soothing and deeply hydrating for my flaky winter skin. Zero fragrance and no redness.",
        },
        {
            "product_id": "P103",
            "product_name": "Ultra-Rich Coconut & Shea Glow Balm",
            "brand_name": "GlowBotanics",
            "category": "Moisturizers",
            "price_usd": 45.00,
            "ingredients": "Cocos Nucifera Oil, Butyrospermum Parkii Butter, Parfum, Linalool, Limonene",
            "target_skin_type": "Dry",
            "user_skin_type": "Oily",
            "rating": 2,
            "is_recommended": 0,
            "review_text": "Heavy greasy texture clogged my pores and caused breakouts. Strong scent irritated my skin.",
        },
        {
            "product_id": "P104",
            "product_name": "Salicylic Acid 2% + Niacinamide Clarifying Toner",
            "brand_name": "ClearSkin Co",
            "category": "Toners",
            "price_usd": 29.00,
            "ingredients": "Aqua, Niacinamide, Salicylic Acid, Zinc PCA, Panthenol, Green Tea Extract",
            "target_skin_type": "Combination",
            "user_skin_type": "Oily",
            "rating": 4,
            "is_recommended": 1,
            "review_text": "Unclogged my pores and smoothed skin texture quickly, though packaging leaks slightly.",
        },
        {
            "product_id": "P105",
            "product_name": "Centella & Hyaluronic Calming Recovery Gel",
            "brand_name": "CicaCalm",
            "category": "Moisturizers",
            "price_usd": 32.00,
            "ingredients": "Aqua, Centella Asiatica Extract, Hyaluronic Acid, Panthenol, Madecassoside, Allantoin",
            "target_skin_type": "Sensitive",
            "user_skin_type": "Sensitive",
            "rating": 5,
            "is_recommended": 1,
            "review_text": "Calmed my redness and sensitivity overnight. Clean ingredients and absorbs fast.",
        },
    ]
    return pd.DataFrame(data)


ASPECT_KEYWORDS = {
    "effectiveness": ["cleared", "breakouts", "unclogged", "smoothed", "calmed", "hydrating", "clogged"],
    "texture": ["lightweight", "greasy", "heavy", "absorbs", "texture", "flaky"],
    "irritation_safety": ["irritation", "redness", "irritated", "sensitivity", "breakouts"],
    "scent": ["fragrance", "scent", "smell"],
    "price_packaging": ["price", "packaging", "leaks"],
}

POSITIVE_WORDS = {"lightweight", "cleared", "great", "soothing", "hydrating", "zero", "unclogged", "smoothed", "calmed", "clean", "fast"}
NEGATIVE_WORDS = {"heavy", "greasy", "clogged", "breakouts", "irritated", "leaks", "redness", "strong"}
IRRITANT_INGREDIENTS = {"parfum", "linalool", "limonene", "cocos nucifera oil"}


def analyze_review_sentiment(text: str) -> dict:
    """Compute overall polarity and aspect-level flags from customer review text."""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    pos_hits = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_hits = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total_hits = max(pos_hits + neg_hits, 1)
    sentiment_score = round((pos_hits - neg_hits) / total_hits, 3)

    detected_aspects = [
        aspect for aspect, kw_list in ASPECT_KEYWORDS.items()
        if any(kw in tokens for kw in kw_list)
    ]
    return {
        "sentiment_score": sentiment_score,
        "detected_aspects": ", ".join(detected_aspects) if detected_aspects else "general",
    }


def flag_comedogenic_or_irritant(ingredients_str: str) -> int:
    """Return count of common sensitizing or comedogenic ingredients in a formula."""
    ing_list = [i.strip().lower() for i in ingredients_str.split(",")]
    return sum(1 for ing in ing_list if ing in IRRITANT_INGREDIENTS)


def compute_tfidf_cosine_similarity(ingredient_series: pd.Series) -> np.ndarray:
    """Compute TF-IDF matrix and pairwise cosine similarity using NumPy."""
    tokenized_docs = [[ing.strip().lower() for ing in doc.split(",")] for doc in ingredient_series]
    vocab = sorted({token for doc in tokenized_docs for token in doc})
    vocab_index = {term: idx for idx, term in enumerate(vocab)}

    n_docs = len(tokenized_docs)
    n_terms = len(vocab)
    tf_matrix = np.zeros((n_docs, n_terms), dtype=float)

    for d_idx, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        for term, count in counts.items():
            tf_matrix[d_idx, vocab_index[term]] = count / max(len(doc), 1)

    df_counts = np.sum(tf_matrix > 0, axis=0)
    idf = np.log((1 + n_docs) / (1 + df_counts)) + 1.0
    tfidf = tf_matrix * idf

    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_tfidf = tfidf / norms
    return np.dot(normalized_tfidf, normalized_tfidf.T)


def run_exploratory_and_recommendation_pipeline():
    """Execute the EDA summary, ABSA scoring, and hybrid product recommendation demo."""
    df = create_sample_beauty_dataset()

    # 1. Feature Engineering: Irritant Flag & Skin-Type Match
    df["irritant_count"] = df["ingredients"].apply(flag_comedogenic_or_irritant)
    df["skin_type_match"] = (df["target_skin_type"] == df["user_skin_type"]).astype(int)

    # 2. Aspect-Based Sentiment Analysis
    sentiment_results = df["review_text"].apply(analyze_review_sentiment)
    df["sentiment_score"] = [r["sentiment_score"] for r in sentiment_results]
    df["aspects"] = [r["detected_aspects"] for r in sentiment_results]

    print("=== 1. Exploratory Summary: Skin-Type Match vs. Customer Rating ===")
    summary = df.groupby("skin_type_match")[["rating", "sentiment_score", "is_recommended"]].mean()
    print(summary.to_string())

    # 3. Content-Based Filtering on Ingredient TF-IDF Vectors
    sim_matrix = compute_tfidf_cosine_similarity(df["ingredients"])

    # 4. Hybrid Ranking Score combining Ingredient Similarity + Sentiment + Safety Penalty
    anchor_idx = 0  # Anchor product: P101 (Niacinamide Serum for Oily Skin)
    df["ingredient_similarity"] = np.round(sim_matrix[anchor_idx], 3)
    df["hybrid_rank_score"] = np.round(
        0.50 * df["ingredient_similarity"]
        + 0.35 * ((df["sentiment_score"] + 1) / 2)
        - 0.15 * df["irritant_count"],
        3,
    )

    print("\n=== 2. Hybrid Product Recommendations (Ranked by Compatibility & Sentiment) ===")
    cols = ["product_id", "product_name", "target_skin_type", "sentiment_score", "hybrid_rank_score"]
    print(df.sort_values(by="hybrid_rank_score", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    run_exploratory_and_recommendation_pipeline()