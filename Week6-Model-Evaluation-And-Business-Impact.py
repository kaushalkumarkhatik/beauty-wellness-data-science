"""
Beauty & Wellness Data Science Internship — Week 6 Capstone Implementation
Task: Model Evaluation Framework, ROC-AUC Analysis, Cohort Auditing & Business ROI Simulator
Aligned with Weeks 1–5: Skincare Ingredient Compatibility & Aspect Sentiment Personalization

This script evaluates the hybrid predictive model across technical ML metrics
(Accuracy, Precision, Recall, F1-Score, AUC-ROC, RMSE) and translates outcomes into
non-technical business KPIs (avoided product returns, repeat purchase lift, net ROI).
"""

import numpy as np
import pandas as pd


def generate_evaluation_benchmark_cohort() -> pd.DataFrame:
    """Create a 20-sample evaluation holdout set spanning all 5 skin biology cohorts."""
    return pd.DataFrame([
        {"eval_id": "E01", "skin_type": "Oily", "sfci": 5.0, "wasp": 3.22, "safety": 0.85, "y_true": 1, "pred_prob": 0.96, "y_reg_true": 5.0, "y_reg_pred": 4.82},
        {"eval_id": "E02", "skin_type": "Oily", "sfci": 4.8, "wasp": 2.49, "safety": 0.85, "y_true": 1, "pred_prob": 0.91, "y_reg_true": 4.5, "y_reg_pred": 4.55},
        {"eval_id": "E03", "skin_type": "Oily", "sfci": -3.8, "wasp": -4.53, "safety": 0.31, "y_true": 0, "pred_prob": 0.04, "y_reg_true": 1.0, "y_reg_pred": 1.25},
        {"eval_id": "E04", "skin_type": "Oily", "sfci": -2.5, "wasp": -3.10, "safety": 0.38, "y_true": 0, "pred_prob": 0.09, "y_reg_true": 1.5, "y_reg_pred": 1.68},
        {"eval_id": "E05", "skin_type": "Oily", "sfci": 1.2, "wasp": 0.45, "safety": 0.72, "y_true": 0, "pred_prob": 0.54, "y_reg_true": 3.0, "y_reg_pred": 3.35},
        {"eval_id": "E06", "skin_type": "Dry", "sfci": 5.0, "wasp": 3.00, "safety": 0.89, "y_true": 1, "pred_prob": 0.95, "y_reg_true": 5.0, "y_reg_pred": 4.88},
        {"eval_id": "E07", "skin_type": "Dry", "sfci": 4.2, "wasp": 1.95, "safety": 0.84, "y_true": 1, "pred_prob": 0.88, "y_reg_true": 4.5, "y_reg_pred": 4.40},
        {"eval_id": "E08", "skin_type": "Dry", "sfci": -2.2, "wasp": -2.10, "safety": 0.45, "y_true": 0, "pred_prob": 0.14, "y_reg_true": 2.0, "y_reg_pred": 2.15},
        {"eval_id": "E09", "skin_type": "Dry", "sfci": 3.6, "wasp": 1.40, "safety": 0.80, "y_true": 1, "pred_prob": 0.83, "y_reg_true": 4.2, "y_reg_pred": 4.10},
        {"eval_id": "E10", "skin_type": "Sensitive", "sfci": 5.0, "wasp": 2.77, "safety": 0.88, "y_true": 1, "pred_prob": 0.94, "y_reg_true": 4.9, "y_reg_pred": 4.78},
        {"eval_id": "E11", "skin_type": "Sensitive", "sfci": -3.8, "wasp": -1.90, "safety": 0.31, "y_true": 0, "pred_prob": 0.08, "y_reg_true": 2.0, "y_reg_pred": 1.85},
        {"eval_id": "E12", "skin_type": "Sensitive", "sfci": -1.8, "wasp": -2.40, "safety": 0.40, "y_true": 0, "pred_prob": 0.11, "y_reg_true": 1.8, "y_reg_pred": 1.95},
        {"eval_id": "E13", "skin_type": "Sensitive", "sfci": 4.1, "wasp": 2.10, "safety": 0.86, "y_true": 1, "pred_prob": 0.89, "y_reg_true": 4.6, "y_reg_pred": 4.50},
        {"eval_id": "E14", "skin_type": "Combination", "sfci": 4.0, "wasp": 1.10, "safety": 0.82, "y_true": 1, "pred_prob": 0.84, "y_reg_true": 4.5, "y_reg_pred": 4.32},
        {"eval_id": "E15", "skin_type": "Combination", "sfci": -0.4, "wasp": -0.86, "safety": 0.55, "y_true": 0, "pred_prob": 0.28, "y_reg_true": 2.8, "y_reg_pred": 2.65},
        {"eval_id": "E16", "skin_type": "Combination", "sfci": 3.5, "wasp": 1.50, "safety": 0.79, "y_true": 1, "pred_prob": 0.81, "y_reg_true": 4.3, "y_reg_pred": 4.18},
        {"eval_id": "E17", "skin_type": "Combination", "sfci": 0.8, "wasp": 0.20, "safety": 0.65, "y_true": 1, "pred_prob": 0.46, "y_reg_true": 3.8, "y_reg_pred": 3.52},
        {"eval_id": "E18", "skin_type": "Normal", "sfci": 4.5, "wasp": 2.30, "safety": 0.87, "y_true": 1, "pred_prob": 0.92, "y_reg_true": 4.7, "y_reg_pred": 4.60},
        {"eval_id": "E19", "skin_type": "Normal", "sfci": -1.5, "wasp": -1.40, "safety": 0.48, "y_true": 0, "pred_prob": 0.19, "y_reg_true": 2.3, "y_reg_pred": 2.42},
        {"eval_id": "E20", "skin_type": "Normal", "sfci": 3.9, "wasp": 1.75, "safety": 0.83, "y_true": 1, "pred_prob": 0.86, "y_reg_true": 4.4, "y_reg_pred": 4.30},
    ])


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUC-ROC) using rank-based Mann-Whitney U statistic."""
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5
    comparisons = [
        1.0 if p > n else (0.5 if p == n else 0.0)
        for p in pos_probs for n in neg_probs
    ]
    return float(np.mean(comparisons))


def evaluate_model_performance(df: pd.DataFrame, threshold: float = 0.50) -> dict:
    """Calculate technical classification, ranking, and regression metrics."""
    y_true = df["y_true"].values
    y_prob = df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
    auc_roc = compute_roc_auc(y_true, y_prob)

    rmse = float(np.sqrt(np.mean((df["y_reg_true"] - df["y_reg_pred"]) ** 2)))
    mae = float(np.mean(np.abs(df["y_reg_true"] - df["y_reg_pred"])))

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy_Pct": round(accuracy * 100, 1),
        "Precision_Pct": round(precision * 100, 1),
        "Recall_Pct": round(recall * 100, 1),
        "Specificity_Pct": round(specificity * 100, 1),
        "F1_Score_Pct": round(f1 * 100, 1),
        "AUC_ROC": round(auc_roc, 3),
        "Regression_RMSE": round(rmse, 3),
        "Regression_MAE": round(mae, 3),
    }


def simulate_business_impact(annual_orders: int = 100_000, avg_order_value: float = 38.0):
    """
    Translate technical model precision/recall improvements into non-technical executive KPIs:
    - Baseline (Star-Rating Filter Only): 22.0% mismatch return rate, 28.0% repeat purchase rate
    - Post-Deployment (Hybrid SFCI + WASP Model): 8.5% return rate, 41.5% repeat purchase rate
    """
    return_cost_per_order = 14.50
    baseline_return_rate = 0.220
    model_return_rate = 0.085

    avoided_returns = int(round(annual_orders * (baseline_return_rate - model_return_rate)))
    direct_cost_savings = avoided_returns * return_cost_per_order

    baseline_repeat_rate = 0.280
    model_repeat_rate = 0.415
    incremental_repeat_orders = int(round(annual_orders * (model_repeat_rate - baseline_repeat_rate)))
    incremental_revenue = incremental_repeat_orders * avg_order_value

    return {
        "Annual_Orders_Analyzed": annual_orders,
        "Avoided_Product_Returns_Units": avoided_returns,
        "Direct_Return_Cost_Savings_USD": round(direct_cost_savings, 2),
        "Incremental_Repeat_Orders": incremental_repeat_orders,
        "Incremental_Repeat_Revenue_USD": round(incremental_revenue, 2),
        "Total_Annual_Economic_Benefit_USD": round(direct_cost_savings + incremental_revenue, 2),
    }


if __name__ == "__main__":
    eval_df = generate_evaluation_benchmark_cohort()
    metrics = evaluate_model_performance(eval_df, threshold=0.50)

    print("=== 1. Holdout Confusion Matrix & Technical Performance Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== 2. Subgroup Fairness Audit Across Skin-Biology Cohorts ===")
    eval_df["y_pred"] = (eval_df["pred_prob"] >= 0.50).astype(int)
    eval_df["correct"] = (eval_df["y_true"] == eval_df["y_pred"]).astype(int)
    eval_df["abs_err"] = np.abs(eval_df["y_reg_true"] - eval_df["y_reg_pred"])
    cohort_audit = pd.DataFrame({
        "Samples": eval_df.groupby("skin_type")["eval_id"].count(),
        "Cohort_Accuracy_Pct": (eval_df.groupby("skin_type")["correct"].mean() * 100).round(1),
        "Mean_Abs_Rating_Err": eval_df.groupby("skin_type")["abs_err"].mean().round(3),
    })
    print(cohort_audit.to_string())

    print("\n=== 3. Executive Business Impact & ROI Simulation (Per 100,000 Orders) ===")
    roi = simulate_business_impact()
    for k, v in roi.items():
        print(f"  {k}: {v:,}" if isinstance(v, (int, float)) else f"  {k}: {v}")