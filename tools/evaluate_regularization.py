"""
evaluate_regularization.py
=============================
تقييم تجريبي صارم: هل تنظيم أشجار Random Forest (max_depth, min_samples_leaf)
يحسّن فعلياً Macro-F1 على مرضى لم يرهم النموذج؟

المنهجية: GroupKFold (5 طيّات) على مستوى patient_id لكل قطب على حدة —
بذلك لا يظهر أي مريض بالتدريب والاختبار معاً بنفس الطيّة (مطابق لمبدأ
"تقسيم صارم على مستوى المريض" المستخدم بالمشروع أصلاً). لكل طيّة:
  1. تدريب نموذج "أساسي" (إعدادات RF الأصلية بلا تقييد عمق) على مرضى التدريب
  2. تدريب نموذج "منظَّم" (max_depth=12, min_samples_leaf=4) على نفس مرضى التدريب
  3. تقييم الاثنين على نفس مرضى الاختبار (لم يرهما أي نموذج) عبر خط الاستدلال
     الكامل الفعلي (LeadModel.predict_beat، بما فيه قاعدة is_likely_normal)
  4. حساب Macro-F1 لكل نموذج على مستوى النبضة، ثم متوسط الطيّات الخمس
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_pipeline.reference import build_normal_envelope, stem_beat, is_likely_normal
from ecg_pipeline.weighting import weight_with_class_specificity, build_weight_lookup, weight_single_beat
from ecg_pipeline.classifier import LeadModel, _compute_idf_specificity

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}

BASELINE_RF_KWARGS = dict(n_estimators=300, random_state=42, class_weight="balanced")
REGULARIZED_RF_KWARGS = dict(n_estimators=300, random_state=42, class_weight="balanced",
                              max_depth=12, min_samples_leaf=4, max_features="sqrt")


def train_and_get_model(lead, cats, normal_beats, pathological_beats, rf_kwargs) -> LeadModel:
    window_len = normal_beats.shape[1]
    ref_min, ref_max = build_normal_envelope(normal_beats, k=2.5)

    X, y = [], []
    for cls, beats in pathological_beats.items():
        for b in beats:
            X.append(stem_beat(b, ref_min, ref_max))
            y.append(cls)
    X = np.array(X)
    y = np.array(y)

    idf_table, specificity_table = _compute_idf_specificity(X, y)
    X_weighted = weight_with_class_specificity(X, y)

    clf = RandomForestClassifier(**rf_kwargs)
    clf.fit(X_weighted, y)

    classes = sorted(set(y)) + ["Normal"]
    return LeadModel(lead, classes, window_len, ref_min, ref_max, idf_table, specificity_table, clf)


def evaluate_lead(df: pd.DataFrame, lead: str, cats: list[str], n_splits: int = 5, seed: int = 0):
    sub = df[df["lead"] == lead].copy()
    sub = sub[sub["label"].isin(cats + ["Normal"])]

    patients = sub["patient_id"].values
    X_idx = np.arange(len(sub))
    gkf = GroupKFold(n_splits=n_splits)

    baseline_f1s, regularized_f1s = [], []
    baseline_depths, regularized_depths = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_idx, groups=patients)):
        train_df = sub.iloc[train_idx]
        test_df = sub.iloc[test_idx]

        normal_train = np.stack(train_df[train_df["label"] == "Normal"]["beat"].values)
        path_train = {c: np.stack(train_df[train_df["label"] == c]["beat"].values)
                      for c in cats if (train_df["label"] == c).any()}

        model_base = train_and_get_model(lead, cats, normal_train, path_train, BASELINE_RF_KWARGS)
        model_reg = train_and_get_model(lead, cats, normal_train, path_train, REGULARIZED_RF_KWARGS)

        y_true, y_pred_base, y_pred_reg = [], [], []
        for _, row in test_df.iterrows():
            y_true.append(row["label"])
            y_pred_base.append(max(model_base.predict_beat(row["beat"]).items(), key=lambda kv: kv[1])[0])
            y_pred_reg.append(max(model_reg.predict_beat(row["beat"]).items(), key=lambda kv: kv[1])[0])

        labels_present = sorted(set(y_true) | set(y_pred_base) | set(y_pred_reg))
        f1_base = f1_score(y_true, y_pred_base, labels=labels_present, average="macro", zero_division=0)
        f1_reg = f1_score(y_true, y_pred_reg, labels=labels_present, average="macro", zero_division=0)
        baseline_f1s.append(f1_base)
        regularized_f1s.append(f1_reg)

        baseline_depths.append(np.mean([e.get_depth() for e in model_base.clf.estimators_]))
        regularized_depths.append(np.mean([e.get_depth() for e in model_reg.clf.estimators_]))

        print(f"  [{lead}] طيّة {fold+1}/{n_splits}: "
              f"مرضى تدريب={train_df['patient_id'].nunique()}, اختبار={test_df['patient_id'].nunique()} | "
              f"Macro-F1 أساسي={f1_base:.3f} (عمق={baseline_depths[-1]:.0f}) | "
              f"Macro-F1 منظَّم={f1_reg:.3f} (عمق={regularized_depths[-1]:.0f})")

    return {
        "lead": lead,
        "baseline_f1_mean": np.mean(baseline_f1s), "baseline_f1_std": np.std(baseline_f1s),
        "regularized_f1_mean": np.mean(regularized_f1s), "regularized_f1_std": np.std(regularized_f1s),
        "baseline_depth_mean": np.mean(baseline_depths),
        "regularized_depth_mean": np.mean(regularized_depths),
    }


if __name__ == "__main__":
    df = pd.read_pickle("/home/claude/ecg_app/ptbxl_beats.pkl")
    results = []
    for lead, cats in LEAD_CATEGORIES.items():
        print(f"\n=== تقييم قطب {lead} ===")
        res = evaluate_lead(df, lead, cats, n_splits=5)
        results.append(res)

    print("\n\n=== ملخص نهائي (متوسط 5 طيّات، مرضى مختلفين تماماً بكل طيّة) ===")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv("/home/claude/ecg_app/regularization_comparison.csv", index=False)
