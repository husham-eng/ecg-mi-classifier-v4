"""
evaluate_full_pipeline.py
============================
يقيّم بشكل متراكم كل التحسينات المقترحة على بيانات PTB-XL المُحاذاة
حديثاً (Lead II كمرجع موحّد لمواقع R):

  A) الأساس القديم: Stemming بمدى طبيعي واحد فقط + RF غير منظَّم
  B) + تنظيم RF (max_depth=12, min_samples_leaf=4)
  C) + Stemming متعدد الفئات (مدى لكل فئة مرضية مع استبعاد شواذ، يُبقي
     فقط الانحرافات المطابقة لنمط مرضي معروف فعلياً)

كل هذا عبر GroupKFold (5 طيّات) صارم على مستوى المريض، لكل قطب على حدة.
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
from ecg_pipeline.reference import (build_class_envelope, stem_beat,
                                     stem_beat_with_class_envelopes)
from ecg_pipeline.weighting import weight_with_class_specificity, build_weight_lookup, weight_single_beat
from ecg_pipeline.classifier import _compute_idf_specificity

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}

REGULARIZED_RF_KWARGS = dict(n_estimators=300, random_state=42, class_weight="balanced",
                              max_depth=12, min_samples_leaf=4, max_features="sqrt")
BASELINE_RF_KWARGS = dict(n_estimators=300, random_state=42, class_weight="balanced")


class SimpleModel:
    """نسخة مبسّطة عن LeadModel لأغراض التجربة فقط (تدعم أسلوبي Stemming)."""

    def __init__(self, ref_min, ref_max, class_ranges, idf_lookup, spec_lookup, clf, use_class_envelopes):
        self.ref_min, self.ref_max = ref_min, ref_max
        self.class_ranges = class_ranges
        self.idf_lookup, self.spec_lookup = idf_lookup, spec_lookup
        self.clf = clf
        self.use_class_envelopes = use_class_envelopes

    def _stem(self, beat):
        if self.use_class_envelopes:
            return stem_beat_with_class_envelopes(beat, (self.ref_min, self.ref_max), self.class_ranges)
        return stem_beat(beat, self.ref_min, self.ref_max)

    def predict(self, beat):
        stemmed = self._stem(beat)
        if int(np.count_nonzero(np.round(stemmed, 1))) < 50:
            return "Normal"
        weighted = weight_single_beat(stemmed, self.idf_lookup, self.spec_lookup)
        return self.clf.predict([weighted])[0]


def build_model(normal_beats, pathological_beats, rf_kwargs, use_class_envelopes, trim_fraction=0.1):
    ref_min, ref_max = build_class_envelope(normal_beats, k=2.5, trim_fraction=trim_fraction)

    class_ranges = {}
    if use_class_envelopes:
        for cls, beats in pathological_beats.items():
            class_ranges[cls] = build_class_envelope(beats, k=2.5, trim_fraction=trim_fraction)

    def stem(b):
        if use_class_envelopes:
            return stem_beat_with_class_envelopes(b, (ref_min, ref_max), class_ranges)
        return stem_beat(b, ref_min, ref_max)

    X, y = [], []
    for cls, beats in pathological_beats.items():
        for b in beats:
            X.append(stem(b))
            y.append(cls)
    X = np.array(X)
    y = np.array(y)

    idf_table, spec_table = _compute_idf_specificity(X, y)
    X_weighted = weight_with_class_specificity(X, y)

    clf = RandomForestClassifier(**rf_kwargs)
    clf.fit(X_weighted, y)

    return SimpleModel(ref_min, ref_max, class_ranges,
                        build_weight_lookup(idf_table), build_weight_lookup(spec_table),
                        clf, use_class_envelopes)


def evaluate_lead(df, lead, cats, n_splits=5):
    sub = df[df["lead"] == lead].copy()
    sub = sub[sub["label"].isin(cats + ["Normal"])]
    patients = sub["patient_id"].values
    gkf = GroupKFold(n_splits=n_splits)

    configs = {
        "A_baseline": (BASELINE_RF_KWARGS, False),
        "B_regularized": (REGULARIZED_RF_KWARGS, False),
        "C_regularized_classenv": (REGULARIZED_RF_KWARGS, True),
    }
    fold_scores = {name: [] for name in configs}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.arange(len(sub)), groups=patients)):
        train_df = sub.iloc[train_idx]
        test_df = sub.iloc[test_idx]
        normal_train = np.stack(train_df[train_df["label"] == "Normal"]["beat"].values)
        path_train = {c: np.stack(train_df[train_df["label"] == c]["beat"].values)
                      for c in cats if (train_df["label"] == c).any()}

        models = {name: build_model(normal_train, path_train, rf_kwargs, use_ce)
                  for name, (rf_kwargs, use_ce) in configs.items()}

        y_true = test_df["label"].tolist()
        for name, model in models.items():
            y_pred = [model.predict(b) for b in test_df["beat"]]
            labels_present = sorted(set(y_true) | set(y_pred))
            f1 = f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)
            fold_scores[name].append(f1)

        print(f"  [{lead}] طيّة {fold+1}/{n_splits}: " +
              " | ".join(f"{name}={fold_scores[name][-1]:.3f}" for name in configs))

    return {"lead": lead, **{f"{name}_mean": np.mean(v) for name, v in fold_scores.items()},
            **{f"{name}_std": np.std(v) for name, v in fold_scores.items()}}


if __name__ == "__main__":
    df = pd.read_pickle("/home/claude/ecg_app/ptbxl_beats.pkl")
    results = []
    for lead, cats in LEAD_CATEGORIES.items():
        print(f"\n=== تقييم قطب {lead} (بيانات مُحاذاة بـLead II) ===")
        results.append(evaluate_lead(df, lead, cats))

    print("\n\n=== الملخص النهائي ===")
    summary = pd.DataFrame(results)
    cols = ["lead", "A_baseline_mean", "B_regularized_mean", "C_regularized_classenv_mean"]
    print(summary[cols].to_string(index=False))
    summary.to_csv("/home/claude/ecg_app/full_pipeline_comparison.csv", index=False)
