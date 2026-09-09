"""
classifier_comparison_experiment_colab.py
============================================
مقارنة منهجية شاملة بين عدة عائلات تصنيف مختلفة، كلها على **نفس التغذية**
(نفس Stemming بالتقاطع، نفس مسح k، نفس بيانات PTB-XL المخزَّنة بالكاش)
ونفس منهجية التقييم (GroupKFold على مستوى المريض + Macro-F1) — امتداد مباشر
لتجربة nearest_centroid_experiment_colab.py السابقة (نتيجتها كانت سلبية:
أقرب سنترويد لم يتفوّق على Random Forest)، لكن هذي المرة بمقارنة أوسع بدل
طريقتين فقط، تمهيداً لمقالة مقارنة شاملة.

⚠️ الفكرة المنهجية الأساسية: **نفس تمثيل الميزة (Feature) لكل الطرق** —
النبضة بعد Stemming بالتقاطع (خارج المدى الطبيعي وداخل مدى فئة مرضية واحدة
على الأقل)، بنفس k لكل الطرق بكل تركيبة. هذا يعزل أثر "طريقة اتخاذ القرار
النهائي" (Nearest Centroid / Random Forest / شجرة قرار مفردة / SVM /
عناقيد K-Means / اختزال أبعاد + Logistic Regression) عن أثر أي فرق آخر
بالمعالجة -- بعكس التجربة السابقة اللي قارنت RF (Stemming كلاسيكي) بأقرب
سنترويد (Stemming بالتقاطع) كخط أساس مختلف قليلاً لكل طريقة.

الطرق الست المقارَنة (كلها على نفس X_stemmed):
  1. NearestCentroid    -- sklearn.neighbors.NearestCentroid (سنترويد واحد/فئة)
  2. RandomForest       -- نفس إعدادات ecg_pipeline/classifier.py
  3. DecisionTree       -- شجرة قرار مفردة (لعزل أثر التجميع/Ensemble نفسه)
  4. SVM (RBF)          -- شعاع الدعم الآلي، نواة RBF، class_weight=balanced
  5. ClusterPrototype   -- K-Means داخل كل فئة (عدة عناقيد/فئة)، التصنيف
                           بأقرب عنقود من بين كل عناقيد كل الفئات معاً --
                           تعميم "أقرب سنترويد" بعدة نماذج بدل واحد لكل فئة
  6. PCA+LogisticReg    -- اختزال أبعاد (PCA) قبل تصنيف خطي بسيط -- يفحص
                           هل "لعنة الأبعاد" (400 نقطة خام) جزء من المشكلة

⚠️ شبكة k مُصغَّرة عمداً (5 قيم بدل 10 بالتجربة السابقة) -- 6 طرق × 5 قيم
× 5 طيّات × 4 أقطاب = تشغيل أطول بكثير، والوقت محدود على Colab المجاني.

تشغيل على Google Colab: نفس خطوات nearest_centroid_experiment_colab.py
بالضبط (اربط Drive بخلية منفصلة أولاً، ثم !python classifier_comparison_experiment_colab.py).
يكتشف تلقائياً نفس نوع الكاش (beat/cutoff_idx جاهزين، أو beat_highpass_only
+ rr_to_next_samples يُشتق منهما تلقائياً -- مطابق لما اكتُشِف بالجلسة السابقة).
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# تحذير غير ضار متوقَّع دائماً هنا (بعض مواضع النبضة تكون صفراً بكل عيّنات
# فئة معيّنة بعد Stemming -- طبيعي تماماً بهذي البيانات المتفرقة، ليس خطأً)
warnings.filterwarnings("ignore", message=".*within_class_std_dev_.*")

# ============================================================
# 0) إعدادات التجربة
# ============================================================
CACHE_PATH = Path("/content/drive/MyDrive/ecg_project_cache/ptbxl_beats_cache.pkl")
OUTPUT_CSV = CACHE_PATH.parent / "classifier_comparison_results.csv"

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}

K_GRID = [0.75, 1.25, 1.75, 2.25, 2.5]   # مُصغَّرة عمداً -- راجع الملاحظة أعلى الملف
N_SPLITS = 5
TRIM_FRACTION = 0.1
MIN_BEATS_PER_CLASS_PER_FOLD = 5
N_CLUSTERS_PER_CLASS = 3   # لطريقة ClusterPrototype -- عدد العناقيد داخل كل فئة
PCA_VARIANCE_RETAINED = 0.95  # لطريقة PCA+LogisticReg -- نسبة التباين المُبقاة

# نفس القيم المؤكَّدة تجريبياً بالجلسة السابقة (كاش beat_highpass_only)
PRE = 100
POST = 300
ISO_WINDOW_SAMPLES = (-50, -20)
SAFETY_MARGIN_SAMPLES = 30


def ensure_drive_mounted() -> None:
    if not Path("/content/drive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")


# ============================================================
# 1) تجهيز الكاش -- مطابق تماماً لـ nearest_centroid_experiment_colab.py
# ============================================================
def local_isoelectric_correct(beat: np.ndarray, pre: int,
                               iso_start_samples: int = -50,
                               iso_end_samples: int = -20) -> np.ndarray:
    lo, hi = pre + iso_start_samples, pre + iso_end_samples
    lo, hi = max(0, lo), min(len(beat), hi)
    if hi <= lo:
        return beat
    iso_ref = np.median(beat[lo:hi])
    return beat - iso_ref


def compute_dynamic_cutoff(rr_to_next: float, pre: int, post: int,
                            safety_margin_samples: int = 30) -> int:
    length = pre + post
    if rr_to_next is None or (isinstance(rr_to_next, float) and np.isnan(rr_to_next)):
        return length
    valid_post = max(0, int(rr_to_next) - safety_margin_samples)
    return pre + min(post, valid_post)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if {"beat", "cutoff_idx"}.issubset(df.columns):
        return df

    required_raw = {"beat_highpass_only", "rr_to_next_samples"}
    missing = required_raw - set(df.columns)
    if missing:
        raise KeyError(
            f"الكاش لا يحتوي أياً من مجموعتَي الأعمدة المتوقَّعتين. "
            f"الأعمدة الناقصة من (beat_highpass_only, rr_to_next_samples): {missing}. "
            f"الأعمدة الموجودة فعلياً: {list(df.columns)}"
        )
    print(f"ℹ️ الكاش بمرحلة مبكرة -- يُشتق beat/cutoff_idx تلقائياً (pre={PRE}, post={POST}).")
    beats_corrected = [
        local_isoelectric_correct(np.asarray(b, dtype=float), PRE, *ISO_WINDOW_SAMPLES)
        for b in df["beat_highpass_only"].values
    ]
    cutoffs = [
        compute_dynamic_cutoff(rr, PRE, POST, SAFETY_MARGIN_SAMPLES)
        for rr in df["rr_to_next_samples"].values
    ]
    df = df.copy()
    df["beat"] = beats_corrected
    df["cutoff_idx"] = np.array(cutoffs, dtype=int)
    return df


# ============================================================
# 2) بناء المدى + Stemming بالتقاطع -- مطابق للتجربة السابقة
# ============================================================
def _build_masked_beats(beats: np.ndarray, cutoffs: np.ndarray, length: int) -> np.ma.MaskedArray:
    arr = np.asarray(beats, dtype=float)
    mask = np.zeros_like(arr, dtype=bool)
    for i, c in enumerate(cutoffs):
        c = int(c)
        if c < length:
            mask[i, c:] = True
    return np.ma.array(arr, mask=mask)


def build_class_envelope(beats: np.ndarray, cutoffs: np.ndarray, k: float,
                          trim_fraction: float = TRIM_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    beats = np.asarray(beats, dtype=float)
    cutoffs = np.asarray(cutoffs, dtype=int)
    n, length = beats.shape
    if trim_fraction > 0 and n >= 10:
        masked = _build_masked_beats(beats, cutoffs, length)
        pilot_median = np.ma.median(masked, axis=0)
        distances = np.ma.sum((masked - pilot_median) ** 2, axis=1).filled(np.inf)
        n_keep = max(int(n * (1 - trim_fraction)), n - 1)
        keep_idx = np.argsort(distances)[:n_keep]
        beats, cutoffs = beats[keep_idx], cutoffs[keep_idx]
    masked = _build_masked_beats(beats, cutoffs, length)
    mu = np.ma.mean(masked, axis=0).filled(0.0)
    sigma = np.ma.std(masked, axis=0).filled(0.0)
    return mu - k * sigma, mu + k * sigma


def stem_beat_with_class_envelopes(beat: np.ndarray, cutoff: int,
                                    normal_min: np.ndarray, normal_max: np.ndarray,
                                    class_envelopes: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    beat = np.asarray(beat, dtype=float)
    length = len(beat)
    outside_normal = (beat < normal_min) | (beat > normal_max)
    inside_any_pathological = np.zeros(length, dtype=bool)
    for cmin, cmax in class_envelopes.values():
        inside_any_pathological |= (beat >= cmin) & (beat <= cmax)
    keep_mask = outside_normal & inside_any_pathological
    stemmed = np.where(keep_mask, beat, 0.0)
    cutoff = int(cutoff)
    if cutoff < length:
        stemmed[cutoff:] = 0.0
    return stemmed


def build_stemmed_features(beats_train, cutoffs_train, y_train, beats_test, cutoffs_test, k: float,
                            pathological_classes: list[str]):
    """يبني X_train/X_test بعد Stemming بالتقاطع، بنفس k -- تُستخدَم كتغذية
    مشتركة موحّدة لكل الطرق الست (هذا هو أساس عدالة المقارنة)."""
    normal_mask = y_train == "Normal"
    normal_min, normal_max = build_class_envelope(beats_train[normal_mask], cutoffs_train[normal_mask], k=k)

    class_envelopes = {}
    for c in pathological_classes:
        cls_mask = y_train == c
        if cls_mask.sum() < MIN_BEATS_PER_CLASS_PER_FOLD:
            continue
        class_envelopes[c] = build_class_envelope(beats_train[cls_mask], cutoffs_train[cls_mask], k=k)

    X_train = np.array([
        stem_beat_with_class_envelopes(b, c, normal_min, normal_max, class_envelopes)
        for b, c in zip(beats_train, cutoffs_train)
    ])
    X_test = np.array([
        stem_beat_with_class_envelopes(b, c, normal_min, normal_max, class_envelopes)
        for b, c in zip(beats_test, cutoffs_test)
    ])
    return X_train, X_test


# ============================================================
# 3) طريقة "التصنيف القائم على عنقود" (ClusterPrototype) -- تعميم أقرب
#    سنترويد: عدة عناقيد K-Means داخل كل فئة بدل سنترويد واحد فقط
# ============================================================
def fit_predict_cluster_prototype(X_train, y_train, X_test, n_clusters_per_class: int) -> np.ndarray:
    prototypes = []   # كل عنصر: (متجه العنقود، اسم الفئة)
    for c in sorted(set(y_train)):
        X_c = X_train[y_train == c]
        n_clusters = min(n_clusters_per_class, len(X_c))
        if n_clusters < 1:
            continue
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        km.fit(X_c)
        for center in km.cluster_centers_:
            prototypes.append((center, c))

    centers = np.stack([p[0] for p in prototypes])
    labels = np.array([p[1] for p in prototypes])
    dists = ((X_test[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    best_idx = dists.argmin(axis=1)
    return labels[best_idx]


# ============================================================
# 4) طريقة "اختزال الأبعاد" (PCA + Logistic Regression)
# ============================================================
def fit_predict_pca_logreg(X_train, y_train, X_test, variance_retained: float) -> np.ndarray:
    pca = PCA(n_components=variance_retained, random_state=42)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_reduced, y_train)
    return clf.predict(X_test_reduced)


# ============================================================
# 5) طيّة واحدة: تبني التغذية المشتركة مرة واحدة بكل k، ثم تُشغِّل كل
#    الطرق الست على نفس التغذية بلا إعادة بنائها لكل طريقة
# ============================================================
METHOD_NAMES = ["NearestCentroid", "RandomForest", "DecisionTree", "SVM_RBF",
                "ClusterPrototype", "PCA_LogisticRegression"]


def run_fold_all_methods(beats_train, cutoffs_train, y_train,
                          beats_test, cutoffs_test, y_test,
                          pathological_classes: list[str], k: float) -> dict[str, float]:
    X_train, X_test = build_stemmed_features(beats_train, cutoffs_train, y_train,
                                              beats_test, cutoffs_test, k, pathological_classes)
    scores = {}

    nc = NearestCentroid()
    nc.fit(X_train, y_train)
    scores["NearestCentroid"] = f1_score(y_test, nc.predict(X_test), average="macro")

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4,
                                 max_features="sqrt", random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    scores["RandomForest"] = f1_score(y_test, rf.predict(X_test), average="macro")

    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=4, random_state=42, class_weight="balanced")
    dt.fit(X_train, y_train)
    scores["DecisionTree"] = f1_score(y_test, dt.predict(X_test), average="macro")

    svm = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)
    scores["SVM_RBF"] = f1_score(y_test, svm.predict(X_test), average="macro")

    y_pred_cluster = fit_predict_cluster_prototype(X_train, y_train, X_test, N_CLUSTERS_PER_CLASS)
    scores["ClusterPrototype"] = f1_score(y_test, y_pred_cluster, average="macro")

    y_pred_pca = fit_predict_pca_logreg(X_train, y_train, X_test, PCA_VARIANCE_RETAINED)
    scores["PCA_LogisticRegression"] = f1_score(y_test, y_pred_pca, average="macro")

    return scores


# ============================================================
# 6) مقارنة كاملة لقطب واحد (كل الطيّات × كل قيم k × كل الطرق الست)
# ============================================================
def run_lead_comparison(df_lead: pd.DataFrame, pathological_classes: list[str]) -> list[dict]:
    # ملاحظة: df_lead يصل هنا مُفلتَراً مسبقاً (من main()) ليشمل فقط الفئات
    # المدعومة فعلياً بهذا القطب (Normal + pathological_classes) -- تصحيح
    # لمشكلة اكتُشِفت لاحقاً: بيانات المريض نفسه موجودة بكل الأقطاب الأربعة
    # معاً (تسجيل 12-قطب متزامن)، فبدون هذا الفلتر تتسرّب فئات غير مدعومة
    # (مثل "A" على aVR/V2/V6) للتدريب/الاختبار بلا أي envelope مميِّز لها.
    beats = np.stack(df_lead["beat"].values)
    cutoffs = df_lead["cutoff_idx"].values
    labels = df_lead["label"].values
    groups = df_lead["patient_id"].values
    lead_name = df_lead["lead"].iloc[0]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_splits = list(gkf.split(beats, labels, groups))

    # {method: {k: [scores عبر الطيّات]}}
    all_scores = {m: {k: [] for k in K_GRID} for m in METHOD_NAMES}

    for fold_i, (tr, te) in enumerate(fold_splits, 1):
        for k in K_GRID:
            fold_scores = run_fold_all_methods(
                beats[tr], cutoffs[tr], labels[tr], beats[te], cutoffs[te], labels[te],
                pathological_classes, k,
            )
            for method, score in fold_scores.items():
                all_scores[method][k].append(score)
        print(f"    طيّة {fold_i}/{N_SPLITS} اكتملت")

    results = []
    for method in METHOD_NAMES:
        for k in K_GRID:
            scores = all_scores[method][k]
            results.append({
                "lead": lead_name, "method": method, "k": k,
                "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)),
                "n_folds": N_SPLITS,
            })
    return results


# ============================================================
# 7) التشغيل الكامل
# ============================================================
def main() -> None:
    ensure_drive_mounted()
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"لم يُعثر على الكاش المتوقَّع: {CACHE_PATH}")

    df = pd.read_pickle(CACHE_PATH)
    required_base_cols = {"patient_id", "label", "lead"}
    missing_base = required_base_cols - set(df.columns)
    if missing_base:
        raise KeyError(f"الكاش لا يحتوي الأعمدة الأساسية المطلوبة: {missing_base}")
    df = prepare_dataframe(df)

    all_results = []
    for lead, pathological_classes in LEAD_CATEGORIES.items():
        print(f"=== قطب {lead} (6 طرق × {len(K_GRID)} قيم k × {N_SPLITS} طيّات) ===")
        sub = df[df["lead"] == lead].reset_index(drop=True)
        if sub.empty:
            print("  ⚠️ لا توجد بيانات لهذا القطب -- تم تجاوزه.")
            continue

        supported_classes = {"Normal", *pathological_classes}
        n_before = len(sub)
        sub = sub[sub["label"].isin(supported_classes)].reset_index(drop=True)
        if len(sub) < n_before:
            print(f"  ℹ️ استُبعدت {n_before - len(sub)} نبضة بفئات غير مدعومة بهذا القطب "
                  f"(الفئات المدعومة: {sorted(supported_classes)})")

        lead_results = run_lead_comparison(sub, pathological_classes)
        all_results.extend(lead_results)

        print(f"  عدد المرضى: {sub['patient_id'].nunique()} | عدد النبضات: {len(sub)}")
        print("  أفضل نتيجة لكل طريقة (عبر كل قيم k):")
        for method in METHOD_NAMES:
            method_rows = [r for r in lead_results if r["method"] == method]
            best = max(method_rows, key=lambda r: r["macro_f1_mean"])
            print(f"    {method:25s}: Macro-F1 = {best['macro_f1_mean']:.3f} ± {best['macro_f1_std']:.3f} (k={best['k']})")
        print()

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ انتهت التجربة. النتائج الكاملة (كل قطب × كل طريقة × كل k) محفوظة بـ:\n   {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
