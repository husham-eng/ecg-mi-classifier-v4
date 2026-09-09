"""
nearest_centroid_experiment_colab.py
=====================================
يقارن مصنّف "أقرب سنترويد" (Nearest Centroid) المقترح، القائم بالكامل على
مسافة إقليدية بعد Stemming بالتقاطع، بمصنّف Random Forest الحالي المُستخدَم
فعلياً بالتطبيق (ecg_pipeline/classifier.py) -- على نفس بيانات PTB-XL
المخزَّنة مسبقاً بالكاش، بنفس تقسيمات GroupKFold (على مستوى المريض، لا
النبضة، لتفادي تسريب بيانات نفس المريض بين التدريب والاختبار).

⚠️ إعادة بناء بعد جلسة سابقة (الملف الأصلي لم يُعثر عليه محفوظاً لدى
المستخدم) -- أُعيد كتابته من الصفر اعتماداً على: (أ) المواصفة الموثَّقة
بملف تسليم الجلسة السابقة (البند 3: "الموضوع المفتوح: مصنِّف أقرب سنترويد")،
و(ب) الكود الفعلي الحالي بالمشروع (ecg_pipeline/reference.py،
ecg_pipeline/classifier.py) لضمان مطابقة منطق بناء المدى الطبيعي والـStemming
الكلاسيكي المستخدم فعلياً بالإنتاج، حتى تكون المقارنة عادلة.

⚠️ تبسيط مقصود عن نسخة الإنتاج: هذي التجربة **لا تطبّق** ترجيح
IDF/الاختصاص الواعي بالموضع (ecg_pipeline/weighting.py) على أي من
الطريقتين -- كلتاهما تُقيَّمان على القيم الخام بعد Stemming مباشرة، لعزل
أثر "طريقة التصنيف نفسها" (شجرات القرار مقابل أقرب سنترويد) عن أثر
الترجيح. إذا تفوّق أقرب سنترويد هنا، الخطوة التالية الطبيعية هي إضافة نفس
الترجيح له قبل أي قرار نهائي بالدمج.

طريقة أقرب سنترويد (حسب الاتفاق):
  1. يُبنى مدى إحصائي (متوسط ± k×انحراف معياري) لكل فئة على حدة، بما فيها
     كل فئة مرضية (لا الطبيعي فقط كما بالنسخة الكلاسيكية).
  2. Stemming بالتقاطع: تُبقى فقط نقاط النبضة الواقعة *خارج* المدى الطبيعي
     *و* *داخل* مدى فئة مرضية واحدة على الأقل (أي نقطة أخرى تُصفَّر، بما
     فيها أي نقطة بعد cutoff الديناميكي الخاص بالنبضة).
  3. سنترويد كل فئة = متوسط شكل البقايا (بعد التقاطع) لكل نبضات تلك الفئة
     ببيانات التدريب -- بما فيها فئة "Normal" نفسها (نبضة طبيعية حقيقية
     من المفترض ألا تحقق شرط "داخل مدى فئة مرضية"، فسنترويدها يقترب من
     صفر تلقائياً بلا حاجة لأي معالجة خاصة أو استثناء بالكود).
  4. تصنيف أي نبضة جديدة = أقرب سنترويد (مسافة إقليدية) من بين كل الفئات
     معاً (Normal + كل فئة مرضية) -- بلا Random Forest إطلاقاً بهذي الخطوة.
  5. مسح شامل لعرض المدى k (0.5 إلى 2.5) لإيجاد الأفضل حسب Macro-F1، لكل
     قطب على حدة (الأقطاب تختلف بعدد فئاتها المدعومة -- راجع LEAD_CATEGORIES).

تشغيل على Google Colab:
  1. أولاً، بخلية منفصلة (⚠️ drive.mount() يفشل دائماً من داخل سكربت يعمل
     عبر !python -- يحتاج تواصلاً مباشراً مع المتصفح غير متاح لعملية
     فرعية):
         from google.colab import drive
         drive.mount('/content/drive')
  2. تأكد أن الكاش موجود فعلاً بالمسار المتوقَّع (عدّل CACHE_PATH أدناه لو
     كان بمسار مختلف عندك):
         /content/drive/MyDrive/ecg_project_cache/ptbxl_beats_cache.pkl
  3. ثم بخلية أخرى:
         !python nearest_centroid_experiment_colab.py
     (أو انسخ محتوى الملف مباشرة بخلية وشغّلها، نفس النتيجة).

المخرجات: طباعة ملخّص مباشر لكل قطب بالكونسول، بالإضافة لملف CSV كامل
بكل تركيبة (قطب × طريقة × k) بنفس مجلد الكاش:
  /content/drive/MyDrive/ecg_project_cache/nearest_centroid_vs_rf_comparison.csv
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

# ============================================================
# 0) إعدادات التجربة
# ============================================================
CACHE_PATH = Path("/content/drive/MyDrive/ecg_project_cache/ptbxl_beats_cache.pkl")
OUTPUT_CSV = CACHE_PATH.parent / "nearest_centroid_vs_rf_comparison.csv"

# نفس تعريف LEAD_CATEGORIES بـ retrain_final_models.py -- الفئات المرضية
# المدعومة لكل قطب (Normal مشترك دائماً بكل الأقطاب، يُضاف تلقائياً أدناه).
LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}

K_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

# k ثابت لمقارنة Random Forest -- القيمة الأكثر شيوعاً فعلياً بالإنتاج
# (راجع ecg_pipeline/classifier.py::LEAD_DEFAULT_K). ⚠️ ملاحظة منهجية
# معلَّقة من الجلسة السابقة: هذا k ثابت وليس ممسوحاً بالكامل مثل أقرب
# سنترويد -- إن أردت مسحاً عادلاً للاثنين معاً، مرِّر K_GRID بدل قيمة
# واحدة بحلقة RF أدناه (يضاعف وقت التشغيل تقريباً).
RF_FIXED_K = 1.25

N_SPLITS = 5          # GroupKFold على مستوى patient_id
TRIM_FRACTION = 0.1   # نفس القيمة الافتراضية بـ reference.build_normal_envelope
MIN_BEATS_PER_CLASS_PER_FOLD = 5  # تجاهل بناء مدى فئة إذا كانت بيانات التدريب أقل من هذا

# ⚠️ إضافة بعد فحص الكاش الفعلي بجلسة Colab: الكاش المتوفر عملياً بدرايف
# ("ptbxl_beats_cache.pkl") مرحلة أبكر ممّا كان مفترَضاً -- لا يحتوي عمودَي
# "beat"/"cutoff_idx" جاهزين، بل "beat_highpass_only" (قبل التصحيح المحلي
# لمقطع PR) و"rr_to_next_samples" (قبل تحويلها لنقطة قطع فعلية). القيم
# أدناه مؤكَّدة تجريبياً من نفس الكاش (تحقّق إحصائي: متوسط موضع قمة R عبر
# عيّنة 300 نبضة = بالضبط 100.0، وطول النبضة الكلي = 400) -- ومطابقة أيضاً
# للقيم الافتراضية بـ load_ptbxl_data.py (pre=100, post=300) بنفس المشروع.
PRE = 100
POST = 300
ISO_WINDOW_SAMPLES = (-50, -20)   # نفس افتراضي preprocessing.local_isoelectric_correct
SAFETY_MARGIN_SAMPLES = 30        # نفس افتراضي preprocessing.compute_dynamic_cutoff


def ensure_drive_mounted() -> None:
    if not Path("/content/drive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")


# ============================================================
# 0.5) تجهيز الكاش -- يكتشف تلقائياً أي مرحلة الكاش المتوفر ويشتق
#      beat/cutoff_idx إن لزم (راجع الملاحظة أعلى الملف عن PRE/POST)
# ============================================================
def local_isoelectric_correct(beat: np.ndarray, pre: int,
                               iso_start_samples: int = -50,
                               iso_end_samples: int = -20) -> np.ndarray:
    """مطابقة حرفياً لـ ecg_pipeline.preprocessing.local_isoelectric_correct
    الحالية بالإنتاج -- منقولة هنا كي يبقى هذا السكربت مستقلاً (بلا حاجة
    لرفع حزمة ecg_pipeline كاملة على Colab)."""
    lo, hi = pre + iso_start_samples, pre + iso_end_samples
    lo, hi = max(0, lo), min(len(beat), hi)
    if hi <= lo:
        return beat
    iso_ref = np.median(beat[lo:hi])
    return beat - iso_ref


def compute_dynamic_cutoff(rr_to_next: float, pre: int, post: int,
                            safety_margin_samples: int = 30) -> int:
    """مطابقة حرفياً لـ ecg_pipeline.preprocessing.compute_dynamic_cutoff."""
    length = pre + post
    if rr_to_next is None or (isinstance(rr_to_next, float) and np.isnan(rr_to_next)):
        return length
    valid_post = max(0, int(rr_to_next) - safety_margin_samples)
    return pre + min(post, valid_post)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    يتحقق من عمودَي beat/cutoff_idx الجاهزين؛ إن لم يكونا موجودين (حالة
    الكاش الفعلي المكتشَفة بجلسة التشغيل)، يشتقهما تلقائياً من
    beat_highpass_only + rr_to_next_samples بنفس منطق preprocessing.py
    الحقيقي بالإنتاج (تصحيح محلي لمقطع PR + نقطة قطع ديناميكية).
    """
    if {"beat", "cutoff_idx"}.issubset(df.columns):
        return df

    required_raw = {"beat_highpass_only", "rr_to_next_samples"}
    missing = required_raw - set(df.columns)
    if missing:
        raise KeyError(
            f"الكاش لا يحتوي أياً من مجموعتَي الأعمدة المتوقَّعتين "
            f"(beat/cutoff_idx) أو (beat_highpass_only/rr_to_next_samples). "
            f"الأعمدة الناقصة من المجموعة الثانية: {missing}. "
            f"الأعمدة الموجودة فعلياً: {list(df.columns)}"
        )

    print(f"ℹ️ الكاش بمرحلة مبكرة (قبل التصحيح المحلي/القطع الديناميكي) -- "
          f"يُشتق beat/cutoff_idx تلقائياً الآن (pre={PRE}, post={POST}, "
          f"iso_window={ISO_WINDOW_SAMPLES}). قد يستغرق هذا بضع دقائق.")

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
# 1) بناء المدى الإحصائي لأي فئة (طبيعية أو مرضية) -- مطابق تماماً
#    لـ ecg_pipeline.reference.build_normal_envelope الحالي بالإنتاج
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
    """يبني (ref_min, ref_max) لأي فئة -- الدالة عامة الآن (لا محصورة بالطبيعي
    فقط)، هذا هو التوسيع المطلوب بالبند 1 من المواصفة."""
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


# ============================================================
# 2) Stemming -- نسختان: الكلاسيكية (لـ Random Forest، كما بالإنتاج
#    فعلياً)، والتقاطع الجديدة (لأقرب سنترويد، حسب البند 2 بالمواصفة)
# ============================================================
def classic_stem_beat(beat: np.ndarray, cutoff: int,
                       ref_min: np.ndarray, ref_max: np.ndarray) -> np.ndarray:
    """مطابقة حرفياً لـ ecg_pipeline.reference.stem_beat الحالية بالإنتاج."""
    stemmed = beat.copy()
    stemmed[(stemmed >= ref_min) & (stemmed <= ref_max)] = 0
    cutoff = int(cutoff)
    if cutoff < len(stemmed):
        stemmed[cutoff:] = 0
    return stemmed


def stem_beat_with_class_envelopes(beat: np.ndarray, cutoff: int,
                                    normal_min: np.ndarray, normal_max: np.ndarray,
                                    class_envelopes: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Stemming بالتقاطع (البند 2 بالمواصفة): تُبقى فقط عينات النبضة الواقعة
    خارج المدى الطبيعي وداخل مدى فئة مرضية واحدة على الأقل (بغض النظر عن
    الفئة الحقيقية لهذي النبضة -- التقاطع يُبنى من كل الفئات المرضية معاً
    بلا معرفة مسبقة بأي فئة تخص النبضة الحالية، تماماً كحال أي نبضة جديدة
    وقت التصنيف الفعلي).

    class_envelopes: {اسم الفئة المرضية: (ref_min, ref_max)} -- الفئات
    المرضية فقط، لا الطبيعي (يُمرَّر بمعاملين منفصلين أعلاه).
    """
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


def is_likely_normal(stemmed_beat: np.ndarray, threshold: int = 50) -> bool:
    """مطابقة لـ ecg_pipeline.reference.is_likely_normal -- تُستخدَم فقط
    بمسار Random Forest (يحاكي منطق الإنتاج الفعلي بحرفية)."""
    return int(np.count_nonzero(stemmed_beat)) < threshold


# ============================================================
# 3) أقرب سنترويد (البنود 3 و4 بالمواصفة)
# ============================================================
def build_class_centroids(X_stemmed: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """سنترويد كل فئة = متوسط شكل البقايا بعد التقاطع -- بما فيها Normal،
    بلا أي معاملة خاصة (راجع الملاحظة بالبند 3 أعلى الملف)."""
    return {c: X_stemmed[y == c].mean(axis=0) for c in sorted(set(y))}


def nearest_centroid_predict_batch(X: np.ndarray, centroids: dict[str, np.ndarray]) -> np.ndarray:
    classes = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in classes])  # (n_classes, window_len)
    # مسافة إقليدية تربيعية لكل نبضة اختبار مقابل كل سنترويد دفعة واحدة
    dists = ((X[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)  # (n_beats, n_classes)
    best_idx = dists.argmin(axis=1)
    return np.array([classes[i] for i in best_idx])


# ============================================================
# 4) طيّة واحدة من كل طريقة (يُستدعى داخل حلقة GroupKFold)
# ============================================================
def run_random_forest_fold(beats_train, cutoffs_train, y_train,
                            beats_test, cutoffs_test, y_test, k: float) -> float:
    """يحاكي منطق الإنتاج الفعلي: مدى طبيعي واحد، RF مدرَّب على الفئات
    المرضية فقط، وأي نبضة اختبار تُصنَّف Normal مباشرة عبر is_likely_normal
    قبل حتى الوصول لـRF (تماماً كـ LeadModel.predict_beat الحالية)."""
    normal_mask_train = y_train == "Normal"
    ref_min, ref_max = build_class_envelope(beats_train[normal_mask_train], cutoffs_train[normal_mask_train], k=k)

    path_mask_train = ~normal_mask_train
    X_train_path = np.array([
        classic_stem_beat(b, c, ref_min, ref_max)
        for b, c in zip(beats_train[path_mask_train], cutoffs_train[path_mask_train])
    ])
    y_train_path = y_train[path_mask_train]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        max_features="sqrt", random_state=42, class_weight="balanced",
    )
    clf.fit(X_train_path, y_train_path)

    y_pred = []
    for b, c in zip(beats_test, cutoffs_test):
        stemmed = classic_stem_beat(b, c, ref_min, ref_max)
        if is_likely_normal(stemmed):
            y_pred.append("Normal")
        else:
            y_pred.append(clf.predict([stemmed])[0])

    return f1_score(y_test, np.array(y_pred), average="macro")


def run_nearest_centroid_fold(beats_train, cutoffs_train, y_train,
                               beats_test, cutoffs_test, y_test,
                               pathological_classes: list[str], k: float) -> float:
    normal_mask_train = y_train == "Normal"
    normal_min, normal_max = build_class_envelope(beats_train[normal_mask_train], cutoffs_train[normal_mask_train], k=k)

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

    centroids = build_class_centroids(X_train, y_train)
    y_pred = nearest_centroid_predict_batch(X_test, centroids)
    return f1_score(y_test, y_pred, average="macro")


# ============================================================
# 5) مقارنة كاملة لقطب واحد (كل الطيّات × كل قيم k)
# ============================================================
def run_lead_comparison(df_lead: pd.DataFrame, pathological_classes: list[str]) -> list[dict]:
    beats = np.stack(df_lead["beat"].values)
    cutoffs = df_lead["cutoff_idx"].values
    labels = df_lead["label"].values
    groups = df_lead["patient_id"].values
    lead_name = df_lead["lead"].iloc[0]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_splits = list(gkf.split(beats, labels, groups))
    results = []

    # --- Random Forest (خط الأساس الحالي بالإنتاج، k ثابت) ---
    rf_scores = [
        run_random_forest_fold(beats[tr], cutoffs[tr], labels[tr], beats[te], cutoffs[te], labels[te], RF_FIXED_K)
        for tr, te in fold_splits
    ]
    results.append({
        "lead": lead_name, "method": "RandomForest", "k": RF_FIXED_K,
        "macro_f1_mean": float(np.mean(rf_scores)), "macro_f1_std": float(np.std(rf_scores)),
        "n_folds": N_SPLITS,
    })

    # --- أقرب سنترويد (مسح كامل لـk) ---
    for k in K_GRID:
        nc_scores = [
            run_nearest_centroid_fold(beats[tr], cutoffs[tr], labels[tr], beats[te], cutoffs[te], labels[te],
                                       pathological_classes, k)
            for tr, te in fold_splits
        ]
        results.append({
            "lead": lead_name, "method": "NearestCentroid", "k": k,
            "macro_f1_mean": float(np.mean(nc_scores)), "macro_f1_std": float(np.std(nc_scores)),
            "n_folds": N_SPLITS,
        })

    return results


# ============================================================
# 6) التشغيل الكامل
# ============================================================
def main() -> None:
    ensure_drive_mounted()
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(
            f"لم يُعثر على الكاش المتوقَّع بالمسار: {CACHE_PATH}\n"
            "تحقق من اسم المجلد/الملف بدرايف، أو عدّل CACHE_PATH أعلى الملف."
        )

    df = pd.read_pickle(CACHE_PATH)
    required_base_cols = {"patient_id", "label", "lead"}
    missing_base = required_base_cols - set(df.columns)
    if missing_base:
        raise KeyError(f"الكاش لا يحتوي الأعمدة الأساسية المطلوبة: {missing_base}")

    df = prepare_dataframe(df)

    all_results = []
    for lead, pathological_classes in LEAD_CATEGORIES.items():
        print(f"=== قطب {lead} ===")
        sub = df[df["lead"] == lead].reset_index(drop=True)
        if sub.empty:
            print("  ⚠️ لا توجد بيانات لهذا القطب بالكاش -- تم تجاوزه.")
            continue

        lead_results = run_lead_comparison(sub, pathological_classes)
        all_results.extend(lead_results)

        rf_row = next(r for r in lead_results if r["method"] == "RandomForest")
        best_nc = max((r for r in lead_results if r["method"] == "NearestCentroid"),
                      key=lambda r: r["macro_f1_mean"])

        print(f"  عدد المرضى: {sub['patient_id'].nunique()} | عدد النبضات: {len(sub)}")
        print(f"  RandomForest (k={RF_FIXED_K}, ثابت): "
              f"Macro-F1 = {rf_row['macro_f1_mean']:.3f} ± {rf_row['macro_f1_std']:.3f}")
        print(f"  NearestCentroid الأفضل (k={best_nc['k']}, ممسوح بالكامل): "
              f"Macro-F1 = {best_nc['macro_f1_mean']:.3f} ± {best_nc['macro_f1_std']:.3f}")
        diff = best_nc["macro_f1_mean"] - rf_row["macro_f1_mean"]
        verdict = "✅ تفوّق أقرب سنترويد" if diff > 0 else "❌ Random Forest أفضل أو متعادل"
        print(f"  الفارق: {diff:+.3f}  →  {verdict}")
        print()

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ انتهت التجربة. النتائج الكاملة (كل قطب × كل طريقة × كل k) محفوظة بـ:\n   {OUTPUT_CSV}")
    print("\nملاحظة القراءة: قارن macro_f1_mean بين الطريقتين لكل قطب، وتأكد أن أي فارق أكبر "
          "فعلياً من macro_f1_std النموذجي (~0.03-0.06 حسب تجارب سابقة على نفس البيانات) قبل "
          "اعتباره تحسناً حقيقياً وليس تذبذباً عادياً بين الطيّات.")


if __name__ == "__main__":
    main()
