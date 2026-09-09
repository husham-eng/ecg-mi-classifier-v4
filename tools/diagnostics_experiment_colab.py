"""
diagnostics_experiment_colab.py
=================================
سكربت تشخيصي إضافي (امتداد لـclassifier_comparison_experiment_colab.py)
يُنتج ثلاثة أنواع من الأدلة الإحصائية الحقيقية المطلوبة للمقالة الإنجليزية
الموسَّعة -- لا يعيد حساب أي مقارنة Macro-F1 (تلك النتائج معروفة مسبقاً)،
فقط يُنتج تفاصيل تكميلية:

  1. نسبة التخفيض الفعلية بعد Stemming بالتقاطع (كم % من عيّنات كل نبضة
     تُصفَّر) -- لكل قطب، بـk=1.25 (نفس k الفائز لـRandom Forest بكل
     الأقطاب الأربعة حسب النتائج السابقة).
  2. مصفوفة ارتباك (Confusion Matrix) حقيقية لـRandom Forest -- الطريقة
     الفائزة -- من تنبؤات GroupKFold مجتمعة (Out-of-Fold): كل مريض يُتنبَّأ
     به مرة واحدة فقط (بطيّة لم يشارك بتدريبها)، فتُجمَّع كل التنبؤات بمصفوفة
     واحدة تمثّل كامل العيّنة بلا تحيّز أو تكرار.
  3. رسم توضيحي (Pipeline) لمراحل معالجة نبضة واحدة فعلية: خام (بعد تمرير
     عالٍ فقط) → بعد التصحيح المحلي لمقطع PR → بعد Stemming بالتقاطع --
     لقطب واحد ونبضة مرضية واحدة حقيقية، يوضح بصرياً "أين تذهب" أغلب
     عيّنات الإشارة الخام.

المخرجات (تُحفَظ كلها بمجلد الكاش على Drive):
  - diagnostics_stemming_reduction.csv       (نسبة التصفير لكل قطب)
  - diagnostics_confusion_matrix_<lead>.csv  (4 ملفات، واحد لكل قطب)
  - diagnostics_pipeline_example_<lead>.png  (رسم توضيحي لقطب واحد كمثال)

تشغيل: نفس خطوات السكربتات السابقة بالضبط (Drive مربوط أولاً بخلية منفصلة،
ثم !python diagnostics_experiment_colab.py).
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ============================================================
# إعدادات -- مطابقة لبقية السكربتات بنفس الجلسة
# ============================================================
CACHE_PATH = Path("/content/drive/MyDrive/ecg_project_cache/ptbxl_beats_cache.pkl")
OUT_DIR = CACHE_PATH.parent

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}

# k الفائز فعلياً لـRandom Forest بكل الأقطاب الأربعة (من نتائج التجربة
# السابقة -- classifier_comparison_experiment_colab.py) -- نستخدمه هنا
# مباشرة بدل إعادة مسح k من الصفر (توفير وقت، النتيجة معروفة مسبقاً).
BEST_K_RF = {"LeadI": 1.25, "aVR": 1.25, "V2": 1.25, "V6": 1.25}

N_SPLITS = 5
TRIM_FRACTION = 0.1
MIN_BEATS_PER_CLASS_PER_FOLD = 5
PRE = 100
POST = 300
ISO_WINDOW_SAMPLES = (-50, -20)
SAFETY_MARGIN_SAMPLES = 30


def ensure_drive_mounted() -> None:
    if not Path("/content/drive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")


# ============================================================
# تجهيز الكاش + بناء المدى + Stemming -- مطابقة تماماً للسكربتات السابقة
# ============================================================
def local_isoelectric_correct(beat, pre, iso_start_samples=-50, iso_end_samples=-20):
    lo, hi = pre + iso_start_samples, pre + iso_end_samples
    lo, hi = max(0, lo), min(len(beat), hi)
    if hi <= lo:
        return beat
    iso_ref = np.median(beat[lo:hi])
    return beat - iso_ref


def compute_dynamic_cutoff(rr_to_next, pre, post, safety_margin_samples=30):
    length = pre + post
    if rr_to_next is None or (isinstance(rr_to_next, float) and np.isnan(rr_to_next)):
        return length
    valid_post = max(0, int(rr_to_next) - safety_margin_samples)
    return pre + min(post, valid_post)


def prepare_dataframe(df):
    if {"beat", "cutoff_idx"}.issubset(df.columns):
        return df
    required_raw = {"beat_highpass_only", "rr_to_next_samples"}
    missing = required_raw - set(df.columns)
    if missing:
        raise KeyError(f"أعمدة ناقصة: {missing}. الأعمدة الموجودة: {list(df.columns)}")
    print(f"ℹ️ يُشتق beat/cutoff_idx تلقائياً (pre={PRE}, post={POST}).")
    df = df.copy()
    df["beat"] = [local_isoelectric_correct(np.asarray(b, dtype=float), PRE, *ISO_WINDOW_SAMPLES)
                  for b in df["beat_highpass_only"].values]
    df["cutoff_idx"] = np.array([compute_dynamic_cutoff(rr, PRE, POST, SAFETY_MARGIN_SAMPLES)
                                  for rr in df["rr_to_next_samples"].values], dtype=int)
    return df


def _build_masked_beats(beats, cutoffs, length):
    arr = np.asarray(beats, dtype=float)
    mask = np.zeros_like(arr, dtype=bool)
    for i, c in enumerate(cutoffs):
        c = int(c)
        if c < length:
            mask[i, c:] = True
    return np.ma.array(arr, mask=mask)


def build_class_envelope(beats, cutoffs, k, trim_fraction=TRIM_FRACTION):
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


def stem_beat_with_class_envelopes(beat, cutoff, normal_min, normal_max, class_envelopes):
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


# ============================================================
# 1) نسبة التخفيض بعد Stemming (وصفية، على كل بيانات القطب دفعة واحدة)
# ============================================================
def compute_reduction_stats(df_lead, pathological_classes, k):
    beats = np.stack(df_lead["beat"].values)
    cutoffs = df_lead["cutoff_idx"].values
    labels = df_lead["label"].values

    normal_mask = labels == "Normal"
    normal_min, normal_max = build_class_envelope(beats[normal_mask], cutoffs[normal_mask], k=k)
    class_envelopes = {}
    for c in pathological_classes:
        cls_mask = labels == c
        if cls_mask.sum() < MIN_BEATS_PER_CLASS_PER_FOLD:
            continue
        class_envelopes[c] = build_class_envelope(beats[cls_mask], cutoffs[cls_mask], k=k)

    stemmed = np.array([
        stem_beat_with_class_envelopes(b, c, normal_min, normal_max, class_envelopes)
        for b, c in zip(beats, cutoffs)
    ])
    total_samples = stemmed.size
    zero_samples = int((stemmed == 0).sum())
    reduction_pct = 100.0 * zero_samples / total_samples
    return reduction_pct, normal_min, normal_max, class_envelopes


# ============================================================
# 2) مصفوفة ارتباك Out-of-Fold لـRandom Forest
# ============================================================
def compute_oof_confusion_matrix(df_lead, pathological_classes, k):
    beats = np.stack(df_lead["beat"].values)
    cutoffs = df_lead["cutoff_idx"].values
    labels = df_lead["label"].values
    groups = df_lead["patient_id"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    y_true_all, y_pred_all = [], []

    for tr, te in gkf.split(beats, labels, groups):
        y_train, y_test = labels[tr], labels[te]
        normal_mask = y_train == "Normal"
        normal_min, normal_max = build_class_envelope(beats[tr][normal_mask], cutoffs[tr][normal_mask], k=k)
        class_envelopes = {}
        for c in pathological_classes:
            cls_mask = y_train == c
            if cls_mask.sum() < MIN_BEATS_PER_CLASS_PER_FOLD:
                continue
            class_envelopes[c] = build_class_envelope(beats[tr][cls_mask], cutoffs[tr][cls_mask], k=k)

        X_train = np.array([stem_beat_with_class_envelopes(b, c, normal_min, normal_max, class_envelopes)
                             for b, c in zip(beats[tr], cutoffs[tr])])
        X_test = np.array([stem_beat_with_class_envelopes(b, c, normal_min, normal_max, class_envelopes)
                            for b, c in zip(beats[te], cutoffs[te])])

        clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4,
                                      max_features="sqrt", random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    all_classes = sorted(set(y_true_all) | set(y_pred_all))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=all_classes)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in all_classes], columns=[f"pred_{c}" for c in all_classes])


# ============================================================
# 3) رسم توضيحي (Pipeline) لنبضة واحدة فعلية
# ============================================================
def plot_pipeline_example(df_lead, normal_min, normal_max, class_envelopes, lead_name, out_path):
    # نختار أول نبضة مرضية فعلية (لا Normal) لتوضيح التأثير بأوضح شكل
    path_rows = df_lead[df_lead["label"] != "Normal"]
    if path_rows.empty:
        return
    row = path_rows.iloc[0]
    beat_raw = np.asarray(row["beat_highpass_only"] if "beat_highpass_only" in df_lead.columns else row["beat"], dtype=float)
    beat_corrected = np.asarray(row["beat"], dtype=float)
    cutoff = int(row["cutoff_idx"])
    stemmed = stem_beat_with_class_envelopes(beat_corrected, cutoff, normal_min, normal_max, class_envelopes)

    t = np.arange(len(beat_corrected)) - PRE  # زمن نسبي لقمة R

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(t, beat_raw, color="tab:gray", linewidth=1.1)
    axes[0].set_title(f"{lead_name} — Stage 1: after high-pass baseline removal (raw beat)", fontsize=10)

    axes[1].fill_between(t, normal_min, normal_max, color="tab:blue", alpha=0.15, label="Normal envelope")
    axes[1].plot(t, beat_corrected, color="tab:orange", linewidth=1.2, label=f"Patient beat (class={row['label']})")
    if cutoff < len(beat_corrected):
        axes[1].axvspan(t[cutoff], t[-1], color="gray", alpha=0.15)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=0.7)
    axes[1].set_title("Stage 2: after local PR-segment isoelectric correction", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper right")

    n_nonzero = int(np.count_nonzero(stemmed))
    axes[2].plot(t, stemmed, color="tab:red", linewidth=1.2)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_title(f"Stage 3: after intersection stemming ({n_nonzero}/{len(stemmed)} samples kept, "
                       f"{100 * n_nonzero / len(stemmed):.1f}%)", fontsize=10)
    axes[2].set_xlabel("Time relative to R-peak (samples)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# التشغيل الكامل
# ============================================================
def main():
    ensure_drive_mounted()
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"لم يُعثر على الكاش: {CACHE_PATH}")

    df_raw = pd.read_pickle(CACHE_PATH)
    df = prepare_dataframe(df_raw)
    # نحتفظ بعمود beat_highpass_only الأصلي إن وُجد (لعرض المرحلة الخام بالرسم)
    if "beat_highpass_only" in df_raw.columns:
        df["beat_highpass_only"] = df_raw["beat_highpass_only"].values

    reduction_rows = []
    for lead, pathological_classes in LEAD_CATEGORIES.items():
        print(f"=== قطب {lead} ===")
        sub = df[df["lead"] == lead].reset_index(drop=True)

        # ⚠️ نفس تصحيح سكربت المقارنة الشاملة: نستبعد فئات غير مدعومة
        # بهذا القطب (مثل "A" على aVR/V2/V6) قبل أي حساب.
        supported_classes = {"Normal", *pathological_classes}
        n_before = len(sub)
        sub = sub[sub["label"].isin(supported_classes)].reset_index(drop=True)
        if len(sub) < n_before:
            print(f"  ℹ️ استُبعدت {n_before - len(sub)} نبضة بفئات غير مدعومة بهذا القطب "
                  f"(الفئات المدعومة: {sorted(supported_classes)})")

        k = BEST_K_RF[lead]

        reduction_pct, normal_min, normal_max, class_envelopes = compute_reduction_stats(sub, pathological_classes, k)
        reduction_rows.append({"lead": lead, "k": k, "reduction_pct": reduction_pct})
        print(f"  نسبة التصفير بعد Stemming (k={k}): {reduction_pct:.1f}%")

        cm_df = compute_oof_confusion_matrix(sub, pathological_classes, k)
        cm_path = OUT_DIR / f"diagnostics_confusion_matrix_{lead}.csv"
        cm_df.to_csv(cm_path)
        print(f"  مصفوفة الارتباك محفوظة بـ: {cm_path}")
        print(cm_df)

        plot_path = OUT_DIR / f"diagnostics_pipeline_example_{lead}.png"
        plot_pipeline_example(sub, normal_min, normal_max, class_envelopes, lead, plot_path)
        print(f"  رسم توضيحي محفوظ بـ: {plot_path}")
        print()

    reduction_df = pd.DataFrame(reduction_rows)
    reduction_csv = OUT_DIR / "diagnostics_stemming_reduction.csv"
    reduction_df.to_csv(reduction_csv, index=False)
    print(f"✅ انتهى التشخيص. ملخّص التخفيض محفوظ بـ: {reduction_csv}")
    print(reduction_df)


if __name__ == "__main__":
    main()
