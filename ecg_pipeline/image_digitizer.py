"""
ecg_image_digitizer.py
=========================
نموذج أولي لتحويل صورة ورقة ECG (مصوّرة بكاميرا الجوال أو ممسوحة) إلى
إشارة رقمية مستمرة قابلة لتمريرها لخط الأنابيب الحالي (إزالة ضوضاء →
كشف R → تقطيع → Stemming → تصنيف).

الخطوات المطبّقة بهذا النموذج الأولي:
  1. تحويل لتدرج رمادي + تحسين تباين بسيط
  2. عزل الأثر الأسود (أغمق بكثير من الشبكة والخلفية) بعتبة سطوع
  3. لكل عمود بكسل: إيجاد موضع أغمق نقطة (مركز الكتلة الغامقة) = قيمة الإشارة
  4. سد الفجوات (أعمدة بدون أثر واضح) بالاستيفاء الخطي
  5. عكس المحور Y (لأن صف البكسل يزيد لأسفل، والقيمة الفيزيولوجية تزيد لأعلى)

⚠️ هذا النموذج لسة ما يشمل: كشف نبضة المعايرة لتحويل بكسل->mV/ثانية
فعلي (نخرج الإشارة بوحدات بكسل نسبية حالياً)، ولا معالجة الميلان أو
تحديد تخطيط الأقطاب المتعدد. هذي خطوات تالية بعد التحقق من نجاح
الاستخراج الأساسي.
"""

import cv2
import numpy as np


def extract_trace_from_image(image_path: str, dark_threshold: int = 100,
                              min_dark_pixels: int = 1,
                              refine_window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    يستخرج الأثر الأسود من صورة ECG كإشارة 1D، على مرحلتين:
      المرحلة 1: تقدير أولي خام (كل عمود بمفرده) — قد يلتقط عناصر
                 غريبة (رموز، ظلال، حواف الصورة) بعيدة عن المسار الحقيقي.
      المرحلة 2: تنعيم المسار الأولي (median filter) ثم إعادة البحث
                 بكل عمود ضمن نافذة ضيقة حول موضع المسار المتوقع فقط —
                 هذا يستبعد أي بقعة غامقة بعيدة عن الأثر الفعلي.

    Returns
    -------
    signal : قيمة الإشارة لكل عمود (بوحدات بكسل، محور Y معكوس)
    valid_mask : True للأعمدة ذات أثر واضح فعلاً (بعد التنقيح)
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if refine_window is None:
        refine_window = max(15, h // 3)  # يتكيّف تلقائياً مع طول/دقة الصورة

    def pass_extract(search_top, search_bottom):
        sig = np.full(w, np.nan)
        mask = np.zeros(w, dtype=bool)
        for x in range(w):
            col = gray[search_top[x]:search_bottom[x], x]
            dark_rows = np.where(col < dark_threshold)[0]
            if len(dark_rows) >= min_dark_pixels:
                weights = (dark_threshold - col[dark_rows]).astype(float)
                y_center = np.average(dark_rows, weights=weights) + search_top[x]
                sig[x] = y_center
                mask[x] = True
        return sig, mask

    # --- المرحلة 1: بحث بكامل ارتفاع الصورة ---
    top0 = np.zeros(w, dtype=int)
    bottom0 = np.full(w, h, dtype=int)
    raw_signal, raw_mask = pass_extract(top0, bottom0)

    valid_idx = np.where(raw_mask)[0]
    if len(valid_idx) < 2:
        raise ValueError("لم يُعثر على أثر واضح كافٍ بالصورة — تحقق من عتبة dark_threshold")
    all_idx = np.arange(w)
    raw_filled = np.interp(all_idx, valid_idx, raw_signal[valid_idx])

    # تنعيم قوي (median filter) لتقدير "خط اتجاه" المسار العام بدون تأثير النتوءات الغريبة
    from scipy.signal import medfilt
    smooth_estimate = medfilt(raw_filled, kernel_size=101)

    # --- المرحلة 2: إعادة البحث بنافذة ضيقة حول التقدير الممهّد ---
    top1 = np.clip(smooth_estimate - refine_window, 0, h - 1).astype(int)
    bottom1 = np.clip(smooth_estimate + refine_window, 1, h).astype(int)
    signal, valid_mask = pass_extract(top1, bottom1)

    valid_idx2 = np.where(valid_mask)[0]
    if len(valid_idx2) < 2:
        signal_filled = raw_filled
    else:
        signal_filled = np.interp(all_idx, valid_idx2, signal[valid_idx2])

    signal_flipped = -signal_filled
    return signal_flipped, valid_mask


def find_content_region(image_path: str, dark_threshold: int = 145) -> tuple[int, int]:
    """
    يحدد تلقائياً نطاق الصفوف (row range) اللي فيها فعلاً محتوى (شبكة+أثر)،
    ويستبعد الهوامش الفارغة/المحروقة بالإضاءة أعلى وأسفل الصورة.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_has_content = (gray < dark_threshold).sum(axis=1)
    threshold_count = gray.shape[1] * 0.02  # على الأقل 2% من عرض الصف فيه بكسلات غامقة
    rows_with_content = np.where(row_has_content > threshold_count)[0]
    if len(rows_with_content) == 0:
        return 0, gray.shape[0]
    return int(rows_with_content.min()), int(rows_with_content.max())


# ============================================================
# المعايرة: بكسل -> mV / ثانية حقيقية (بافتراض معيار ورق ECG القياسي)
# ============================================================

STANDARD_PAPER_SPEED_MM_S = 25.0   # مم/ثانية (المعيار الأشهر سريرياً)
STANDARD_GAIN_MM_MV = 10.0         # مم لكل mV (المعيار الأشهر سريرياً)
SMALL_SQUARE_MM = 1.0              # المربع الصغير بورق ECG = 1مم دايماً


def detect_grid_spacing(image_path: str, axis: str = "x") -> int | None:
    """
    يكتشف تباعد الشبكة الصغيرة (بالبكسل) عبر تحليل قناة 'الاحمرار'
    (تبرز خطوط الشبكة الوردية/الحمراء عن الأسود والأبيض) والارتباط
    الذاتي (autocorrelation) لإيجاد التكرار الدوري.
    """
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img.astype(int))
    redness = r - (g + b) / 2

    profile = redness.mean(axis=0) if axis == "x" else redness.mean(axis=1)
    profile_centered = profile - profile.mean()
    autocorr = np.correlate(profile_centered, profile_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(autocorr, distance=3)
    if len(peaks) == 0:
        return None
    best_peak = peaks[np.argmax(autocorr[peaks[:5]])] if len(peaks) >= 5 else peaks[0]
    return int(best_peak)


def calibrate_signal(pixel_signal: np.ndarray, image_path: str,
                      target_fs: float = 500.0,
                      paper_speed_mm_s: float = STANDARD_PAPER_SPEED_MM_S,
                      gain_mm_mv: float = STANDARD_GAIN_MM_MV,
                      auto_correct_via_heart_rate: bool = True) -> tuple[np.ndarray, dict]:
    """
    يحوّل الإشارة من وحدات بكسل نسبية إلى mV حقيقية على معدل عينات موحّد.

    ⚠️ اكتشفنا عملياً إن الصور التجارية/التوضيحية (Stock Images) غالباً
    غير دقيقة مترولوجياً حتى لو مكتوب عليها "25mm/s" — كشف الشبكة وحده
    غير كافٍ. لذلك نضيف تصحيحاً تلقائياً: إذا نتج عن المعايرة الأولية
    معدل قلب غير منطقي فسيولوجياً (خارج 40-180 نبضة/دقيقة)، نبحث عن
    أفضل عامل تصحيح (من مضاعفات شائعة: 1x, 2x, 5x, 10x) يعيد معدل
    القلب لمدى معقول. هذا يعتمد على معرفة مسبقة (معدل القلب البشري
    الطبيعي) بدل الاعتماد الأعمى على تفسير حرفي لبكسلات الشبكة.
    """
    px_per_mm_x = detect_grid_spacing(image_path, "x") / SMALL_SQUARE_MM
    px_per_mm_y = detect_grid_spacing(image_path, "y") / SMALL_SQUARE_MM

    if px_per_mm_x is None or px_per_mm_y is None:
        raise ValueError("تعذّر اكتشاف تباعد الشبكة — لا يمكن المعايرة الفعلية")

    seconds_per_pixel = 1.0 / (px_per_mm_x * paper_speed_mm_s)
    mv_per_pixel = 1.0 / (px_per_mm_y * gain_mm_mv)

    correction = 1.0
    if auto_correct_via_heart_rate:
        from scipy.signal import find_peaks
        centered = pixel_signal - np.median(pixel_signal)
        TYPICAL_RESTING_HR = 75.0  # نقطة مرجعية لاختيار أقرب تصحيح، لا "أول معقول فقط"
        best_candidate, best_distance = 1.0, float("inf")
        for candidate in [1, 2, 5, 10]:
            test_sec_per_px = seconds_per_pixel * candidate
            test_fs_equiv = 1.0 / test_sec_per_px
            thresh = 0.5 * np.percentile(np.abs(centered), 99)
            peaks, _ = find_peaks(np.abs(centered), height=thresh,
                                   distance=max(1, int(0.3 * test_fs_equiv)))
            if len(peaks) > 1:
                rr = np.diff(peaks) * test_sec_per_px
                hr = 60 / rr.mean()
                if 40 <= hr <= 180:
                    # لا نتوقف عند أول عامل "معقول تقنياً" (قد يكون هذا خطأً
                    # صامتاً كما ثبت عملياً: عامل=1 أعطى 157.5 نبضة/دقيقة وهو
                    # "معقول" حسب المدى الواسع لكنه غير واقعي مقارنة بعامل
                    # أدق أعطى 78.8 نبضة/دقيقة). نقيّم كل العوامل ونختار
                    # الأقرب لمعدل استراحة نموذجي.
                    distance = abs(hr - TYPICAL_RESTING_HR)
                    if distance < best_distance:
                        best_candidate, best_distance = candidate, distance
        correction = best_candidate

    seconds_per_pixel *= correction
    mv_per_pixel *= correction  # نفترض نفس عامل التصحيح ينطبق على المحورين (شبكة مربعة عادة)

    total_seconds = len(pixel_signal) * seconds_per_pixel
    mv_signal = pixel_signal * mv_per_pixel
    mv_signal = mv_signal - np.median(mv_signal)

    n_target_samples = int(round(total_seconds * target_fs))
    original_t = np.linspace(0, total_seconds, len(mv_signal))
    target_t = np.linspace(0, total_seconds, n_target_samples)
    resampled = np.interp(target_t, original_t, mv_signal)

    meta = {
        "px_per_mm_x": px_per_mm_x, "px_per_mm_y": px_per_mm_y,
        "total_seconds": total_seconds, "target_fs": target_fs,
        "assumed_paper_speed_mm_s": paper_speed_mm_s,
        "assumed_gain_mm_mv": gain_mm_mv,
        "auto_correction_factor": correction,
    }
    return resampled, meta
