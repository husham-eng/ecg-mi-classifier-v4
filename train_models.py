"""
train_models.py
==================
يدرّب النماذج النهائية (نموذج مستقل لكل قطب مدعوم) من كل البيانات
الأرشيفية المتاحة (تدريب + اختبار مدموجين معاً — لأن هذا النموذج
النهائي للنشر الفعلي، بعد ما انتهى التقييم المنهجي على تقسيم منفصل
وتم توثيقه بالمقالة العلمية المرفقة). يحفظ نموذجاً كاملاً لكل قطب
تحت models/<lead>/.

الاستخدام:
    python train_models.py --data-dir /path/to/locked_split
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecg_pipeline.classifier import train_lead_model, MODELS_DIR

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}


def load_combined(data_dir: Path, cat: str, lead: str) -> np.ndarray:
    """يدمج بيانات التدريب والاختبار معاً لتدريب النموذج النهائي للنشر."""
    train = np.load(data_dir / f"{cat}_{lead}_train.npy")
    test = np.load(data_dir / f"{cat}_{lead}_test.npy")
    min_len = min(train.shape[1], test.shape[1])
    return np.vstack([train[:, :min_len], test[:, :min_len]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                         help="مجلد البيانات المقفولة (locked_split) الناتج من train_test_split_lock.py")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    for lead, cats in LEAD_CATEGORIES.items():
        print(f"=== تدريب نموذج قطب {lead} ===")
        normal_beats = load_combined(data_dir, "Normal", lead)

        raw_pathological = {cat: load_combined(data_dir, cat, lead) for cat in cats}
        min_len = min([normal_beats.shape[1]] + [b.shape[1] for b in raw_pathological.values()])

        normal_beats = normal_beats[:, :min_len]
        pathological_beats = {cat: b[:, :min_len] for cat, b in raw_pathological.items()}

        model = train_lead_model(lead, normal_beats, pathological_beats)
        model.save(MODELS_DIR / lead)
        n_total = sum(len(b) for b in pathological_beats.values())
        print(f"  تم الحفظ: {n_total} نبضة مرضية + {len(normal_beats)} نبضة طبيعية للمرجع")
        print(f"  الفئات المدعومة: {model.classes}")

    print("\n✅ انتهى تدريب كل النماذج. الملفات محفوظة في:", MODELS_DIR)


if __name__ == "__main__":
    main()
