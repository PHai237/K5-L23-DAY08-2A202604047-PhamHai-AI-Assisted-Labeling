"""Lấy mẫu QC trên nhãn đã duyệt của tập batch và tạo phiếu kiểm.

    python scripts/qc_sample.py --n 5 --seed 42

Mẫu = tối đa --max-hard (3) ảnh có box tick `can_review` (ca khó bạn tự đánh dấu) + n ảnh ngẫu nhiên trong phần còn lại.
Nguồn: submission/release/final_det.coco.json (chạy build_release.py trước; đã gồm bản sửa qc_fixed nếu có).
Đầu ra: submission/qc/qc_sample.csv, submission/qc/qc_review.csv (phiếu để điền), work/qc/review/<ảnh>.jpg (có id từng vật).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import IMAGES_DIR, SUB, WORK, die, images_of, load_coco, load_taxonomy, read_csv, rel, write_csv


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def _existing_sample(sample_path: Path, review_path: Path, batch: list[str]) -> tuple[list[dict], list[str]]:
    """Load a frozen QC sample and ensure its review sheet still covers the same images."""
    if not sample_path.exists():
        die(f"Đã có {rel(review_path)} nhưng thiếu {rel(sample_path)}. Không thể biết mẫu gốc; "
            "không ghi đè tự động. Khôi phục file hoặc chạy lại với --force nếu chấp nhận mất phiếu cũ.")
    sample_rows = read_csv(sample_path)
    sample = [(r.get("image") or "").strip() for r in sample_rows]
    if not sample or any(not n for n in sample) or len(sample) != len(set(sample)):
        die(f"{rel(sample_path)} rỗng, có tên ảnh trống hoặc trùng; không ghi đè phiếu QC đã có.")
    outside = sorted(set(sample) - set(batch))
    if outside:
        die(f"{rel(sample_path)} có ảnh không thuộc batch: {', '.join(outside[:8])}.")
    review_rows = read_csv(review_path)
    reviewed_images = {(r.get("image") or "").strip() for r in review_rows if (r.get("image") or "").strip()}
    missing = sorted(set(sample) - reviewed_images)
    extra = sorted(reviewed_images - set(sample))
    if missing or extra:
        details = []
        if missing:
            details.append(f"thiếu trong phiếu: {', '.join(missing[:8])}")
        if extra:
            details.append(f"ngoài mẫu: {', '.join(extra[:8])}")
        die(f"{rel(sample_path)} và {rel(review_path)} không đồng bộ ({'; '.join(details)}). "
            "Không ghi đè tự động; dùng --force chỉ khi chấp nhận tạo lại cả hai file.")
    return sample_rows, sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SUB / "release" / "final_det.coco.json"), help="mặc định: nhãn cuối từ build_release.py")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-hard", type=int, default=3, help="tối đa bao nhiêu ảnh `can_review` đưa vào mẫu (lấy ảnh có nhiều box can_review nhất)")
    ap.add_argument("--force", action="store_true", help="ghi đè qc_review.csv đã có (CẨN THẬN: mất phần đã điền)")
    a = ap.parse_args()
    tax = load_taxonomy()
    p = Path(a.src)
    if not p.exists():
        die(f"Chưa có {rel(p)}. Chạy `python scripts/build_release.py` trước.")
    cs = load_coco(p)
    batch = [n for n in images_of(["batch"]) if n in cs.objects]
    nobj = lambda n: sum(1 for o in cs.objects[n] if o.cls != tax.ignore)  # noqa: E731
    sample_path = SUB / "qc" / "qc_sample.csv"
    rv = SUB / "qc" / "qc_review.csv"
    if rv.exists() and not a.force:
        sample_rows, sample = _existing_sample(sample_path, rv, batch)
        hard = [r["image"] for r in sample_rows if (r.get("la_ca_kho") or "").strip().lower() == "yes"]
        k = len(sample) - len(hard)
        print(f"Giữ nguyên {rel(sample_path)} và {rel(rv)} (đã tồn tại; dùng --force để tạo lại cả hai).")
    else:
        cnt = {n: sum(truthy(o.attrs.get("can_review")) for o in cs.objects[n]) for n in batch}
        hard_all = sorted([n for n in batch if cnt[n] > 0], key=lambda n: (-cnt[n], n))
        hard = sorted(hard_all[:a.max_hard])
        if len(hard_all) > a.max_hard:
            print(f"Có {len(hard_all)} ảnh can_review; chỉ lấy {a.max_hard} ảnh nhiều nhất (--max-hard để đổi).")
        pool = [n for n in batch if n not in hard]
        k = min(a.n, len(pool))
        rnd = sorted(random.Random(a.seed).sample(pool, k))
        sample = sorted(set(hard) | set(rnd))
        write_csv(sample_path, ["image", "la_ca_kho", "so_doi_tuong"],
                  [{"image": n, "la_ca_kho": "yes" if n in hard else "no", "so_doi_tuong": nobj(n)} for n in sample])
        write_csv(rv, ["image", "object_id", "error_type", "reviewer", "note"],
                  [{"image": n, "object_id": "", "error_type": "", "reviewer": "", "note": ""} for n in sample])
        print(f"Đã tạo {rel(sample_path)} và phiếu kiểm {rel(rv)}.")
    from viz import draw_image
    out = WORK / "qc" / "review"
    out.mkdir(parents=True, exist_ok=True)
    for n in sample:
        draw_image(IMAGES_DIR / n, cs.objects[n], tax, show_ids=True).save(out / (Path(n).stem + ".jpg"), quality=88)
    print(f"Mẫu QC: {len(sample)}/{len(batch)} ảnh batch ({len(hard)} ca khó + {k} ngẫu nhiên), "
          f"{sum(nobj(n) for n in sample)} đối tượng. Ảnh để duyệt (id từng vật): {rel(out)}/")
    print("Bước tiếp: mở ảnh, điền qc_review.csv (docs/CALIBRATION_QC.md), rồi `python scripts/qc_report.py`")


if __name__ == "__main__":
    main()
