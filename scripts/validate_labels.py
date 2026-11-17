"""Kiểm file COCO export từ CVAT trước khi dùng/commit.

    python scripts/validate_labels.py submission/labels/gold_manual.json --expect gold
    python scripts/validate_labels.py submission/labels/batch_reviewed.json --expect batch
    python scripts/validate_labels.py --all

--expect: các tập (pilot,gold,batch) mà file PHẢI chứa đủ ảnh. --all kiểm mọi file chuẩn trong submission/labels/.
Kiểm: tên lớp có trong taxonomy, box nằm trong ảnh, box quá nhỏ, đủ/thừa ảnh, polygon hợp lệ (--mode seg).
Không kiểm được: box có ôm sát vật không, lớp có đúng thực tế không.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from common import SUB, images_of, load_coco, load_split, load_taxonomy, rel

STANDARD = {  # file -> (tập ảnh phải có, có polygon?)
    "gold_manual.json": ("gold", False),
    "pilot_reviewed.json": ("pilot", False),
    "batch_reviewed.json": ("batch", False),
    "seg_reviewed.json": ("seg", True),
    "self_check_blind.json": ("", False),
}


def check(path: Path, expect: str, need_poly: bool, tax) -> int:
    errs, warns = [], []
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        cs = load_coco(path)
    except Exception as e:  # noqa
        print(f"[{rel(path)}] LỖI: không đọc được JSON: {e}")
        return 1
    if not isinstance(raw, dict):
        print(f"[{rel(path)}] LỖI: gốc JSON phải là object COCO")
        return 1
    sp = load_split()
    if expect == "seg":
        want = sorted(k for k, v in sp.items() if v["split"] == "batch" and v["seg"])
    elif expect:
        want = images_of([s.strip() for s in expect.split(",") if s.strip()])
    else:
        want = []
    errs += cs.problems
    for field in ("images", "categories", "annotations"):
        if not isinstance(raw.get(field), list):
            errs.append(f"trường '{field}' phải là danh sách")
    images_raw = raw.get("images", []) if isinstance(raw.get("images"), list) else []
    cats_raw = raw.get("categories", []) if isinstance(raw.get("categories"), list) else []
    anns_raw = raw.get("annotations", []) if isinstance(raw.get("annotations"), list) else []
    for label, rows in (("image id", images_raw), ("category id", cats_raw), ("annotation id", anns_raw)):
        vals = [r.get("id") for r in rows if isinstance(r, dict)]
        dup = [v for v, n in Counter(vals).items() if n > 1]
        if dup:
            errs.append(f"{label} bị trùng: {dup[:5]}")
    cat_names = [c.get("name") for c in cats_raw if isinstance(c, dict)]
    dup_names = [v for v, n in Counter(cat_names).items() if n > 1]
    if dup_names:
        errs.append(f"tên category bị trùng: {dup_names[:5]}")
    unknown_categories = sorted({str(v) for v in cat_names if v not in tax.classes})
    if unknown_categories:
        errs.append(f"category không có trong taxonomy: {unknown_categories[:5]}")
    for n in want:
        if n not in cs.images:
            errs.append(f"thiếu ảnh {n}")
    extra = [n for n in cs.images if n not in sp]
    if extra:
        errs.append(f"ảnh không thuộc bộ dữ liệu: {extra[:3]}")
    if want:
        extra2 = [n for n in cs.images if n in sp and n not in want]
        if extra2:
            errs.append(f"có {len(extra2)} ảnh ngoài phạm vi mong đợi ({', '.join(extra2[:3])}...)")
    ann_raw_by_id = {}
    for idx, ann in enumerate(anns_raw, 1):
        if not isinstance(ann, dict):
            errs.append(f"annotations[{idx}] không phải object")
            continue
        ann_raw_by_id.setdefault(ann.get("id"), ann)
    nobj = 0
    for n, objs in cs.objects.items():
        w, h = cs.images[n]
        for o in objs:
            nobj += 1
            if o.cls not in tax.classes:
                errs.append(f"{n}: lớp lạ '{o.cls}' (CVAT còn dùng bộ nhãn cũ? tạo lại Project từ cvat/labels_generated.json)")
                continue
            x1, y1, x2, y2 = o.box
            if not all(math.isfinite(v) for v in o.box):
                errs.append(f"{n}: box #{o.id} có toạ độ không hữu hạn")
                continue
            if x2 <= x1 or y2 <= y1:
                errs.append(f"{n}: box #{o.id} {o.cls} có chiều rộng/cao không dương")
            if x2 > w + 1 or y2 > h + 1 or x1 < -1 or y1 < -1:
                errs.append(f"{n}: box #{o.id} {o.cls} nằm ngoài ảnh")
            if o.cls != tax.ignore and (y2 - y1) < tax.rules["min_box_height_px"]:
                warns.append(f"{n}: box #{o.id} {o.cls} cao {y2 - y1:.1f}px < {tax.rules['min_box_height_px']}px (guideline: vật quá nhỏ dùng ignore_region)")
            if need_poly and o.cls != tax.ignore and not o.polys:
                errs.append(f"{n}: #{o.id} {o.cls} không có polygon (task segmentation phải vẽ polygon/mask, không phải rectangle)")
            for poly in o.polys:
                if len(poly) < 6 or len(poly) % 2:
                    errs.append(f"{n}: polygon của #{o.id} phải có ≥ 3 điểm và số toạ độ chẵn")
                    continue
                if not all(math.isfinite(v) for v in poly):
                    errs.append(f"{n}: polygon của #{o.id} có toạ độ không hữu hạn")
                elif any(x < -1 or x > w + 1 for x in poly[0::2]) or any(y < -1 or y > h + 1 for y in poly[1::2]):
                    errs.append(f"{n}: polygon của #{o.id} nằm ngoài ảnh")
            raw_ann = ann_raw_by_id.get(o.id, {})
            bbox_raw = raw_ann.get("bbox")
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                errs.append(f"{n}: #{o.id} thiếu bbox COCO dạng [x,y,w,h]")
            attrs = raw_ann.get("attributes")
            if o.cls != tax.ignore:
                if not isinstance(attrs, dict):
                    errs.append(f"{n}: #{o.id} {o.cls} thiếu object attributes")
                else:
                    can_review = attrs.get("can_review")
                    if can_review is None:
                        errs.append(f"{n}: #{o.id} {o.cls} thiếu thuộc tính can_review")
                    elif str(can_review).strip().lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                        errs.append(f"{n}: #{o.id} can_review='{can_review}' không phải boolean hợp lệ")
                    if "conf" not in attrs:
                        errs.append(f"{n}: #{o.id} {o.cls} thiếu thuộc tính conf")
                    conf = attrs.get("conf")
                    if conf not in (None, ""):
                        try:
                            cf = float(conf)
                            if not math.isfinite(cf) or not 0 <= cf <= 1:
                                raise ValueError
                        except (TypeError, ValueError):
                            errs.append(f"{n}: #{o.id} conf='{conf}' phải rỗng hoặc là số trong [0,1]")
            area_value = raw_ann.get("area")
            if area_value is None:
                errs.append(f"{n}: #{o.id} thiếu area")
            else:
                try:
                    area_value = float(area_value)
                    if not math.isfinite(area_value) or area_value <= 0:
                        raise ValueError
                    if not o.polys:
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > 0 and abs(area_value - box_area) > max(1.0, box_area * 0.01):
                            errs.append(f"{n}: area của #{o.id} ({area_value:.1f}) không khớp diện tích bbox ({box_area:.1f})")
                except (TypeError, ValueError):
                    errs.append(f"{n}: area của #{o.id} phải là số dương")
    seen_objects = Counter((an.get("image_id"), an.get("category_id"), json.dumps(an.get("bbox"), sort_keys=True))
                           for an in anns_raw if isinstance(an, dict))
    exact_dups = [k for k, count in seen_objects.items() if count > 1]
    if exact_dups:
        errs.append(f"có {len(exact_dups)} annotation trùng chính xác image/category/bbox")
    for m in warns[:12]:
        print(f"[{rel(path)}] cảnh báo: {m}")
    if len(warns) > 12:
        print(f"[{rel(path)}] ... và {len(warns) - 12} cảnh báo nữa")
    for m in errs[:20]:
        print(f"[{rel(path)}] LỖI: {m}")
    if len(errs) > 20:
        print(f"[{rel(path)}] ... và {len(errs) - 20} lỗi nữa")
    print(f"[{rel(path)}] {len(cs.images)} ảnh, {nobj} đối tượng: {'LỖI ' + str(len(errs)) if errs else 'OK'}")
    return len(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--expect", default="")
    ap.add_argument("--mode", choices=["det", "seg"], default="det")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    tax = load_taxonomy()
    bad = 0
    if a.all:
        found = False
        for fn, (exp, poly) in STANDARD.items():
            p = SUB / "labels" / fn
            if p.exists():
                found = True
                bad += check(p, "seg" if fn == "seg_reviewed.json" else exp, poly, tax)
        if not found:
            print("Chưa có file nhãn nào trong submission/labels/.")
            sys.exit(1)
    elif a.file:
        bad = check(Path(a.file), a.expect, a.mode == "seg", tax)
    else:
        ap.error("cần <file> hoặc --all")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
