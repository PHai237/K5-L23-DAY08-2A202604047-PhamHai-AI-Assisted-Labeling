"""Kiểm định lượng và tạo overlay cho mask trong một file COCO.

Ví dụ (không sửa file COCO nguồn):
    python scripts/inspect_sam_masks.py --coco work/prelabel/t_sam/pre.coco.json --out work/sam_audit

Đầu ra gồm ảnh overlay (mask, biên mask, box), mask_stats.csv và mask_stats.json.
Các cờ chỉ là kiểm tra hình học; học viên vẫn phải tự nhìn vật thể để kết luận chất lượng mask.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import IMAGES_DIR, box_iou, load_coco, polys_to_mask, rel, write_csv


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def inspect_object(obj, w: int, h: int) -> tuple[dict, np.ndarray]:
    flags: list[str] = []
    if not obj.polys:
        return {
            "annotation_id": obj.id, "class": obj.cls, "mask_pixels": 0, "mask_image_pct": 0.0,
            "mask_box_fill": 0.0, "outside_box_pct": 0.0, "mask_bbox_iou": 0.0,
            "components": 0, "flags": "NO_MASK",
        }, np.zeros((h, w), np.uint8)

    if any(x < 0 or x > w for p in obj.polys for x in p[0::2]) or any(y < 0 or y > h for p in obj.polys for y in p[1::2]):
        flags.append("POLYGON_OUT_OF_IMAGE")
    mask = polys_to_mask(obj.polys, w, h)
    pixels = int(mask.sum())
    if pixels == 0:
        flags.append("EMPTY_MASK")
    x1, y1, x2, y2 = obj.box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box_mask = np.zeros_like(mask)
    bx1, by1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    bx2, by2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    box_mask[by1:by2, bx1:bx2] = 1
    outside = int(np.logical_and(mask, box_mask == 0).sum())
    outside_frac = outside / pixels if pixels else 0.0
    fill = pixels / box_area if box_area > 0 else 0.0
    mb = mask_bbox(mask)
    mb_iou = box_iou(obj.box, mb) if mb else 0.0

    import cv2
    components = max(0, int(cv2.connectedComponents(mask.astype(np.uint8))[0]) - 1)
    if outside_frac > 0.05:
        flags.append("MASK_OUTSIDE_BOX_GT_5PCT")
    if pixels and fill < 0.02:
        flags.append("MASK_TINY_VS_BOX_LT_2PCT")
    if pixels / max(1, w * h) > 0.8:
        flags.append("MASK_GT_80PCT_IMAGE")
    if components > 5:
        flags.append("MANY_COMPONENTS_GT_5")
    return {
        "annotation_id": obj.id,
        "class": obj.cls,
        "box_x": round(x1, 2), "box_y": round(y1, 2),
        "box_w": round(x2 - x1, 2), "box_h": round(y2 - y1, 2),
        "mask_pixels": pixels,
        "mask_image_pct": round(pixels / max(1, w * h) * 100, 3),
        "mask_box_fill": round(fill, 4),
        "outside_box_pct": round(outside_frac * 100, 3),
        "mask_bbox_iou": round(mb_iou, 4),
        "components": components,
        "flags": ";".join(flags) if flags else "OK",
    }, mask


def draw_overlay(image_path: Path, objects, masks: list[np.ndarray], rows: list[dict], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.float32).copy()
    colors = [np.array((230, 25, 75)), np.array((20, 170, 255)), np.array((80, 200, 80)), np.array((255, 150, 20))]
    for idx, mask in enumerate(masks):
        if mask.any():
            color = colors[idx % len(colors)]
            arr[mask.astype(bool)] = arr[mask.astype(bool)] * 0.55 + color * 0.45
    overlay = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for idx, (obj, mask, row) in enumerate(zip(objects, masks, rows)):
        x1, y1, x2, y2 = obj.box
        draw.rectangle((x1, y1, x2, y2), outline="yellow", width=2)
        for poly in obj.polys:
            pts = list(zip(poly[0::2], poly[1::2]))
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill="cyan", width=2)
        label = f"#{obj.id} {obj.cls} fill={row['mask_box_fill']:.2f} out={row['outside_box_pct']:.1f}%"
        draw.rectangle((x1, max(0, y1 - 14), min(overlay.width, x1 + 260), y1), fill="black")
        draw.text((x1 + 2, max(0, y1 - 13)), label, fill="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="file COCO có segmentation polygon")
    ap.add_argument("--out", required=True, help="thư mục ghi overlay và thống kê")
    args = ap.parse_args()
    coco_path, out_dir = Path(args.coco), Path(args.out)
    if not coco_path.exists():
        ap.error(f"không thấy {coco_path}")
    cs = load_coco(coco_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    structural = 0
    for name, objects in cs.objects.items():
        image_path = IMAGES_DIR / name
        if not image_path.exists():
            print(f"LỖI: không thấy ảnh nguồn {rel(image_path)}", file=sys.stderr)
            structural += 1
            continue
        w, h = Image.open(image_path).size
        rows, masks = [], []
        for obj in objects:
            row, mask = inspect_object(obj, w, h)
            row = {"image": name, **row}
            rows.append(row)
            masks.append(mask)
            if "EMPTY_MASK" in row["flags"] or "POLYGON_OUT_OF_IMAGE" in row["flags"]:
                structural += 1
        all_rows.extend(rows)
        draw_overlay(image_path, objects, masks, rows, out_dir / f"{Path(name).stem}_sam_audit.jpg")
    header = ["image", "annotation_id", "class", "box_x", "box_y", "box_w", "box_h", "mask_pixels",
              "mask_image_pct", "mask_box_fill", "outside_box_pct", "mask_bbox_iou", "components", "flags"]
    write_csv(out_dir / "mask_stats.csv", header, all_rows)
    (out_dir / "mask_stats.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    flagged = sum(r["flags"] != "OK" for r in all_rows)
    print(f"Đã kiểm {len(all_rows)} mask trên {len(cs.images)} ảnh: {flagged} mask có cờ hình học, "
          f"{structural} lỗi cấu trúc. Kết quả: {rel(out_dir)}")
    print("Cờ hình học không thay thế quan sát vật thể; mở các ảnh *_sam_audit.jpg để tự xác nhận biên mask.")
    sys.exit(1 if structural else 0)


if __name__ == "__main__":
    main()
