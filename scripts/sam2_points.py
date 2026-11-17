"""SAM (bản bất kỳ: SAM 1 / SAM 2.1) với prompt là CLICK (giống công cụ Interactor trong CVAT): thử trên một ảnh, xem mask thay đổi theo click.

    python scripts/sam2_points.py --image img_013.jpg --pos "520,330" --neg "130,300" --out work/sam2_click
    python scripts/sam2_points.py --image img_013.jpg --pos "520,330" --box "90,30,960,640" --out work/sam2_click

--pos "x,y"  : click dương (thuộc vật)    [dùng nhiều lần được]
--neg "x,y"  : click âm (KHÔNG thuộc vật) [dùng nhiều lần được]
--box "x1,y1,x2,y2" : khung gợi ý (tuỳ chọn)
Toạ độ là pixel trên ảnh gốc (mở ảnh bằng trình xem ảnh bất kỳ và rê chuột; hoặc thử rồi nhìn kết quả).
Ghi <out>/<name>.jpg (mask tô màu, chấm xanh = dương, chấm đỏ = âm) và in diện tích mask / số polygon.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import IMAGES_DIR, ROOT, die, mask_to_polys, pick_device, rel
from prelabel import weights_path


def pt(s: str):
    try:
        x, y = [v.strip() for v in s.split(",")]
        return float(x), float(y)
    except ValueError:
        die(f"Sai định dạng toạ độ '{s}': cần x,y (ví dụ \"520,330\").")


def parse_box(s: str):
    parts = [v.strip() for v in s.split(",")]
    if len(parts) != 4:
        die(f"Sai định dạng --box '{s}': cần x1,y1,x2,y2 (4 số, ví dụ \"90,30,960,640\").")
    try:
        x1, y1, x2, y2 = [float(v) for v in parts]
    except ValueError:
        die(f"Sai định dạng --box '{s}': các toạ độ phải là số thực.")
    if x2 <= x1 or y2 <= y1:
        die(f"Sai toạ độ --box '{s}': cần x2 > x1 và y2 > y1.")
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="tên ảnh trong data/images/")
    ap.add_argument("--pos", action="append", default=[])
    ap.add_argument("--neg", action="append", default=[])
    ap.add_argument("--box")
    ap.add_argument("--sam", "--sam2", dest="sam", default="sam2.1_b.pt", help="mobile_sam.pt | sam_b.pt (SAM 1) | sam2.1_t/s/b/l.pt (SAM 2.1)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="mask_overlay", help="tên file ảnh kết quả (không đuôi), ví dụ tuktuk_img013")
    a = ap.parse_args()
    path = IMAGES_DIR / a.image
    if not path.exists():
        die(f"Không thấy {rel(path)}")
    if not a.pos and not a.box:
        die("Cần ít nhất một --pos hoặc --box.")
    pts = [pt(s) for s in a.pos] + [pt(s) for s in a.neg]
    labels = [1] * len(a.pos) + [0] * len(a.neg)
    box = parse_box(a.box) if a.box else None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    for x, y in pts:
        if not (0 <= x < w and 0 <= y < h):
            die(f"Click ({x:g},{y:g}) nằm ngoài ảnh {w}x{h}.")
    if box and (box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h):
        die(f"--box phải nằm trong ảnh {w}x{h}.")
    from ultralytics import SAM
    model = SAM(weights_path(a.sam))
    kw = {}
    if pts:
        kw["points"] = [pts]
        kw["labels"] = [labels]
    if box:
        kw["bboxes"] = [box]
    device = pick_device(a.device)
    r = model(str(path), device=device, verbose=False, **kw)[0]
    if r.masks is None or len(r.masks.data) == 0:
        die("SAM không trả mask nào: thử click vào giữa vật hơn.")
    m = r.masks.data[0].cpu().numpy() > 0.5
    ov = np.array(im).astype(np.float32)
    ov[m] = ov[m] * 0.45 + np.array([230, 25, 75], np.float32) * 0.55
    out = Image.fromarray(ov.astype(np.uint8))
    d = ImageDraw.Draw(out)
    for (x, y), l in zip(pts, labels):
        c = (0, 200, 0) if l else (255, 0, 0)
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=c, outline="white", width=2)
    if box:
        d.rectangle(box, outline="yellow", width=2)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    polys = mask_to_polys(m.astype(np.uint8))
    for poly in polys:
        points = list(zip(poly[0::2], poly[1::2]))
        if len(points) >= 3:
            d.line(points + [points[0]], fill=(0, 255, 255), width=2)
    out.save(o / f"{a.name}.jpg", quality=90)
    metadata = {
        "image": a.image,
        "sam": a.sam,
        "device": device,
        "positive_clicks": [list(p) for p in pts[:len(a.pos)]],
        "negative_clicks": [list(p) for p in pts[len(a.pos):]],
        "n_clicks": len(pts),
        "box": box,
        "mask_pixels": int(m.sum()),
        "mask_image_pct": round(float(m.sum() / m.size * 100), 3),
        "polygons": len(polys),
    }
    (o / f"{a.name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mask: {int(m.sum())} px ({m.sum() / m.size * 100:.1f}% ảnh), {len(polys)} polygon -> {rel(o / f'{a.name}.jpg')}")
    print(f"Metadata thật (model, click, box, diện tích): {rel(o / f'{a.name}.json')}")


if __name__ == "__main__":
    main()
