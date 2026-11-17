"""Tự kiểm bài nộp trước khi push.   python scripts/check_submission.py

Chỉ kiểm CÓ / ĐỦ (file, cột, số dòng, định dạng), KHÔNG chấm nội dung. Hết `LỖI` rồi mới nộp.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from common import ROOT, SUB, load_taxonomy, read_csv

errs, warns = [], []


def E(m): errs.append(m)
def W(m): warns.append(m)


def exists(p: Path, what: str = "") -> bool:
    if not p.exists():
        E(f"thiếu {p.relative_to(ROOT)} {what}".strip())
        return False
    return True


def md_text(p: Path, n_min: int, hint: str) -> str:
    t = p.read_text(encoding="utf-8-sig")
    heads = re.findall(r"^##\s+.+$", t, flags=re.M)
    if len(heads) < n_min:
        E(f"{p.relative_to(ROOT)}: chỉ có {len(heads)} mục '##', cần ≥ {n_min} ({hint})")
    if "<<" in t:
        E(f"{p.relative_to(ROOT)}: còn chỗ chưa điền `<<...>>`")
    return t


def report_decision(text: str) -> str | None:
    """Return an explicit student decision, not a template/list of alternatives."""
    m = re.search(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?quyết\s+định(?:\*\*)?\s*:\s*(?:\*\*)?(release|hold|rework)\b",
        text,
    )
    return m.group(1).upper() if m else None


def model_families(rows: list[dict]) -> set[str]:
    have = set()
    for row in rows:
        text = " ".join(str(row.get(k, "")) for k in ("model", "params")).lower()
        if "yolo26" in text:
            have.add("YOLO26")
        if "yoloe" in text:
            have.add("YOLOE")
        if re.search(r"\bsam(?:\s*2(?:\.1)?|\s*3|[(_+-])", text):
            have.add("SAM")
    return have


EXPECTED_RULES = {"min_box_height_px": 20, "match_iou": 0.5, "accept_iou": 0.9, "accept_mask_iou": 0.85, "trap_catch_min": 0.8,
                  "error_rate_img_max": 0.02, "self_iou_min": 0.85, "self_class_min": 0.95, "recall_target": 0.9, "self_unmatched_max": 0.1}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tax = load_taxonomy()
    if tax.rules != EXPECTED_RULES:
        E("taxonomy.json: mục `rules` bị đổi so với bản gốc (RULES.md: không được đổi ngưỡng)")
    rd = (ROOT / "README.md").read_text(encoding="utf-8-sig") if (ROOT / "README.md").exists() else ""
    m = re.search(r"^#\s*Day 08 Lab\s*[—-]\s*([A-Za-z0-9_]{3,})\s*-\s*\S+", rd, flags=re.M)
    mshv = m.group(1) if m else None
    if not m:
        E("README.md phải có dòng tiêu đề: `# Day 08 Lab — <MSHV> - <Họ và tên>` (xem SUBMISSION.md)")
    if exists(ROOT / "STUDENT.md"):
        t = (ROOT / "STUDENT.md").read_text(encoding="utf-8-sig")
        if "<<" in t:
            E("STUDENT.md còn chỗ chưa điền")

    # báo cáo chính (đã gộp recipe, QC, ghi chú cá nhân)
    if exists(SUB / "REPORT.md"):
        t = md_text(SUB / "REPORT.md", 7, "khuôn trong chính file REPORT.md")
        if not report_decision(t):
            E("REPORT.md: chưa có quyết định rõ theo dạng `Quyết định: Release|Hold|Rework` (liệt kê lựa chọn hoặc ghi CHƯA không được tính)")
        if "sam" not in t.lower():
            E("REPORT.md: chưa nêu phiên bản SAM bạn chọn (SAM là bắt buộc)")

    # guideline v1.1 (bản bổ sung)
    g = SUB / "guidelines" / "guideline_v1.1_calibrated.md"
    if exists(g):
        t = md_text(g, 3, "ca khó mới, quy tắc mới, lịch sử thay đổi")
        if not re.search(r"img_0\d\d", t):
            E("guideline_v1.1: mỗi ca khó mới phải dẫn ảnh ví dụ (img_0xx)")
        if "v1.1" not in t:
            W("guideline_v1.1: không thấy chữ 'v1.1' trong bảng lịch sử thay đổi")

    # log
    ml = SUB / "model_log.csv"
    if exists(ml):
        rows = [r for r in read_csv(ml) if (r.get("run_id") or "").strip()]
        if len(rows) < 4:
            E(f"model_log.csv: {len(rows)} dòng, cần ≥ 4 lần thử")
        for i, r in enumerate(rows, 2):
            if not (r.get("what_went_wrong") or "").strip():
                W(f"model_log.csv dòng {i}: cột what_went_wrong trống (ghi lỗi quan sát được)")
        missing_families = {"YOLO26", "YOLOE", "SAM"} - model_families(rows)
        if missing_families:
            E("model_log.csv: thiếu họ model bắt buộc: " + ", ".join(sorted(missing_families)))
    tl = SUB / "time_log.csv"
    if exists(tl):
        have = {(r.get("step") or "").strip() for r in read_csv(tl) if (r.get("minutes") or "").strip()}
        if not {"gold_manual", "batch_review"} <= have:
            E(f"time_log.csv thiếu bước: {', '.join(sorted({'gold_manual', 'batch_review'} - have))}")

    # nhãn
    labels = SUB / "labels"
    for f in ("gold_manual.json", "batch_reviewed.json"):
        exists(labels / f)
    sub_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_labels.py"), "--all"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=sub_env,
    )
    for line in r.stdout.splitlines():
        if "] LỖI: " in line:
            E("validate_labels " + line.replace("] LỖI: ", "] ", 1))
    if r.returncode != 0 and not any(e.startswith("validate_labels ") for e in errs):
        tail = next((ln.strip() for ln in reversed((r.stderr or r.stdout).splitlines()) if ln.strip()), "")
        E(f"validate_labels lỗi thực thi (exit {r.returncode})" + (f": {tail}" if tail else ""))
    for f in ("pre_traps.coco.json", "traps_key.json"):
        exists(SUB / "prelabel" / f, "(xem GUIDE.md Part 4)")
    # Không mở traps_key.json tại bước tự kiểm: honor system cấm đọc trước khi export batch_reviewed.json.
    # Tính hợp lệ của bẫy được kiểm gián tiếp từ review_stats.json sau Part 4.
    masks = list((SUB / "masks").glob("*.jpg")) + list((SUB / "masks").glob("*.png")) if (SUB / "masks").exists() else []
    if len(masks) < 3:
        E(f"submission/masks/: {len(masks)} ảnh mask, cần ≥ 3 (kết quả SAM của 3 vật, xem GUIDE.md Part 5)")

    # số đo
    qc = SUB / "qc"
    for f in ("review_stats.json", "threshold_sweep.csv", "qc_sample.csv", "qc_review.csv", "qc_report_auto.md", "calibration_decisions.csv"):
        exists(qc / f)
    if (qc / "review_stats.json").exists():
        d = json.loads((qc / "review_stats.json").read_text(encoding="utf-8-sig"))
        if not d.get("traps"):
            E("review_stats.json không có phần bẫy: chạy review_stats.py với --key submission/prelabel/traps_key.json")
        elif d["traps"].get("n", 0) < 12:
            E(f"review_stats.json: chỉ có {d['traps'].get('n', 0)} bẫy được chấm (cần đủ 12)")
        if not d.get("gold"):
            E("review_stats.json không có phần gold: chạy review_stats.py với --pre submission/prelabel/pre_traps.coco.json "
              "(file đầy đủ batch+gold, không dùng pre_traps_import) và --gold submission/labels/gold_manual.json")
    if (qc / "qc_review.csv").exists():
        if any(not (r.get("error_type") or "").strip() for r in read_csv(qc / "qc_review.csv")):
            E("qc_review.csv còn dòng error_type trống (chưa kiểm)")
    if (qc / "calibration_decisions.csv").exists():
        rows = [r for r in read_csv(qc / "calibration_decisions.csv") if (r.get("quyet_dinh") or "").strip()]
        if len(rows) < 3:
            E(f"calibration_decisions.csv: {len(rows)} quyết định, cần ≥ 3")
        srcs = {(r.get("nguon") or "").strip() for r in rows}
        for need in ("loi_may", "cau_hoi_mo"):
            if need not in srcs:
                E(f"calibration_decisions.csv: chưa có ca nào có nguon = {need}")
        bad = srcs - {"cau_hoi_mo", "can_review", "loi_may", "bay_lot", "self_check", ""}
        if bad:
            W(f"calibration_decisions.csv: nguon lạ: {', '.join(sorted(bad))}")
    for f in ("final_det.coco.json", "release_manifest.json"):
        exists(SUB / "release" / f, "(python scripts/build_release.py)")

    # git
    try:
        cfg_name = subprocess.run(
            ["git", "config", "user.name"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        ).stdout.strip()
        cfg_email = subprocess.run(
            ["git", "config", "user.email"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        ).stdout.strip().lower()
        authors = subprocess.run(
            ["git", "log", "--format=%an%x09%ae"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        ).stdout.splitlines()
        n = sum(1 for line in authors if "\t" in line and (
            (cfg_email and line.rsplit("\t", 1)[1].strip().lower() == cfg_email)
            or (cfg_name and line.rsplit("\t", 1)[0].strip() == cfg_name)
        ))
        if n < 4:
            W(f"chỉ tìm thấy {n} commit khớp git user.name/user.email hiện tại; cần ≥ 4 commit của chính học viên theo từng Part")
        if mshv:
            remote = subprocess.run(
                ["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            ).stdout.strip().rstrip("/")
            repo_name = re.sub(r"\.git$", "", remote.rsplit("/", 1)[-1]) if remote else ""
            if repo_name and not re.fullmatch(rf"K5-L23-DAY08-{re.escape(mshv)}-[A-Za-z]+-AIPrelabel", repo_name, flags=re.I):
                W(f"tên repo '{repo_name}' không khớp mẫu SUBMISSION.md cho MSHV {mshv}; bỏ qua nếu coach đã cho quy ước khác")
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        ).stdout.splitlines()
        if any(t.startswith("work/") for t in tracked):
            E("thư mục work/ đang bị commit (đã có trong .gitignore?)")
        if any(t.startswith("models/") and t != "models/.gitkeep" for t in tracked):
            E("thư mục models/ đang bị commit (đã có trong .gitignore?)")
        if any(t.endswith((".pt", ".pth", ".onnx", ".ts")) for t in tracked):
            E("đang commit file trọng số model (.pt/.pth/.onnx/.ts): bỏ ra khỏi git")
        for t in tracked:
            if t.endswith((".md", ".json", ".csv", ".py", ".txt", ".sh", ".yaml", ".yml")) and (ROOT / t).exists():
                if re.search(r"hf_[A-Za-z0-9]{25,}|(?i:password|secret|token)\s*[:=]\s*\S{12,}", (ROOT / t).read_text(encoding="utf8", errors="ignore")):
                    E(f"có thể lộ secret/token trong {t}")
    except Exception:
        W("không kiểm được git (chưa git init?)")
    for m in warns:
        print(f"CẢNH BÁO: {m}")
    for m in errs:
        print(f"LỖI: {m}")
    print(f"\n{len(errs)} lỗi, {len(warns)} cảnh báo." + (" Sạch lỗi: có thể nộp." if not errs else " Sửa hết LỖI rồi nộp."))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
