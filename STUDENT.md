# Thông tin học viên

- Họ và tên: Phạm Hải
- MSHV: 2A202604047
- Lớp (L2 / L3 / khác): L2B
- Tài khoản GitHub: PHai237
- Repo bài nộp (URL): https://github.com/PHai237/K5-L23-DAY08-2A202604047-PhamHai-AI-Assisted-Labeling
- Commit chốt bài (hash): 6f769ba (mốc chốt nội dung; commit sau chỉ cập nhật metadata)
- Máy: Windows PowerShell, Python 3.12 (`C:\My Future\Day08_CPU\.venv`); các lượt model đã xác minh chạy bằng CPU; GPU không dùng trong các lượt thực nghiệm; cấu hình GPU phần cứng chưa xác minh
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted **280/429 = 65,3%**; edited **99/429 = 23,1%**; deleted **50/429 = 11,7%**; added **142/521 = 27,3%** đối tượng sau duyệt. Nguồn: `submission/qc/review_stats.json`; thống kê đã loại 21 đối tượng quanh bẫy.
- So với Gold: precision của **máy** **45/60 = 75,0%**; recall **45/58 = 77,6%** trên 4 ảnh. Ngưỡng được chọn trên chính Gold nên số đo lạc quan.
- Bắt bẫy: lượt đầu **8/12** (không lưu snapshot thống kê riêng); **sau Rework 11/12 = 91,7%** trong `submission/qc/review_stats.json`; còn một bẫy `missing` ở `img_005.jpg`.
- Thời gian: Gold **31 phút 26 giây/4 ảnh (7,86 phút/ảnh)**, theo mốc tham khảo; review Batch **101,05 phút/24 ảnh (4,21 phút/ảnh)**.
- Mask SAM 2.1 Tiny: **3 ảnh** `img_004`, `img_013`, `img_017`; số click tương ứng **2, 3, 2**, đối chiếu `submission/masks/*.json` và `REPORT.md` mục 5.
- Calibration: **3 quyết định** CAL-01/02/03 đã ghi trong `submission/qc/calibration_decisions.csv` và guideline v1.1.
- Lỗi QC vòng 1: **1/216 đối tượng**, **1/8 ảnh**, script gợi ý **REWORK**. Sau sửa `img_018`: phiếu ghi **0/216**, **0/8**, script gợi ý **RELEASE**; chỉ ảnh đã sửa được kiểm lại, **không phải** kiểm độc lập lại 8 ảnh. Nguồn: `submission/qc/qc_review_round{1,2}.csv`, `qc_report_round{1,2}.md`.
- Release candidate: `submission/release/release_manifest.json` ghi **24 ảnh, 531 đối tượng**, taxonomy v1.0; quyết định phát hành cuối cùng: **Release sau Rework (18/11/2026)**.
