# Báo cáo Day 08 — 2A202604047 - Phạm Hải

> **Trạng thái:** Bản nháp sau Part 4 (18/11/2026). Các mục phụ thuộc Part 5–6 được giữ ở trạng thái **chưa thực hiện**. Số liệu Part 4 lấy từ `submission/qc/review_stats.json` và `submission/time_log.csv`. Các nhận xét và quyết định cuối cùng cần do học viên kiểm tra, diễn đạt và chịu trách nhiệm.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Cơ sở thực nghiệm (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| `person`, `motorbike`, `bus`, `truck` | YOLO26s, `imgsz=1024`, CPU | `0.30` (mặc định) | Đã chạy YOLO26n ở hai ngưỡng (`r02`, `r03`) và YOLO26s (`r08`). YOLO26s cải thiện recall tối đa trên Gold, nhưng chưa đủ căn cứ chọn ngưỡng riêng cho các lớp này. |
| `bicycle` | YOLO26s, `imgsz=1024`, CPU | `0.25` | Threshold sweep với `r08`: recall 100%, precision 71% trên **5** xe đạp Gold. Kết quả dựa trên mẫu nhỏ. |
| `car` | YOLO26s, `imgsz=1024`, CPU | `0.45` | Threshold sweep với `r08`: recall 100%, precision 73% trên **8** ô tô Gold. |
| `three_wheeler` | YOLOE-26s, `imgsz=640`, CPU; prompt `tuk-tuk`, `cyclo`, `cargo tricycle` | `0.30` (mặc định) | `r05`: YOLOE nhận diện tuk-tuk trong `img_002` và xích lô trong `img_032`, nhưng có dự đoán nghi sai ở `img_011` và bỏ sót một số xích lô; recall tối đa trên Gold 86%, chưa đạt mục tiêu 90%. |

- **Đối chiếu hai cấu hình YOLO26:** với 4 ảnh Gold, recall **tối đa trong phép quét ngưỡng** thay đổi từ YOLO26n/640 sang YOLO26s/1024 như sau: `person` 64% → 82% (22 vật Gold), `bicycle` 0% → 100% (5 vật), `motorbike` 53% → 80% (15 vật), `car` 88% → 100% (8 vật). Đây không phải recall đo tại cùng một ngưỡng và không chứng minh kết quả trên ảnh mới. YOLO26n/640 xử lý 36 ảnh ở 0,20 giây/ảnh; YOLO26s/1024 xử lý 36 ảnh ở 0,50 giây/ảnh (thời gian suy luận do script ghi, không bao gồm thời gian nạp model).
- **Lệnh `merge_prelabels.py` đã chạy** (từ thư mục gốc repo; dùng Python của môi trường `Day08_CPU`):

  ```powershell
  & 'C:\My Future\Day08_CPU\.venv\Scripts\python.exe' scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26s_all/pre.coco.json --thr "bicycle=0.25,car=0.45" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```

  Kết quả: **503 box trên 28 ảnh** (24 Batch + 4 Gold), gồm `person=262`, `motorbike=138`, `car=46`, `three_wheeler=31`, `bicycle=14`, `truck=8`, `bus=4`. Đây là nhãn máy **chưa review**; 4 ảnh Gold chỉ phục vụ đo máy, không import vào Task review.
- **SAM đã thử:** SAM 2.1 Tiny (`sam2.1_t.pt`), `r04`, dùng YOLO26n tạo box rồi SAM tạo mask ở `img_004` và `img_013`: **10/10 box có mask**, 3,57 giây/ảnh trên CPU. Kiểm tra hình học bằng `scripts/inspect_sam_masks.py` cho thấy 10/10 mask không rỗng, không có đỉnh polygon ngoài ảnh; phần mask ngoài box từ 0,0% đến 0,9% và IoU giữa box mask với box detector từ 0,906 đến 0,978. Đây chỉ là kiểm tra cấu trúc. **Chất lượng biên, việc mask ăn sang người/nền và tính đúng của box vẫn phải do học viên tự nhìn hai ảnh rồi xác nhận**, không suy ra từ các số hình học. Part 5 (SAM bằng click cho 3 vật) chưa thực hiện.
- **Lỗi không khắc phục được chỉ bằng đổi ngưỡng:** YOLO26 theo danh mục COCO không có lớp `three_wheeler`, nên có thể gán xe ba bánh thành `car` hoặc `truck`. Cần kết hợp YOLOE và người duyệt xác nhận đúng lớp; thay đổi confidence không thể tự bổ sung lớp còn thiếu.
- **Ví dụ về điểm tin cậy:** trong `img_011`, các dự đoán như `person 0.28` và `motorbike 0.35` xuất hiện ở ngưỡng 0.25 nhưng mất khi tăng lên 0.50; cần kiểm tra bằng mắt trước khi loại. Trong `img_011`, YOLOE tạo box `three_wheeler 0.34` ở mép trái cho một phương tiện lớn, **nghi sai lớp**. Trong `img_002`, tuk-tuk có dự đoán `three_wheeler 0.97` phù hợp với vật nhìn thấy. **Chưa xác nhận một ví dụ điểm cao nhưng sai**, không tự tạo ví dụ.
- **Giới hạn phép đo:** chỉ có **4 ảnh Gold**, gồm 22 `person`, 5 `bicycle`, 15 `motorbike`, 8 `car`, 0 `bus`, 1 `truck`, 7 `three_wheeler` và 13 vùng `ignore_region` (tổng 71 annotation). Ngưỡng được chọn và đánh giá trên cùng Gold nên precision/recall là ước lượng **lạc quan**, đặc biệt với lớp ít vật. `person`, `motorbike`, `three_wheeler` chưa đạt recall mục tiêu 90% ngay cả tại mức recall tối đa trong phép quét; `bus` và `truck` không đủ mẫu để chọn ngưỡng theo lớp. Nguồn: `submission/qc/threshold_sweep.{csv,md,png}`.

## 2. AI error profile (từ `qc/review_stats.md`)

Đã export **24 ảnh Batch** (`submission/labels/batch_reviewed.json`). `review_stats.json` loại **21 đối tượng quanh bẫy** khỏi profile máy: 11 box từ 440 box pre-label Batch và 10 đối tượng từ 531 annotation sau duyệt. Vì vậy mẫu số dùng dưới đây là **429 box máy** và **521 đối tượng sau duyệt**. Không dùng 531 annotation COCO thô làm mẫu số cho profile đã loại bẫy.

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 280 | 65,3% | 429 box máy |
| edited | 99 | 23,1% | 429 box máy |
| deleted | 50 | 11,7% | 429 box máy |
| added | 142 | 27,3% | 521 đối tượng sau duyệt |

- Trong `review_stats.json`, **8** trường hợp `deleted + added` được ước tính là đổi lớp; thuật toán ghép box chỉ là phép đo gần đúng, cần đối chiếu ca thực tế trước khi kết luận loại lỗi.
- Lớp máy yếu nhất và ảnh minh chứng: **học viên cần tự chọn và xác nhận bằng ảnh**, bảng theo lớp trong `submission/qc/review_stats.md` là nguồn số liệu.
- Trên **4 ảnh Gold**, máy có **45 TP, 15 FP, 13 FN**: precision **45/60 = 75,0%**, recall **45/58 = 77,6%** (58 vật Gold không tính `ignore_region`). Vì chọn ngưỡng trên cùng Gold, phép đo này lạc quan; không coi là hiệu năng trên ảnh mới.

## 3. Thiên kiến tự động hóa và ảnh bẫy

Sau export Batch và chạy `review_stats.py`, kết quả tự động ghi **11/12 = 91,7%** bẫy được bắt. Theo loại: `missing` 2/3, `extra` 3/3, `loose` 3/3, `wrong_class` 3/3. Nguồn: `submission/qc/review_stats.json`.

- Bẫy lọt: `img_005.jpg`, loại `missing`. **Học viên cần tự xác nhận** vì sao bẫy lọt và bước checklist nào đã bỏ; con số không tự xác định được nguyên nhân.
- Mối quan hệ giữa tỉ lệ `accepted` **280/429** và bắt bẫy **11/12**: **học viên tự phân tích**, không suy luận chỉ từ hai tỉ lệ.

## 4. Máy có đáng dùng không?

- **Thời gian gán Gold:** `submission/time_log.csv` ghi **25 phút/4 ảnh**, tức **ước lượng 6,25 phút/ảnh** sau khi gán, không phải thời gian bấm đồng hồ. **Thời gian review Batch:** log ghi khoảng **101,05 phút/24 ảnh = 4,21 phút/ảnh**, gồm ba phiên 28:05 + 14:50 + 58:08. Đây cũng là thời gian xấp xỉ; không dùng hai số này để khẳng định mức tiết kiệm thời gian chính xác.
- **Kết luận có sử dụng pre-label cho công việc thật hay không:** **chưa quyết định**; cần đối chiếu `accepted / edited / deleted / added`, recall trên Gold, số bẫy bắt được và thời gian Batch thực tế.
- **Ba quy tắc giảm thiên kiến có thể đo được:** học viên tự viết sau khi xem lại bẫy lọt và quy trình duyệt; chưa ghi thay quyết định cá nhân.

## 5. Mask SAM cho 3 vật (Part 5)

**Chưa thực hiện Part 5 bằng click.** Kết quả 10 mask từ box → SAM 2.1 Tiny ở Part 2 **không thay thế** yêu cầu ít nhất 3 ảnh mask với số click được ghi nhận.

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| Chưa chọn | SAM 2.1 Tiny (dự kiến dùng lại) | Chưa thực hiện | Chưa đánh giá | Chưa đánh giá | Chưa có |

- Đối chiếu guideline mục 4.2 và so sánh CVAT Interactor (nếu dùng): **đợi kết quả thực nghiệm**.

## 6. Calibration (Part 6)

**Chưa thực hiện Part 6.** Chưa có đủ ít nhất 3 quyết định hiệu chỉnh, gồm nguồn `loi_may` và `cau_hoi_mo`; `guideline_v1.1_calibrated.md` hiện là mẫu chưa hoàn thiện. Sau khi review, học viên tự quyết định các quy tắc, ghi `CAL-xx`, dẫn ảnh ví dụ và cập nhật lịch sử v1.1.

## 7. QC và quyết định (Part 6)

**Chưa thực hiện QC.** Chưa tạo mẫu QC, chưa đo tỉ lệ lỗi theo ảnh/đối tượng và chưa có `qc_report_auto.md` để đọc cận trên 95%.

- Mẫu: **chưa chọn**; người kiểm dự kiến là học viên (cần thừa nhận giới hạn tự kiểm).
- Tỉ lệ lỗi theo ảnh: **chưa có tử số/mẫu số**.
- Tỉ lệ lỗi theo đối tượng: **chưa có tử số/mẫu số**.
- Top lỗi và ví dụ: **chưa thu thập**.
- Script kiểm được / không kiểm được: **sẽ đối chiếu sau khi chạy QC**.

**Quyết định Release / Hold / Rework: CHƯA ĐƯA RA.** Chỉ kết luận sau khi đối chiếu ngưỡng trong `submission/taxonomy.json`, dữ liệu QC và báo cáo thống kê. Ngày quyết định: chưa xác định.

## 8. Ghi chú cá nhân

- **Công cụ AI đã dùng:** YOLO26n, YOLO26s, YOLOE-26s và SAM 2.1 Tiny để chạy dự đoán trong các Part được phép; ChatGPT hỗ trợ giải thích lệnh PowerShell, kiểm tra cấu trúc COCO, so sánh ảnh dự đoán Pilot, tổ chức dữ liệu vào `model_log.csv` và chuẩn bị bản nháp báo cáo này. Các số liệu trích từ output của script, `model_log.csv` và `threshold_sweep.*`; các nhận định và quyết định cuối phải được học viên tự đối chiếu với dữ liệu thực tế.
- **Lưu ý về việc dùng AI đám mây:** trong quá trình trao đổi, đã gửi ảnh kết quả Pilot (`img_002`, `img_011`, `img_032`) cho ChatGPT để tham khảo phân tích. `RULES.md` quy định không gửi ảnh dataset lên dịch vụ AI đám mây nếu chưa được giảng viên cho phép. **Chưa có thông tin xác nhận đã được cho phép; cần khai báo trung thực và làm rõ với giảng viên.** Từ đây không tiếp tục gửi ảnh dataset lên dịch vụ đám mây khi chưa có sự cho phép.
- **Gold có bị nhiễm không?** Theo trình tự thao tác trong buổi làm, 4 ảnh Gold được gán tay và export trước khi chạy model trên Gold. **Học viên cần tự xác nhận** không xem pre-label Gold hoặc nhờ AI chỉ cách gán từng vật trước thời điểm export; nếu có, phải ghi lại cụ thể thay vì để khẳng định này.
- **Máy / CVAT:** Windows PowerShell; PyTorch dùng CPU (`--device cpu`), Python 3.12 trong `C:\My Future\Day08_CPU\.venv`; CVAT Docker local **v2.81.0**, `http://localhost:8080`. Đã dùng hard link trọng số model vào thư mục `models/` của repo (không commit trọng số). Có `FutureWarning` về `torch.jit.load` khi chạy YOLOE nhưng quá trình suy luận hoàn thành.
- **Dùng cache `data/cache/`?** **Không:** hai model đã được chạy trực tiếp trên 36 ảnh (`y26_all`, `y26s_all`, `yoloe_all`) ở confidence 0.05.
- **Bug script đã sửa với hỗ trợ của Codex (nhánh `codex/lab8-fixes`):** bảo vệ `qc_sample.py` khỏi làm lệch mẫu và phiếu review khi chạy lại; buộc `qc_report.py` gợi ý Hold khi mẫu chưa kiểm đủ; siết kiểm tra cấu trúc COCO, họ model, quyết định báo cáo, đủ 12 bẫy và commit của học viên; cấp `run_id` theo mã lớn nhất và giữ mã khi chạy lại cùng output. Có thêm unit test và công cụ overlay/kiểm tra hình học mask. Các thay đổi chỉ tác động code/tài liệu, không sửa Gold, trap/pre-trap, threshold hay kết quả review. Học viên cần đọc diff và xác nhận trước khi merge.
- **Giới hạn thời gian:** mục `gold_manual` trong `time_log.csv` là **ước lượng 25 phút**, không phải thời gian bấm đồng hồ. Cần khai báo điều này, không sửa thành số đo thật.
- **Cảnh báo kiểm nhãn trước Part 5–6:** `validate_labels.py --all` báo box `car` #278 tại `img_023.jpg` cao **15,4 px**, thấp hơn quy tắc 20 px. Script vẫn trả `OK` vì đây là cảnh báo; học viên cần tự đối chiếu ảnh và guideline, không tự động sửa nhãn đã export.
- **Một điều sẽ làm khác lần sau:** **học viên tự bổ sung** sau khi hoàn thành review và tự đánh giá quy trình.
