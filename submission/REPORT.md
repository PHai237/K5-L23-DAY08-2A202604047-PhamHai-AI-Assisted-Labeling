# Báo cáo Day 08 — 2A202604047 - Phạm Hải

> **Tổng kết:** Tôi đã hoàn thành Part 1–6, gồm Gold, thử model, review Batch, ba mask SAM, calibration và QC. Sau khi sửa lỗi ở `img_018`, tôi chọn Release sau Rework. Các số liệu và giới hạn được trình bày trong từng mục.

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
- **SAM đã thử:** SAM 2.1 Tiny (`sam2.1_t.pt`), `r04`, dùng YOLO26n tạo box rồi SAM tạo mask ở `img_004` và `img_013`: **10/10 box có mask**, 3,57 giây/ảnh trên CPU. Kiểm tra hình học bằng `scripts/inspect_sam_masks.py` cho thấy 10/10 mask không rỗng, không có đỉnh polygon ngoài ảnh; phần mask ngoài box từ 0,0% đến 0,9% và IoU giữa box mask với box detector từ 0,906 đến 0,978. Đây chỉ là kiểm tra cấu trúc, không chứng minh chất lượng biên. Part 5 bằng click đã có ba cặp JPG/JSON; xem mục 5.
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
- **Lớp cần chú ý theo tỉ lệ giữ nguyên là `bicycle`:** máy có 7 box, chỉ 1 box được giữ nguyên (14,3%), 5 box phải sửa và 1 box bị xóa. Tôi còn phải thêm 6 box xe đạp. Mẫu này nhỏ nên chưa thể kết luận model luôn kém ở lớp `bicycle`. Nếu tính số thao tác, `motorbike` tốn công hơn: 52 box sửa, 21 box xóa và 45 box thêm.
- Trên **4 ảnh Gold**, máy có **45 TP, 15 FP, 13 FN**: precision **45/60 = 75,0%**, recall **45/58 = 77,6%** (58 vật Gold không tính `ignore_region`). Vì chọn ngưỡng trên cùng Gold, phép đo này lạc quan; không coi là hiệu năng trên ảnh mới.

## 3. Thiên kiến tự động hóa và ảnh bẫy

Ở lượt đầu tôi bắt được **8/12** bẫy (không lưu riêng bản thống kê cho mốc này). Sau Rework, `review_stats.py` cho kết quả **11/12 = 91,7%**: `missing` 2/3, `extra` 3/3, `loose` 3/3 và `wrong_class` 3/3. Bảng chi tiết nằm trong `submission/qc/review_stats.json`.

- Bẫy còn lọt là lỗi `missing` ở `img_005.jpg`. Khi review ảnh đông xe, tôi thường kiểm từng xe cùng người ngồi trên xe. Cách này có thể khiến tôi tập trung vào các vật đã có box mà bỏ qua vật chưa được gán. Tôi chưa có log thao tác để xác định chắc chắn nguyên nhân.
- `accepted` **280/429** và bắt bẫy **11/12 sau Rework** đo hai việc khác nhau: box máy được giữ nguyên và lỗi cố ý được phát hiện. Cả hai không phủ nhận bẫy `missing` còn lọt ở `img_005.jpg`, nên không dùng tỉ lệ accepted để thay cho lượt rà vật bị bỏ sót; chưa có bằng chứng để quy lỗi lọt cho một nguyên nhân tâm lý cụ thể.

**Tôi rút ra ba cách kiểm tra để áp dụng ở lần review sau.** Các chỉ số bên phải là cách đo mức tuân thủ, chưa phải kết quả đã đo:

| Quy tắc | Cách đo mức tuân thủ (chưa đo) |
|---|---|
| Sau khi duyệt từng vật, quét **độc lập toàn ảnh** để tìm vật chưa có box. | Số ảnh có dấu hoàn tất lượt quét thứ hai / tổng ảnh cần review; ghi riêng số vật `missing` phát hiện ở lượt này. |
| Kiểm lớp phương tiện theo **cấu trúc thân, cabin và bánh nhìn thấy**, không chỉ theo tên model hay confidence. | Số ca phương tiện mơ hồ có ghi đủ các dấu hiệu quan sát được và quyết định lớp / tổng ca phương tiện mơ hồ; ghi riêng ca phải đánh dấu `can_review`. |
| Có lượt rà riêng **vật nhỏ, vật bị che và vật gần mép ảnh**. | Số ảnh có checklist hoàn tất cho cả ba nhóm / tổng ảnh review; đếm lỗi bỏ sót được phát hiện theo từng nhóm. Không có vật thuộc nhóm thì ghi “không áp dụng”, không tính là đã phát hiện lỗi. |

## 4. Máy có đáng dùng không?

- **Thời gian:** Gold **31 phút 26 giây/4 ảnh (7,86 phút/ảnh)**, theo mốc tham khảo; review Batch **101,05 phút/24 ảnh (4,21 phút/ảnh)**. Tôi không dùng hai mốc này để tính chính xác mức tiết kiệm thời gian.
- **Có nên dùng pre-label trong công việc thật?** Có, nhưng chỉ để có nhãn ban đầu; tôi vẫn cần kiểm tra toàn bộ ảnh. Trong lô này, 99/429 box máy phải sửa, 50/429 bị xóa và 142/521 đối tượng sau duyệt là box thêm. Recall của máy trên Gold là 45/58. Khối lượng kiểm và sửa vẫn đáng kể; từ dữ liệu thời gian hiện có tôi chưa kết luận được pre-label tiết kiệm bao nhiêu.

## 5. Mask SAM cho 3 vật (Part 5)

Đã lưu **3 mask click** từ SAM 2.1 Tiny (`sam2.1_t.pt`, CPU); số click và diện tích dưới đây lấy từ JSON cùng tên, quan sát biên lấy từ ảnh overlay JPG. Đây là kết quả riêng của Part 5, không phải 10 mask box-prompt ở Part 2.

| Ảnh / vật | SAM | Click dương / âm | Quan sát overlay so với guideline 4.2 | File trong `masks/` |
|---|---|---|---|---|
| `img_004.jpg` / tuk-tuk xanh ở tiền cảnh | 2.1 Tiny | 2 / 0 = **2** | Mask phủ thân và mui xe, biên tổng thể bám vật; không thấy gộp người đứng bên phải hoặc hàng hoá. Vùng cửa/kính tối cần đối chiếu quy tắc lỗ và kính, không coi mọi vùng tối là nền. | `tuktuk_blue_img004.jpg`, `.json` |
| `img_013.jpg` / tuk-tuk có cabin | 2.1 Tiny | 2 / 1 = **3** | Mask bao xe nhưng **còn dính người ngồi trong khoang**; click âm chưa tách sạch. Không nên coi người là phần mask xe; cần tách person và kiểm lại khoảng trống cabin/kính. | `tuktuk_img013.jpg`, `.json` |
| `img_017.jpg` / tuk-tuk phía trước | 2.1 Tiny | 2 / 0 = **2** | Mask bám thân, mui và bánh xe; hai người đi bộ bên trái không nằm trong mask. Không thấy hàng hoá; các vùng tối trong cabin cần kiểm theo quy tắc kính/lỗ, không kết luận chỉ từ màu overlay. | `tuktuk_front_img017.jpg`, `.json` |

Ba JSON ghi lần lượt **92.773 / 275.125 / 84.963 pixel mask**. Đây là diện tích mask, không nói lên chất lượng biên. Với mask xe ba bánh, tôi giữ phần thân và kính thật, nhưng không tính người ngồi hoặc khoảng cửa mở vào mask xe. Ở `img_013`, mask vẫn dính người dù đã có click âm; tôi chưa có phép so sánh riêng với CVAT Interactor.

## 6. Calibration (Part 6)

Đã có **3 quyết định** trong `submission/qc/calibration_decisions.csv`, cùng quy tắc và ảnh ví dụ tương ứng trong `submission/guidelines/guideline_v1.1_calibrated.md`:

| Ca / nguồn | Ảnh | Quyết định đã ghi trong CSV và v1.1 |
|---|---|---|
| `CAL-01` / `loi_may` | `img_011` | Không nhận `three_wheeler` chỉ theo confidence YOLOE; xét bánh nhìn thấy và cấu trúc xe, không suy diễn bánh bị che; còn mơ hồ thì chọn lớp có bằng chứng mạnh nhất và tick `can_review` (mục 4.3). |
| `CAL-02` / `cau_hoi_mo` | `img_013`, `img_017` | Kính thật thuộc mask xe; khoảng cửa/cửa sổ mở và người ngồi không thuộc mask xe. Giữ ngoại lệ lỗ nhỏ ~5% của v1.0 (mục 4.2). |
| `CAL-03` / `bay_lot` | `img_005` | Sau khi kiểm từng xe/người, quét độc lập toàn ảnh để tìm vật hoàn toàn chưa có box; ảnh đông có thể chia bốn vùng. Không đổi quy tắc `ignore_region`. |

Ba ca xử lý ba vấn đề khác nhau: CAL-01 là nhầm lớp do model gợi ý, CAL-02 là câu hỏi chưa rõ về kính và khoảng cửa, CAL-03 là cách rà vật bị bỏ sót. Tôi giữ nguyên định nghĩa lớp trong guideline v1.0 và bổ sung cách xử lý vào v1.1.

## 7. QC và quyết định (Part 6)

`qc_sample.csv` chọn **8 ảnh** (3 ca khó và 5 ảnh ngẫu nhiên), gồm **216 đối tượng**. Tôi tự kiểm các ảnh này, không có người kiểm độc lập.

| Mốc | Lỗi theo đối tượng | Ảnh có lỗi | Kết quả từ script |
|---|---|---|---|
| Vòng 1 | **1/216 = 0,46%** | **1/8 = 12,5%** | `qc_report_round1.md` gợi ý **REWORK** vì 12,5% > ngưỡng 2%. Lỗi `sai_lop` ở `img_018.jpg`, object #199: xe trắng bị gán `three_wheeler` thay vì `bus`. |
| Sau sửa | **0/216 = 0%** | **0/8 = 0%** | `qc_report_round2.md` và `qc_report_auto.md` gợi ý **RELEASE** sau khi tôi sửa nhãn trong `qc_fixed.json` và tạo lại bộ `release/`. |

**Giới hạn vòng 2:** phiếu `qc_review_round2.csv` giữ bảy dòng `khong` của vòng 1 và chỉ ghi kiểm lại `img_018.jpg` sau sửa. Vì vậy **0/8 là trạng thái phiếu QC sau sửa, không phải một lần kiểm độc lập lại cả 8 ảnh**. `qc_report.py` chỉ tổng hợp các lỗi người kiểm đã ghi trên phiếu; nó không tự xác minh các vật thể ngoài phiếu hay tính đúng ngữ nghĩa của nhãn. Mẫu tự kiểm nhỏ: cận trên Wilson 95% cho tỉ lệ ảnh lỗi là **47%** ở vòng 1 và **32%** sau sửa; 0/8 không chứng minh tỉ lệ lỗi thật dưới 2%. `release_manifest.json` ghi **24 ảnh, 531 đối tượng**, chỉ `img_018.jpg` được thay bởi QC fix, taxonomy vẫn v1.0; guideline v1.1 chỉ là phần bổ sung.

**Quyết định: Release** (18/11/2026), sau Rework. Tôi sửa xe trắng trong `img_018` từ `three_wheeler` sang `bus` rồi kiểm lại ảnh đó. Phiếu QC sau sửa ghi 0/8 ảnh lỗi; vì mẫu nhỏ và bảy ảnh còn lại giữ kết quả kiểm trước đó, tôi không xem đây là bằng chứng toàn bộ lô có tỉ lệ lỗi dưới 2%.

## 8. Ghi chú cá nhân

- **Công cụ sử dụng:** Tôi chạy YOLO26n, YOLO26s, YOLOE-26s và SAM 2.1 Tiny để thử model và tạo gợi ý nhãn. ChatGPT hỗ trợ lệnh PowerShell, đọc output và biên tập báo cáo; Codex hỗ trợ sửa script và viết test. Tôi kiểm lại các kết quả bằng log, file nhãn và ảnh overlay.
- **Ảnh Pilot:** Tôi có gửi ảnh minh họa kết quả model ở `img_002`, `img_011`, `img_032` cho ChatGPT để cùng xem lỗi. Tôi chưa có xác nhận từ giảng viên về việc chia sẻ ảnh này và cần làm rõ theo `RULES.md`.
- **Gold:** Bốn ảnh Gold được tôi gán tay và export trước khi chạy model trên tập này. Tôi không có log độc lập cho toàn bộ thao tác trước lúc export.
- **Máy / CVAT:** Windows PowerShell; PyTorch dùng CPU (`--device cpu`), Python 3.12 trong `C:\My Future\Day08_CPU\.venv`; CVAT Docker local **v2.81.0**, `http://localhost:8080`. Đã dùng hard link trọng số model vào thư mục `models/` của repo (không commit trọng số). Có `FutureWarning` về `torch.jit.load` khi chạy YOLOE nhưng quá trình suy luận hoàn thành.
- **Dùng cache `data/cache/`?** **Không:** hai model đã được chạy trực tiếp trên 36 ảnh (`y26_all`, `y26s_all`, `yoloe_all`) ở confidence 0.05.
- **Sửa script:** Tôi dùng Codex để sửa các bước kiểm tra COCO, QC, nạp model và xuất mask SAM; có thêm test hồi quy. Trước khi đưa vào `main`, 18/18 test đã chạy PASS. Các thay đổi này không sửa dữ liệu nhãn đã export.
- **Cảnh báo nhãn:** Box `car` #278 tại `img_023.jpg` cao **15,4 px**, dưới mốc 20 px của guideline. Box này nằm gần mép ảnh và đã được đánh dấu `can_review`; tôi vẫn giữ cảnh báo vì chưa có cơ sở xác nhận là ngoại lệ hợp lệ.
- **Lần sau tôi sẽ:** bấm giờ ngay từ đầu, quét toàn ảnh thêm một lượt để tìm vật chưa có box, và lưu kết quả QC trước/sau sửa riêng để tiện đối chiếu.
