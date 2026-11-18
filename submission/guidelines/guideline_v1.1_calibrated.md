# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202604047 - Phạm Hải

> Bản này **chỉ bổ sung / làm rõ** guideline v1.0, không thay `submission/taxonomy.json`, ngưỡng QC hay nhãn đã export. Các quyết định tương ứng từng dòng trong `submission/qc/calibration_decisions.csv`. Quy tắc gán nhãn gốc của v1.0 vẫn có hiệu lực.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # (mục 6 v1.0) | Câu hỏi | Quyết định bổ sung | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 5 | Kính chắn gió / cửa sổ: phần nào thuộc mask xe khi có thể nhìn xuyên qua cabin? | Kính **thực sự gắn trên xe**, dù trong suốt, thuộc mask xe. Khoảng **cửa/cửa sổ mở không có kính** không thuộc mask xe; người lái/hành khách nhìn thấy là vật thể `person` riêng, không gộp vào mask xe. Giữ quy tắc lỗ nhỏ dưới khoảng 5% diện tích vật ở v1.0 mục 4.2.2. | `img_013`, `img_017` | CAL-02 |

**Các câu hỏi mở khác trong mục 6 v1.0:** chưa chốt ở vòng calibration này nếu thiếu ví dụ xác minh đủ rõ; giữ nguyên v1.0 và ghi `can_review` cho ca chưa chắc. Riêng CAL-01 là làm rõ cách áp dụng mục 4.3; không thay định nghĩa lớp.

## 2. Ca khó mới (từ lỗi máy, bẫy lọt và mask SAM)

| Ảnh | Tình huống / nguồn | Quan sát | Quyết định và cách kiểm tra bổ sung | Liên hệ guideline v1.0 / case |
|---|---|---|---|---|
| `img_011` | Xe lớn bị YOLOE gợi ý nhãn `three_wheeler` (`loi_may`) | Người gán nhớ đây có thể là `bus` hoặc `truck`, **chưa xác nhận chính xác lớp nào**. | Không tin nhãn chỉ vì confidence. Quan sát số bánh **nếu nhìn thấy rõ** cùng cấu trúc khung gầm, đầu xe, thân và khoang chở người/hàng. Khi bánh bị che, không suy đoán số bánh từ phần không nhìn thấy; dựa vào các đặc trưng cấu trúc còn nhìn được. Nếu vẫn mơ hồ, chọn lớp có bằng chứng mạnh nhất và tick `can_review`. | Làm rõ **mục 4.3**; CAL-01 |
| `img_013`, `img_017` | Mask xe ba bánh có cabin; `img_013` còn dính người (`cau_hoi_mo`) | SAM có thể gom cả người hoặc để lỗ ở cabin. | Mask của xe gồm phần xe và kính thật; **không** gồm người ngồi hay khoảng trống cửa mở không có kính. Mỗi người đủ điều kiện gán nhãn là một `person` riêng. Khi xử lý lỗ nhỏ tuân theo ngưỡng đã có ở v1.0 mục 4.2.2. | Làm rõ **mục 4.2.3** và câu hỏi mở số 5; CAL-02 |
| `img_005` | Ảnh dày xe máy; còn một bẫy `missing` sau rework (`bay_lot`) | Người gán thường sửa từng xe máy kèm người nhìn thấy rồi mới chuyển sang xe khác; cách này có thể bỏ qua cả một phương tiện. | **Giữ** cách kiểm tra theo từng phương tiện cùng người liên quan. Sau khi xử lý hết các xe đã nhận ra, thực hiện thêm **một lượt quét độc lập toàn ảnh để tìm vật hoàn toàn chưa có box**; với cảnh đông có thể quét lần lượt theo bốn vùng. Đây là quy tắc **quy trình review**, không phải thay đổi định nghĩa `ignore_region` hay ngưỡng chiều cao 20 px. | Bổ sung checklist review (bước tìm `missing`), vẫn tuân **mục 4.1.6**; CAL-03 |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do / `case_id` |
|---|---|---|
| v1.0 | Bản do giảng viên cung cấp. | — |
| v1.1 | Làm rõ cách phân biệt xe ba bánh với xe lớn khi bánh bị che; quy định ranh giới kính, cửa mở và người trong mask; thêm lượt quét tìm vật bỏ sót sau khi review từng xe. | CAL-01, CAL-02, CAL-03 |

**Giới hạn:** Các ca được ghi theo quan sát và phản hồi thực tế; `img_011` chưa được xác minh là `bus` hay `truck`. Phần bẫy `img_005` được phát hiện qua đánh giá sau review, không được trình bày là đã bắt đúng trong lượt review đầu.
