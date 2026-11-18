# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 429; đối tượng sau duyệt: 521; ngưỡng accepted: IoU >= 0.9
- Đã **loại 21 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 280 | 280/429 = 65.3% | 429 box máy |
| edited (sửa) | 99 | 99/429 = 23.1% | 429 box máy |
| deleted (xoá) | 50 | 50/429 = 11.7% | 429 box máy |
| added (vẽ thêm) | 142 | 142/521 = 27.3% | 521 đối tượng sau duyệt |

Trong số deleted + added, **8 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 134.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 177 | 32 | 18 | 64 |
| bicycle | 1 | 5 | 1 | 6 |
| motorbike | 52 | 52 | 21 | 45 |
| car | 30 | 5 | 3 | 19 |
| bus | 3 | 0 | 1 | 2 |
| truck | 5 | 1 | 1 | 2 |
| three_wheeler | 12 | 4 | 5 | 4 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 58 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 45 | 15 | 13 | 75.0% (KTC 95%: 63–84%) | 77.6% (KTC 95%: 65–86%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 11/12 = 91.7% (KTC 95%: 65–99%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - missing: 2/3
  - extra: 3/3
  - loose: 3/3
  - wrong_class: 3/3
  - lọt: img_005.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
