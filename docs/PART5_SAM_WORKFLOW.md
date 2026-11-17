# Quy trình SAM 2.1 cho Part 5

Mục tiêu là tạo ít nhất 3 ảnh mask từ 3 vật do học viên tự chọn và tự click. Không dùng toạ độ mẫu như dữ liệu thật; mỗi click phải đến từ quan sát trực tiếp trên ảnh gốc.

## 1. Chọn vật và ghi click thật

Chọn ba vật trên ba ảnh phù hợp với mục 4.2 của guideline. Mở ảnh gốc trong trình xem có hiển thị toạ độ pixel. Với mỗi vật:

1. Bắt đầu bằng một click dương ở phần chắc chắn thuộc vật.
2. Nếu mask ăn sang nền/người/hàng, thêm click âm tại vùng sai.
3. Nếu mask thiếu một phần vật, thêm click dương trong phần bị thiếu.
4. Chỉ dùng `--box` khi chính bạn đã xác định box gợi ý từ ảnh.

## 2. Chạy và lưu bằng chứng

PowerShell, từ thư mục gốc repo:

```powershell
$PY = 'C:\My Future\Day08_CPU\.venv\Scripts\python.exe'
& $PY scripts/sam2_points.py --image img_XXX.jpg --pos "x,y" --sam sam2.1_t.pt --device cpu --out submission/masks --name img_XXX_vat_1
```

Thêm `--pos "x,y"`, `--neg "x,y"` hoặc `--box "x1,y1,x2,y2"` khi thực sự đã click/xác định box. Script ghi:

- `<name>.jpg`: overlay mask, biên cyan, click dương xanh, click âm đỏ và box vàng nếu có.
- `<name>.json`: model, thiết bị, toàn bộ click thật, số click và diện tích mask để đối chiếu REPORT.

Lặp lại cho đủ ít nhất ba vật. Không sửa số click trong JSON; nếu chạy lại, dùng tên khác để so sánh trước khi chọn kết quả cuối.

## 3. Tự đánh giá trước khi ghi REPORT

Với từng ảnh, tự trả lời:

- Biên mask có bám vật hay ăn nền?
- Theo guideline 4.2, mask có phải gồm người lái hoặc hàng hoá không?
- Click nào sửa được lỗi và tổng cộng đã dùng bao nhiêu click?
- Có lỗi nào lặp lại ở cả ba vật để đề xuất calibration không?

Chỉ sau khi nhìn ảnh overlay mới chép số click từ JSON và viết quan sát bằng lời của học viên vào REPORT mục 5.
