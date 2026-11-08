# Tin nhắn gửi thầy (bản nháp, 2026-10-30)

Gửi qua Zalo. Đính kèm `review_response_2026-10-19_vi.md` và `review_response_2026-10-20_kaggle_vi.md`.

---

Em chào thầy ạ. Nhóm em đã sửa xong 6 góp ý hôm 16/10, chi tiết trong 2 file đính kèm. Em tóm tắt ngắn:

1. **Zero-shot gap của weather:** nhóm em đã đo bằng source model trước khi chạy ladder. Gap của weather là 0,031 AP (CI 95%: 0,023–0,040), nhỏ nhưng có thật, khoảng 1/3 gap của lighting (0,099). Compound là 0,129. Weather vẫn được giữ lại, nhưng mọi kết quả của weather đều kèm CI và mức thay đổi thô, đồng thời mẫu số Recovery có chặn tối thiểu 0,01.
2. **Test của weather/compound:** nhóm em gộp val vào test theo đề xuất của thầy, nên test là 933 / 756 ảnh, còn pool train giữ nguyên. Hai domain này không còn val, nên mọi run target đều lấy checkpoint ở update cuối của một schedule cố định chung.
3. **Reliable class:** dùng chung cho mọi domain gồm pedestrian, car, traffic light, traffic sign. Mean 10 lớp được giữ làm sensitivity analysis.
4. **Smoke test:** smoke test chỉ dùng để xác nhận pipeline chạy đúng, không dùng để chọn kiến trúc. Nhóm em chọn Faster R-CNN vì RQ5: hầu hết implementation domain adaptation công khai đều xây trên Faster R-CNN/FCOS. Throughput thật đo trên T4 là 6,4 ảnh/s.
5. **Nguồn nhãn:** nhãn là bản BDD100K 2018 (Kaggle mirror v2). Bản này khớp SHA256 tuyệt đối với một bản copy độc lập trên máy em. Tuy vậy, nhóm em chưa đối chiếu được với file gốc của Berkeley, vì server chính thức không truy cập được. Nhóm em giữ bản 2018, ghi rõ phiên bản và nêu ở phần hạn chế rằng bản này đã bị det_20 thay thế.
6. **Bảng occluded/truncated và traffic light/ảnh:** đã thêm vào EDA. Recovery cũng được tính theo từng lớp để tách phần học lại class prior khỏi phần học đặc trưng mới.

Source model đã train xong: reliable mAP50:95 trên source val là 0,352.

**Chốt schedule (pilot trên lighting val, không đụng tới test):**
- Luật chọn được ghi trước khi đọc kết quả.
- Với LR 5e-3 (giá trị dự kiến ban đầu), N=100 còn thấp hơn zero-shot. Vì vậy nhóm em hạ xuống LR 1e-3.
- Kết quả chốt: S = 500 update, E = 2 epoch, batch 2, 3 seed. Độ lệch giữa 3 seed ở N=100 là 0,0014 AP.
- E = 2 là giá trị nhỏ nhất đã thử. Nó cách giá trị tốt nhất (E = 5) 0,003 AP, nằm trong ngưỡng δ = 0,005. Nhóm em sẽ ghi rõ điểm này.
- Ở N=10, fine-tune làm kết quả thấp hơn zero-shot. Nhóm em báo cáo đây là một kết quả, không chỉnh lại schedule.

**Kết quả RQ1–RQ4** (grid 87 run đã chạy xong, test chấm đúng một lần; chi tiết trong `results_rq1_rq4_vi.md`):
- **Drop:** lighting 0,097, weather 0,021, compound 0,123. Compound gần như cộng tính: tương tác +0,005, CI −0,010 đến 0,018.
- **Fine-tune với N ≤ 50 làm model tệ hơn zero-shot,** do schedule cố định 500 update khiến model overfit. Điểm hòa vốn khoảng 100–250 ảnh.
- **N75:** compound 1.000 ảnh, lighting 5.000 ảnh. Weather chỉ đạt khi dùng toàn pool, vì gain của weather chỉ khoảng 0,013 AP.
- **Cùng cỡ pool 1.738 ảnh:** lighting và compound hồi phục gần như cùng tốc độ.
- **Pretrain trên source quan trọng:** khởi tạo COCO với 1.000 ảnh target vẫn thấp hơn zero-shot.

Nếu thầy thấy cần điều chỉnh gì, thầy báo giúp em ạ. Em cảm ơn thầy.
