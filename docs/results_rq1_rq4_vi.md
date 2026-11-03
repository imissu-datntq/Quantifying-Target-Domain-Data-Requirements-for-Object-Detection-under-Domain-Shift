# Kết quả RQ1–RQ4: đường cong hiệu quả dữ liệu target (2026-11-03)

Grid supervised gồm 87 run, chạy xong trên Kaggle T4 qua 3 session, kernel `kizhowl/bdd100k-campaign` v1–v3. Test được chấm **đúng một lần**:

- Plan đóng băng: `runs/campaign/final_plan.json`, sha256 `8244023f…5ff7a8`.
- 87 bộ prediction, 176 lượt chấm.
- Bootstrap dùng 1.000 lần lấy mẫu lại ảnh test, seed 20261030, chung cho mọi run trong cùng một domain.

Số liệu gốc nằm trong `runs/campaign/analysis/` (CSV, `summary.json`, hình `recovery_<domain>.png`).

## Thiết lập

- **Model:** Faster R-CNN R50-FPN, ảnh gốc 1280×720. Mọi run target khởi tạo từ source model (day+clear) và giữ update cuối, không chọn checkpoint.
- **Schedule chung,** chốt trên lighting val trước khi chạy (xem `.agent/DECISIONS.md`):
  - số update = max(500, 2 epoch); LR 1e-3; batch 2; AMP; warmup 50; cosine.
  - Hệ quả: N ≤ 500 được 500 update, còn N ≥ 1000 được N update.
- **Seed:** 3 seed lấy mẫu cho mỗi bậc N; ceiling (fine-tune trên toàn pool) chạy 2 seed.
- **Metric chính:** mAP50:95 trung bình trên 4 lớp reliable (pedestrian, car, traffic light, traffic sign).
- **Recovery(N)** = (mAP(N) − mAP(0)) / (mAP_c − mAP(0)), với mẫu số tối thiểu 0,01, không cắt. Khoảng tin cậy 95% lấy từ bootstrap phân cấp: ảnh test, seed lấy mẫu và seed ceiling.
- **Kiểm tra sanity:**
  - Source model trên source test đạt 0,343 (trên source val là 0,352).
  - Recovery tại toàn pool bằng đúng 1 ở cả 3 shift.
  - Không run nào đọc nhãn test trước khi chấm.

## RQ1: domain shift làm giảm bao nhiêu?

| Shift | mAP source | mAP(0) zero-shot | Drop (CI 95%) | Ceiling mAP_c | Gap còn lại sau ceiling |
|---|---:|---:|---:|---:|---:|
| lighting (night+clear) | 0,343 | 0,246 | 0,097 (0,090–0,105) | 0,294 | 0,048 |
| weather (day+rainy) | 0,343 | 0,321 | 0,021 (0,011–0,032) | 0,334 | 0,009 |
| compound (night+rainy) | 0,343 | 0,220 | 0,123 (0,112–0,133) | 0,256 | 0,087 |

- Drop do weather nhỏ nhưng có thật: CI không chứa 0.
- **Tương tác:** drop(compound) − drop(lighting) − drop(weather) = +0,005 (CI −0,010 đến +0,018).
  - Tức là compound **gần như cộng tính**; không có bằng chứng siêu cộng tính.
- Kể cả khi fine-tune trên toàn pool, model target vẫn thấp hơn model source trên source test. Gap còn lại lớn nhất ở compound (0,087).
  - Pool compound chỉ có 1.738 ảnh, nên một phần gap này có thể do thiếu dữ liệu chứ không chỉ do độ khó.

## RQ2: đường cong recovery

Recovery của reliable mAP, so với ceiling toàn pool. Mỗi điểm là trung bình 3 seed, kèm CI 95%.

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 10 | −1,27 (−1,34; −1,20) | −4,53 (−5,81; −3,34) | −1,44 (−1,74; −1,22) |
| 25 | −0,86 | −2,87 | −0,94 |
| 50 | −0,41 | −1,83 | −0,42 |
| 100 | 0,00 (−0,04; 0,04) | −0,63 (−1,07; −0,26) | 0,09 (−0,04; 0,22) |
| 250 | 0,31 (0,27; 0,36) | 0,13 (−0,08; 0,29) | 0,50 (0,44; 0,56) |
| 500 | 0,37 (0,33; 0,41) | 0,44 (0,28; 0,56) | 0,55 (0,48; 0,62) |
| 1000 | 0,50 (0,47; 0,52) | 0,69 (0,57; 0,81) | 0,80 (0,74; 0,86) |
| 2000 | 0,62 (0,60; 0,65) | | |
| 5000 | 0,77 (0,74; 0,79) | | |
| 10000 | 0,89 (0,87; 0,91) | | |
| toàn pool | 1 (18.015) | 1 (1.985) | 1 (1.738) |

**Kết quả quan trọng nhất: với N nhỏ, fine-tune làm model tệ hơn zero-shot.**

- Recovery âm ở N ≤ 50 ở cả 3 shift, và ở N = 100 với weather.
- Pilot đã báo trước hiện tượng này: N=10 trên lighting val thấp hơn zero-shot.
- Nguyên nhân là schedule cố định: N ≤ 500 đều nhận 500 update, nên 10 ảnh bị lặp khoảng 100 epoch và model overfit.
- Vì vậy đây là kết quả về **fine-tune thường với một schedule cố định**, không phải giới hạn của dữ liệu.
- Nhóm em báo cáo đúng như luật đã ghi trước, không chỉnh lại schedule.

**Theo từng lớp.**

- Pedestrian luôn hồi phục chậm nhất và âm sâu nhất ở N nhỏ. Ví dụ ở lighting N=1000: pedestrian 0,37, còn car 0,54, traffic light 0,45, traffic sign 0,57.
- Các lớp còn lại đi gần nhau. Như vậy phần hồi phục không chỉ là học lại class prior; pedestrian cần thêm dữ liệu đặc trưng target.
- Ở weather, traffic light và traffic sign **không xác định được** recovery: ceiling chỉ hơn zero-shot 0,0095 và 0,0058 AP, dưới mẫu số tối thiểu 0,01.

## RQ3: cần bao nhiêu ảnh target?

Ngưỡng tính theo kiểu "bền vững": bậc N nhỏ nhất mà từ đó trở lên recovery luôn ≥ mức yêu cầu. N_BE là điểm hòa vốn, Recovery > 0.

| Shift | N_BE | N75 | N90 | N95 (phụ) |
|---|---|---|---|---|
| lighting | 250 (khoảng 100–250) | 5.000 (5.000–10.000) | > 10.000, chỉ toàn pool đạt | toàn pool |
| weather | 250 (250–500) | toàn pool (1.000–1.985) | toàn pool | toàn pool |
| compound | 100 (100–250) | 1.000 (1.000–1.738) | toàn pool | toàn pool |

Khoảng trong ngoặc lấy từ dải CI 95%: bậc nhỏ nhất mà cận trên đạt mức yêu cầu, và bậc nhỏ nhất mà cận dưới đạt.

**Fit lũy thừa** 1 − a(N^−b − N_c^−b) cho các ước lượng nằm giữa các bậc:

| Shift | N75 (CI) | N90 (CI) | b | R² |
|---|---:|---:|---:|---:|
| lighting | 2.770 (2.560–2.980) | 7.650 (7.310–7.990) | 0,25 | 0,988 |
| weather | 1.010 (840–1.180) | 1.490 (1.350–1.600) | 0,38 | 0,994 |
| compound | 710 (630–780) | 1.190 (1.120–1.240) | 0,24 | 0,987 |

**Cần thận trọng với fit.**

- R² cao chủ yếu là nhờ các điểm âm ở N nhỏ.
- Ở N lớn, fit dự đoán cao hơn thực tế. Ví dụ lighting N=5000: thực tế 0,76, còn fit cho 0,84.
- Vì vậy ngưỡng dựa trên fit là **lạc quan**. Nhóm em dùng bậc trên ladder và dải CI làm kết quả chính; fit chỉ để tham khảo.

## RQ4: so sánh giữa các shift với cùng cỡ pool (1.738 ảnh)

Lighting và weather có thêm một ceiling chỉ dùng 1.738 ảnh, bằng pool của compound. Với compound, ceiling này trùng với ceiling toàn pool.

| Shift | Recovery N=500 | Recovery N=1000 | N75 (ladder / fit) | N90 (ladder / fit) |
|---|---:|---:|---|---|
| lighting | 0,64 | 0,86 | 1.000 / 705 | 1.738 / 1.171 |
| weather | 0,47 | 0,75 | 1.738 / 942 | 1.738 / 1.336 |
| compound | 0,55 | 0,80 | 1.000 / 707 | 1.738 / 1.185 |

- Cùng cỡ pool, lighting và compound hồi phục gần như cùng tốc độ.
- Weather hồi phục chậm hơn và nhiễu hơn: gain từ zero-shot tới ceiling của weather chỉ khoảng 0,013 AP.
- Phần lớn khác biệt giữa các shift ở RQ3 đến từ **độ lớn ceiling và cỡ pool**, không phải từ tốc độ học.
- Lưu ý: ceiling matched-pool chỉ có 1 seed.

## Kết quả phụ

**Quên source (forgetting).** Thay đổi reliable mAP trên source test so với source model:

- Nặng nhất ở N=10, từ −0,052 đến −0,058, do overfit cùng nguyên nhân ở trên.
- Giảm còn khoảng −0,01 ở N = 250–1000.
- Lighting tăng nhẹ lại khi N lớn: −0,024 ở toàn pool, vì số update tăng.

**Đối chứng khởi tạo COCO,** chưa qua source; reliable mAP:

| Shift, N | COCO-init | Source-init | Zero-shot |
|---|---:|---:|---:|
| compound 100 / 1000 | 0,149 / 0,183 | 0,223 / 0,249 | 0,220 |
| weather 100 / 1000 | 0,208 / 0,255 | 0,313 / 0,330 | 0,321 |
| lighting 100 / 1000 | 0,172 / 0,207 | 0,246 / 0,270 | 0,246 |

Pretrain trên source chiếm phần lớn hiệu năng. Với cùng 1.000 ảnh target, COCO-init vẫn thấp hơn cả zero-shot.

## Hạn chế cần nêu

1. **E = 2 là giá trị nhỏ nhất đã thử** khi chốt schedule; cách E = 5 khoảng 0,003 AP trên lighting val. Điểm tối ưu có thể còn thấp hơn.
2. **Schedule cố định gây recovery âm ở N nhỏ**, và số update tăng theo N khi N ≥ 1000. Đường cong vì vậy đo "dữ liệu cộng với lượng huấn luyện tương ứng", không phải dữ liệu đơn thuần.
3. **Weather:** ceiling gain chỉ 0,013 AP; 44/1.000 lần bootstrap không xác định được recovery. Mọi kết luận về weather phải đi kèm CI và mức thay đổi thô.
4. **Ceiling là ceiling của schedule** (2 epoch trên toàn pool), không phải giới hạn của detector. Ceiling chạy 2 seed; matched-pool chỉ 1 seed.
5. **Nhãn** là BDD100K bản 2018. Bản này khớp hash với một bản copy độc lập, nhưng chưa đối chiếu với file gốc của Berkeley.

## Bước tiếp theo

- RQ5: chọn một phương pháp domain adaptation công khai chạy trên torchvision Faster R-CNN R50-FPN, rồi so với đường cong này. Phương pháp unsupervised cho một đường ngang; semi-supervised cho một đường cong thứ hai.
- Có thể bổ sung, nếu thầy yêu cầu: chạy lại N ≤ 100 với số update tỉ lệ theo N, để tách tác động của overfit. Phần này là phân tích phụ, không thay kết quả chính đã đăng ký trước.
