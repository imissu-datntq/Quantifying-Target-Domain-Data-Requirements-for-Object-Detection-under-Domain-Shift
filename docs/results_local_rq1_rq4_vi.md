# Kết quả RQ1–RQ4, chạy lại độc lập trên máy local (2026-11-06)

Toàn bộ nghiên cứu được chạy lại từ đầu trên máy local (RTX 3060 12 GB), không dùng Kaggle (`.agent/DECISIONS.md`, mục 2026-11-05). "Từ đầu" gồm:

- dựng lại split;
- train lại source model;
- chạy lại gate zero-shot và pilot chọn schedule;
- chạy grid 87 run.

Các luật đăng ký trước được giữ nguyên văn; mọi con số được tính lại từ đầu.

Test được chấm **đúng một lần** cho bản chạy local:

- Plan đóng băng: `runs/local/campaign/final_plan.json`, schema 3, sha256 `a1c5eb9f…9436a44`.
- 87 bộ prediction, 176 lượt chấm.
- Bootstrap dùng 1.000 lần lấy mẫu lại ảnh test, seed 20261030.

**Đây là lần mở nhãn test thứ hai.** Lần đầu là campaign Kaggle ngày 2026-11-03 (`docs/results_rq1_rq4_vi.md`). Bản local được báo cáo như một **lần lặp lại độc lập**: không gộp số với bản Kaggle, và không chọn bản nào dựa trên kết quả test.

Số liệu gốc nằm trong `runs/local/campaign/analysis/` (CSV, `summary.json`, hình `recovery_<domain>.png`).

## Khác biệt so với bản Kaggle

| | Kaggle (2026-11-03) | Local (2026-11-06) |
|---|---|---|
| Phần cứng | T4, 3 session | RTX 3060 12 GB, một máy |
| Split | frozen manifest | dựng lại; 7/7 hash SHA256 khớp `data/split_reference/` |
| Source model, source val | 0,352 | 0,3517 |
| Gate zero-shot (lighting / weather / compound) | 0,099 / 0,031 / 0,129 | 0,100 / 0,032 / 0,129 |
| Schedule chốt bằng pilot | S=500, **E=2**, LR 1e-3 | S=500, **E=3**, LR 1e-3 |

**Schedule khác nhau ở E.** Cùng một luật (DECISIONS 2026-10-30) chọn E nhỏ nhất nằm trong δ = 0,005 so với E tốt nhất.

- Trên lighting val local: E2 kém E5 0,00511, tức nằm **ngoài δ đúng 0,0001**. Vì vậy luật chọn E3.
- Trên Kaggle: E2 chỉ kém E5 khoảng 0,003, nên luật chọn E2.

Nhóm em áp dụng luật đúng như đã viết, không tranh luận lại. Hệ quả:

- N ≤ 333 nhận 500 update.
- N ≥ 500 nhận 1,5·N update (3 epoch ở batch 2).
- Run local vì vậy huấn luyện lâu hơn bản Kaggle ở N ≥ 500.

## Thiết lập

Giống bản Kaggle, trừ E = 3:

- **Model:** Faster R-CNN R50-FPN ở 1280×720, khởi tạo từ source model, giữ update cuối.
- **Huấn luyện:** LR 1e-3, batch 2, AMP, warmup 50.
- **Seed:** 3 seed lấy mẫu cho mỗi bậc N; 2 seed cho ceiling.
- **Metric chính:** reliable mAP50:95 (pedestrian, car, traffic light, traffic sign).
- **Recovery:** cùng định nghĩa, mẫu số tối thiểu 0,01.

**Kiểm tra sanity:**

- Source model trên source test đạt 0,342 (Kaggle 0,343).
- Recovery tại toàn pool bằng 1 ở cả 3 shift.
- Không lần bootstrap nào cho recovery không xác định ở metric chính.

## RQ1: domain shift làm giảm bao nhiêu?

| Shift | mAP(0) zero-shot | Drop (CI 95%) | Drop Kaggle | Ceiling mAP_c | Gap còn lại |
|---|---:|---:|---:|---:|---:|
| lighting | 0,244 | 0,098 (0,092–0,106) | 0,097 | 0,296 | 0,046 |
| weather | 0,320 | 0,022 (0,012–0,032) | 0,021 | 0,336 | 0,007 |
| compound | 0,219 | 0,123 (0,112–0,133) | 0,123 | 0,257 | 0,085 |

- Drop gần như trùng bản Kaggle tới chữ số thứ ba. Điều này hợp lý vì cùng split, cùng công thức source, và source model local gần như bằng bản Kaggle.
- **Tương tác** drop(compound) − drop(lighting) − drop(weather) = +0,002 (CI −0,012 đến +0,016).
  - Vẫn **gần như cộng tính**, như ở Kaggle (+0,005).
- Ceiling local cao hơn Kaggle một chút ở cả 3 shift (0,296 / 0,336 / 0,257 so với 0,294 / 0,334 / 0,256), phù hợp với việc huấn luyện 3 epoch thay vì 2.

## RQ2: đường cong recovery

Recovery của reliable mAP so với ceiling toàn pool. Mỗi điểm là trung bình 3 seed, kèm CI 95% (bootstrap phân cấp).

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 10 | −1,17 (−1,23; −1,10) | −3,51 (−4,67; −2,71) | −1,33 (−1,59; −1,07) |
| 25 | −0,78 | −2,26 | −0,90 |
| 50 | −0,36 | −1,39 | −0,36 |
| 100 | 0,01 (−0,03; 0,05) | −0,42 (−0,85; −0,10) | 0,09 (−0,01; 0,20) |
| 250 | 0,33 (0,28; 0,36) | 0,22 (0,05; 0,35) | 0,52 (0,46; 0,58) |
| 500 | 0,44 (0,41; 0,46) | 0,50 (0,39; 0,60) | 0,66 (0,61; 0,71) |
| 1000 | 0,55 (0,53; 0,57) | 0,77 (0,66; 0,86) | 0,87 (0,82; 0,91) |
| 2000 | 0,66 (0,63; 0,69) | | |
| 5000 | 0,80 (0,78; 0,83) | | |
| 10000 | 0,91 (0,90; 0,93) | | |
| toàn pool | 1 (18.015) | 1 (1.985) | 1 (1.738) |

**Kết quả chính của bản Kaggle được lặp lại:** với N nhỏ, fine-tune làm model tệ hơn zero-shot.

- Recovery âm ở N ≤ 50 ở cả 3 shift, và ở N = 100 với weather.
- Pilot local cũng báo trước điều này: N=10 trên lighting val thấp hơn N0 tới 0,05–0,06 AP.
- Nguyên nhân vẫn là schedule cố định: 10 ảnh nhận 500 update, tức khoảng 100 epoch.

**So với Kaggle,** đường local cao hơn ở mọi bậc N ≥ 250, nhiều nhất ở N = 500–1000 (ví dụ compound N=1000: 0,87 so với 0,80; lighting N=1000: 0,55 so với 0,50). Đây là chênh lệch kỳ vọng khi mỗi bậc được huấn luyện 3 epoch thay vì 2. Ở N ≤ 100, hai bản cùng nhận 500 update; lighting và compound gần nhau (chênh ≤ 0,11 đơn vị recovery). Weather chênh nhiều hơn (N=10: −3,51 so với −4,53), vì ở đó một khác biệt nhỏ về mAP bị khuếch đại bởi mẫu số chỉ 0,013–0,015 AP.

**Theo từng lớp** (recovery tại N=1000):

| Shift | pedestrian | car | traffic light | traffic sign |
|---|---:|---:|---:|---:|
| lighting | 0,46 | 0,60 | 0,51 | 0,58 |
| weather | 0,82 | 0,82 | 0,78 | không xác định |
| compound | 0,87 | 0,88 | 0,88 | 0,82 |

- Ở lighting, pedestrian vẫn hồi phục chậm nhất, như bản Kaggle.
- Ở compound và weather tại N=1000, bốn lớp đi gần nhau. Kết luận "pedestrian chậm nhất" vì vậy chỉ đứng vững cho lighting.
- Weather traffic sign **không xác định**: ceiling chỉ hơn zero-shot 0,0089 AP, dưới mẫu số tối thiểu 0,01. Ở Kaggle cả traffic light lẫn traffic sign đều không xác định.

## RQ3: cần bao nhiêu ảnh target?

Ngưỡng tính theo kiểu bền vững, giống bản Kaggle. Khoảng trong ngoặc lấy từ dải CI 95%.

| Shift | N_BE | N75 | N90 | N95 (phụ) | Kaggle (N_BE / N75 / N90) |
|---|---|---|---|---|---|
| lighting | 100 (100–250) | 5.000 (5.000) | 10.000 (10.000–toàn pool) | toàn pool | 250 / 5.000 / toàn pool |
| weather | 250 (250) | **1.000** (1.000–toàn pool) | toàn pool | toàn pool | 250 / toàn pool / toàn pool |
| compound | 100 (100–250) | 1.000 (1.000) | toàn pool (1.000–1.738) | toàn pool | 100 / 1.000 / toàn pool |

**Thay đổi so với Kaggle:**

- **Weather N75 = 1.000.** Trên Kaggle cần cả pool. Recovery N=1000 lên 0,77 (Kaggle 0,69), vừa qua 0,75. Cận dưới CI là 0,66, nên ngưỡng này chưa chắc chắn: dải CI kéo tới toàn pool.
- **Lighting N90 = 10.000.** Recovery N=10.000 là 0,91 (Kaggle 0,89).
- **Lighting N_BE = 100.** Recovery N=100 là +0,01, CI chứa 0. Đây là hòa vốn sát nút, không phải lợi ích rõ ràng.

Mọi thay đổi đi cùng hướng với việc huấn luyện thêm một epoch. Nhóm em không cho rằng chúng mâu thuẫn với bản Kaggle: mỗi bản là một đường cong của **một schedule cụ thể**.

**Fit lũy thừa** 1 − a(N^−b − N_c^−b), chỉ để tham khảo:

| Shift | N75 (CI) | N90 (CI) | b | R² |
|---|---:|---:|---:|---:|
| lighting | 2.530 (2.340–2.720) | 7.280 (6.940–7.610) | 0,26 | 0,993 |
| weather | 950 (800–1.110) | 1.440 (1.320–1.550) | 0,35 | 0,994 |
| compound | 680 (610–750) | 1.160 (1.100–1.220) | 0,24 | 0,988 |

Fit vẫn lạc quan so với ladder, cùng lý do như bản Kaggle: R² cao chủ yếu nhờ các điểm âm ở N nhỏ. Ví dụ ladder lighting N75 = 5.000, còn fit cho 2.530.

## RQ4: so sánh giữa các shift với cùng cỡ pool (1.738 ảnh)

| Shift | Recovery N=500 | Recovery N=1000 | N75 (ladder / fit) | N90 (ladder / fit) |
|---|---:|---:|---|---|
| lighting | 0,69 | 0,88 | 1.000 / 690 | 1.738 / 1.160 |
| weather | 0,54 | 0,81 | 1.000 / 880 | 1.738 / 1.300 |
| compound | 0,66 | 0,87 | 1.000 / 680 | 1.738 / 1.160 |

- Như bản Kaggle: cùng cỡ pool, lighting và compound hồi phục gần như cùng tốc độ.
- Weather vẫn chậm hơn ở N=500 nhưng bắt kịp ở N=1000. Bản local có N75 = 1.000 cho cả 3 shift (Kaggle: weather cần 1.738).
- Kết luận RQ4 giữ nguyên: khác biệt giữa các shift ở RQ3 chủ yếu đến từ **độ lớn ceiling và cỡ pool**, không phải tốc độ học.
- Ceiling matched-pool vẫn chỉ có 1 seed.

## Kết quả phụ

**Quên source (forgetting).** Thay đổi reliable mAP trên source test so với source model (trung bình seed):

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 10 | −0,055 | −0,058 | −0,055 |
| 100 | −0,019 | −0,018 | −0,022 |
| 1000 | −0,012 | −0,006 | −0,016 |
| toàn pool | −0,026 | −0,005 | −0,018 |

Kiểu hình giống Kaggle: nặng nhất ở N=10, nhẹ nhất ở N = 250–1000, tăng nhẹ lại ở lighting khi N lớn vì số update tăng.

**Đối chứng khởi tạo COCO** (reliable mAP):

| Shift, N | COCO-init | Source-init | Zero-shot |
|---|---:|---:|---:|
| compound 100 / 1000 | 0,149 / 0,198 | 0,223 / 0,252 | 0,219 |
| weather 100 / 1000 | 0,209 / 0,267 | 0,314 / 0,332 | 0,320 |
| lighting 100 / 1000 | 0,171 / 0,220 | 0,244 / 0,273 | 0,244 |

Kết luận giữ nguyên: pretrain trên source chiếm phần lớn hiệu năng; với 1.000 ảnh target, COCO-init vẫn thấp hơn zero-shot.

## Hạn chế cần nêu

1. **E = 3 được chọn với biên rất mỏng** (E2 nằm ngoài δ 0,0001). Đường cong local và Kaggle là hai schedule khác nhau, không phải hai lần đo của cùng một cấu hình.
2. **Lần mở nhãn test thứ hai.** Chỉ có thể diễn giải bản local như một lần lặp lại; không chọn giữa hai bản theo kết quả test.
3. **Schedule cố định gây recovery âm ở N nhỏ**, và số update tăng theo N khi N ≥ 500. Đường cong đo "dữ liệu cộng với lượng huấn luyện tương ứng".
4. **Weather:** ceiling gain chỉ 0,015 AP. Ngưỡng N75 = 1.000 có dải CI kéo tới toàn pool; luôn báo kèm CI và mức thay đổi thô.
5. **Ceiling là ceiling của schedule** (3 epoch trên toàn pool). Matched-pool chỉ 1 seed.
6. **Nhãn** là BDD100K bản 2018, như bản Kaggle.

## Bước tiếp theo

RQ5 (ALDI++ so với fine-tune) đã chạy xong 46 run trên máy local. Kết quả nằm ở `docs/results_rq5_vi.md` sau khi chấm một lần.
