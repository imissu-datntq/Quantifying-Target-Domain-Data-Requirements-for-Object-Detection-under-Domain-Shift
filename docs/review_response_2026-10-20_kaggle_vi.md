# Cập nhật 2026-10-20 — chạy GPU trên Kaggle

Chạy trên tài khoản Kaggle `kizhowl`, mọi kernel và dataset đều ở chế độ private.
Cách dựng lại dataset trên Kaggle được mô tả trong `docs/kaggle_training.md`.

## Kiểm tra dataset trên Kaggle (kernel CPU `bdd100k-bootstrap-check`)

- Cả 79.863 ảnh lấy từ mirror `solesensei/solesensei_bdd100k` đều được băm lại,
  và khớp SHA256 trong manifest.
- 7 file protocol (annotation, manifest, metadata) khớp hash của bản local.
- Kiểm tra cấu trúc đạt: 55.000/14.863/10.000 ảnh, không trùng basename/SHA256
  giữa các split.
- 18/18 unit test đạt; dry-run thành công. Kernel này không dùng quota GPU.

## Pilot GPU (T4, kernel `bdd100k-gpu-pilot`)

Môi trường: Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0, Tesla T4.

| Cấu hình | Ảnh/s khi train | Thời gian/update (ổn định) | VRAM đỉnh | Ảnh/s khi đánh giá val |
| --- | ---: | ---: | ---: | ---: |
| batch 2, AMP | 6,1 | 0,31 s | 2,5 GiB | 12,1 |
| batch 4, AMP | 6,4 | 0,60 s | 4,5 GiB | 11,7 |
| batch 2, FP32 | 2,8 | 0,71 s | 2,4 GiB | 7,4 |

- AMP nhanh gấp khoảng 2,3 lần FP32.
- Pilot phát hiện một lỗi thật và đã sửa. Khi dùng AMP, GradScaler có thể sinh
  gradient vô hạn ở vài bước đầu. Đây là hành vi bình thường: bước đó bị bỏ qua.
  Nhưng code cũ lại dừng toàn bộ run ở `clip_grad_norm_`. Code mới bỏ qua bước đó
  và ghi lại `amp_step_skipped`.
## Train source model (kernel `bdd100k-source-train`)

Cấu hình: Faster R-CNN R50-FPN, khởi tạo COCO, ảnh gốc 1280×720, batch 4, AMP,
12 epoch (29.412 update), SGD với LR 0,005, warmup 500 update rồi cosine. Checkpoint
được chọn bằng reliable mAP50:95 trên 1.764 ảnh val daytime-clear.

- Thời gian train 5,1 giờ trên T4 (6,39 ảnh/s), VRAM đỉnh 4,5 GiB.
- 11 bước AMP bị bỏ qua (scale overflow), được xử lý đúng như thiết kế.
- Không đọc nhãn test.

| Update (epoch) | Reliable mAP50:95 | mAP50:95 10 lớp | AP50 10 lớp |
| ---: | ---: | ---: | ---: |
| 2.451 (1) | 0,306 | 0,217 | 0,432 |
| 9.804 (4) | 0,332 | 0,247 | 0,482 |
| 19.608 (8) | 0,346 | 0,268 | 0,505 |
| **26.961 (11, được chọn)** | **0,352** | 0,274 | 0,513 |
| 29.412 (12) | 0,351 | 0,273 | 0,515 |

Đường validation đã phẳng ở 3 epoch cuối.

AP50:95 theo lớp trên val ở update cuối: car 0,477; bus 0,408; truck 0,365;
traffic sign 0,334; pedestrian 0,314; traffic light 0,281; rider 0,190;
motorcycle 0,188; bicycle 0,176; train 0,000 (val chỉ có 2 box).

Số epoch train source dùng 12 (lịch "1x" chuẩn của Faster R-CNN) vì quota có hạn.

## Gate zero-shot chính thức bằng source model (kernel `bdd100k-gate-schedule`, phần gate)

Phép đo dùng `best.pt` của source và 4 lớp reliable. Source được chấm trên val, mỗi
target được chấm trên train pool của nó: model chưa từng thấy các ảnh này và không
đọc nhãn held-out nào. Mỗi domain có tối đa 2.000 ảnh; CI tính bằng 1.000 lần
bootstrap.

| Domain | Ảnh | Reliable mAP50:95 | Gap | CI 95% của gap | Gap tương đối | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Source (val) | 1.764 | 0,352 | — | — | — | — |
| Weather (day-rainy) | 1.985 | 0,322 | **0,031** | 0,023 – 0,040 | 8,8% | **qua gate** |
| Lighting (night-clear) | 2.000 | 0,253 | 0,099 | 0,090 – 0,109 | 28,2% | qua gate |
| Compound (night-rainy) | 1.738 | 0,223 | 0,129 | 0,121 – 0,138 | 36,7% | qua gate |

Gap theo lớp của weather: pedestrian 0,054; traffic light 0,035; traffic sign
0,021; car 0,014.

**Quyết định theo quy tắc đã đăng ký trước (ngưỡng 0,01 AP):** weather **ở lại**
nghiên cứu chính. Gap của nó có thật nhưng nhỏ, chỉ khoảng 1/3 gap của lighting, nên
lo ngại của reviewer vẫn đúng một phần. Mẫu số của Recovery cho weather có thể nhỏ,
và chặn mẫu số tối thiểu 0,01 sẽ được áp dụng. Kết quả weather phải luôn đi kèm CI
và raw change.

Lưu ý: nhãn của source model đến từ val, còn của target đến từ train pool. Hai tập
này cùng phân bố theo stratum, nhưng không phải cùng một tập ảnh. Gap trên test chỉ
được đo khi đánh giá cuối cùng.
