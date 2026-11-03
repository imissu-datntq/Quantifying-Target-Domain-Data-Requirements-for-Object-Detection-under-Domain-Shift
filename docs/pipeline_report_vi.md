# Báo cáo xây pipeline Faster R-CNN

Đã xây pipeline bằng Python cho phần supervised trong proposal: source training,
target fine-tuning theo budget, full-pool reference, generic-pretrained control và
COCO evaluation dùng chung. Dataset đầu vào duy nhất là `data/bdd100k_revised/`.

## 1. Các file đã xây

| File Python | Chức năng |
| --- | --- |
| `scripts/train_frcnn.py` | Train source / fine-tune target / pretrained control |
| `scripts/pilot_frcnn.py` | Pilot nhỏ, đo thời gian training/inference |
| `scripts/evaluate_frcnn.py` | Đánh giá val hoặc final test qua frozen plan |
| `scripts/freeze_evaluation_plan.py` | Cố định checkpoint/domain/cấu hình trước mở test |
| `scripts/verify_pipeline_resume.py` | So sánh resume với chạy liên tục |
| `scripts/report_pipeline.py` | Tạo báo cáo này từ artifact xác minh |
| `scripts/bdd_pipeline/data.py` | Dataset, class ID, bbox, augmentation và budget |
| `scripts/bdd_pipeline/model.py` | Faster R-CNN ResNet-50 FPN, geometry cố định |
| `scripts/bdd_pipeline/training.py` | Optimizer, accumulation, schedule, checkpoint/resume |
| `scripts/bdd_pipeline/evaluation.py` | COCO AP50:95/AP50/AP75/AR100 và từng lớp |
| `scripts/bdd_pipeline/artifacts.py` | Ghi atomic, fingerprint, môi trường và RNG |

## 2. Hành vi theo proposal

- Kiểm tra manifest checksum, 55.000/14.863/10.000 ảnh, overlap và class order.
- RGB tensor [0,1], normalization đúng một lần trong model. Giữ 1280×720;
  FPN chỉ pad chiều cao lên 736 để chia hết cho 32.
- Class ID 1–10, background 0. Cùng sanitation bbox cho train và ground truth
  COCO: bỏ malformed/nonfinite/nonpositive, clip biên, bỏ box rỗng sau clip.
- Augment lật ngang ảnh và bbox trên train, tái lập bằng seed/epoch/tên ảnh.
- Target sample chỉ từ train, lồng nhau theo seed; val cố định. Full pool dùng
  cùng đường training và schedule như các rung budget nhỏ.
- Số optimizer updates = max(S, E × ceil(ceil(N/batch)/accumulation)).
  Microbatch cuối ngắn được cân theo số ảnh thực; ghi actual image-passes.
- Ghi loss components, LR, gradient norm, thời gian, ảnh đã xử lý và metric val.
- Checkpoint lưu model/optimizer/scaler/RNG/epoch/vị trí/step/LR state/best score.
  Atomic writes; resume phải khớp cấu hình, code fingerprint, package và dataset.
- Best checkpoint chọn trên validation; mAP50:95 của reliable classes là primary
  metric. Ngưỡng support và full-class/per-class metrics được ghi rõ.
- Final evaluator chỉ chạy sau khi khóa checkpoint/domain/metric policy; chặn
  smoke, checkpoint chưa hoàn tất và checkpoint bị thay sau freeze. Lập plan
  không mở annotation test. Có marker một lần trước final test access.

## 3. Kết quả xác minh thực tế

Môi trường: Python 3.13.14, torch 2.11.0,
torchvision 0.26.0, pycocotools
2.0.11. Máy chỉ có CPU; CUDA available =
False. Dùng 4 torch threads.

| Kiểm tra | Kết quả |
| --- | --- |
| Unit tests của dataset và pipeline | 14/14 đạt |
| Geometry/normalization/class mapping/lật bbox | Đạt |
| COCO prediction hoàn hảo | AP50:95 = 1.0 trên fixture |
| COCO prediction rỗng | AP50:95 = 0.0 trên fixture có GT |
| Lớp không có GT / không đủ support | null AP / reliable metric không định nghĩa |
| Chạy source smoke liên tục | 2 updates, 3 ảnh train |
| Chạy dừng rồi resume | 2 updates, 3 ảnh train |
| So sánh model/optimizer/scaler/RNG/position/LR state | Tất cả khớp chính xác |
| Số tensor mô hình được so sánh | 295 |
| Sai khác tuyệt đối lớn nhất ở tensor mô hình | 0.0 |
| Reload checkpoint qua evaluation CLI | 2 ảnh val, 55 bbox, 200 predictions |
| Fine-tune target weather từ source checkpoint | 1 update, 2 ảnh target train |
| Pilot COCO-pretrained qua pilot CLI | 1 update, 2 ảnh train |
| CLI từ chối freeze smoke checkpoint và resume đổi LR | 2/2 đạt |
| Annotation test trong các lượt xác minh | Không truy cập |

Source smoke random initialization: loss hai updates lần lượt
12.5672 và 3.8292. Đây là kiểm tra loss hữu hạn và đường backward,
không phải bằng chứng hội tụ hay cải thiện độ chính xác.

Lượt liên tục xử lý 3 image-passes trong
18.00 giây cho data/forward/backward/update,
tương đương 0.167 ảnh/giây. Wall time
30.00 giây còn bao gồm val/checkpoint; setup
model/dataset trước vòng train không nằm trong wall time này.

Pilot COCO-pretrained: 15.89 giây training,
0.126 ảnh/giây; inference validation
3.03 giây. Số mẫu quá nhỏ và có warm-up/cache
effects; không dùng các con số này để dự báo GPU-hours của toàn campaign.

Source smoke random có AP50:95 val = 0.0000 trên
2 ảnh; pretrained pilot có AP50:95 val =
0.0000 trên 1 ảnh.
Các điểm này chỉ xác nhận evaluator chạy với model thật. Model head mới và một
đến hai updates không tạo baseline nghiên cứu.

## 4. Artifact xác minh

Tại `runs/pipeline_verified/`:

- `uninterrupted/`: config, losses, last/best checkpoint, val metrics/predictions.
- `resumed/`: lượt dừng và resume cùng cấu hình.
- `resume_equivalence.json`: kết quả so sánh chính xác các state.
- `unit_tests.json`, `unit_tests.log`, `guard_checks.json`: bằng chứng tests và CLI guards.
- `validation_reload/`: predictions/metric khi reload checkpoint bằng CLI riêng.
- `target_finetune/`: xác minh nhánh target initialization từ source.
- `pretrained_pilot/`: xác minh khởi tạo bằng COCO pretrained và pilot entry point.

Checkpoints/runs/dataset đều nằm ngoài Git. Không thay đổi các file đã được user
stage trong Git index; các thay đổi triển khai chỉ ở working tree.

## 5. Giới hạn và bước kế tiếp

CUDA, AMP và DataLoader nhiều workers chưa được xác minh trên máy này. Resume
khớp chính xác được xác minh trên CPU với workers=0; không suy rộng sang mọi GPU.
Chưa chạy source training đầy đủ, full target grid, adaptation method, bootstrap
interval hoặc recovery thresholds. Final test evaluator được kiểm tra bằng
fixture và guard logic, chưa chạy với held-out test thật.

S=2000, E=30, LR=0.005, batch hiệu dụng=2, threshold reliable=100 cùng optimizer/
augmentation mặc định là cấu hình triển khai tạm thời. Cần pilot GPU đại diện và
chốt chính sách trước training chính thức. Không thay đổi split để tối ưu metric.

Hướng dẫn lệnh đầy đủ: `docs/training_pipeline.md`. Một lượt pilot trên CPU:

```powershell
python scripts/pilot_frcnn.py --output-dir runs/new_pilot --device cpu --steps 5 --images 10 --validation-images 8
```

Sau khi có GPU và chốt cấu hình, bước nghiên cứu kế tiếp là train source trên
9.804 ảnh daytime-clear, chọn checkpoint bằng 1.764 ảnh val, rồi chạy các budget
target và ceiling theo proposal.

API detector tham chiếu [Torchvision Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html);
metric tham chiếu [COCOeval chính thức](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
