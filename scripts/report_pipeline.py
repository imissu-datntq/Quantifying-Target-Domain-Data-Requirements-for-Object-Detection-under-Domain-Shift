"""Write a Vietnamese report from actual pipeline verification artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-dir", type=Path, default=Path("runs/pipeline_verified"))
    parser.add_argument("--output", type=Path, default=Path("docs/pipeline_report_vi.md"))
    args = parser.parse_args()
    root = args.verification_dir

    def read(name: str) -> dict | list:
        return json.loads((root / name).read_text(encoding="utf-8"))

    tests = read("unit_tests.json")
    guards = read("guard_checks.json")
    resume = read("resume_equivalence.json")
    continuous = read("uninterrupted/result.json")
    resumed = read("resumed/result.json")
    environment = read("uninterrupted/environment.json")
    validation = read("validation_reload/metrics.json")
    target = read("target_finetune/result.json")
    pretrained = read("pretrained_pilot/result.json")
    pretrained_val = read("pretrained_pilot/validation_step_1.json")
    if not tests["passed"] or not all(check["exit_code"] != 0 and check["expected_error_found"] for check in guards):
        raise ValueError("Tests or CLI boundary checks did not pass")
    if not resume["passed"] or not all(result["completed"] for result in (continuous, resumed, target, pretrained)):
        raise ValueError("Verification is not complete")
    if any(result["test_labels_accessed"] for result in (continuous, resumed, target, pretrained)):
        raise ValueError("Smoke artifacts must not access held-out labels")
    losses = [json.loads(line)["loss"] for line in (root / "uninterrupted/training.jsonl").read_text(encoding="utf-8").splitlines()]
    report = f"""# Báo cáo xây pipeline Faster R-CNN

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

Môi trường: Python {environment['python']}, torch {environment['packages']['torch']},
torchvision {environment['packages']['torchvision']}, pycocotools
{environment['packages']['pycocotools']}. Máy chỉ có CPU; CUDA available =
{environment['cuda_available']}. Dùng {environment['torch_threads']} torch threads.

| Kiểm tra | Kết quả |
| --- | --- |
| Unit tests của dataset và pipeline | {tests['tests_run']}/{tests['tests_run']} đạt |
| Geometry/normalization/class mapping/lật bbox | Đạt |
| COCO prediction hoàn hảo | AP50:95 = 1.0 trên fixture |
| COCO prediction rỗng | AP50:95 = 0.0 trên fixture có GT |
| Lớp không có GT / không đủ support | null AP / reliable metric không định nghĩa |
| Chạy source smoke liên tục | {continuous['global_step']} updates, {continuous['processed_images']} ảnh train |
| Chạy dừng rồi resume | {resumed['global_step']} updates, {resumed['processed_images']} ảnh train |
| So sánh model/optimizer/scaler/RNG/position/LR state | Tất cả khớp chính xác |
| Số tensor mô hình được so sánh | {resume['model_tensor_count']} |
| Sai khác tuyệt đối lớn nhất ở tensor mô hình | {resume['max_absolute_model_tensor_difference']} |
| Reload checkpoint qua evaluation CLI | {validation['images']} ảnh val, {validation['instances']} bbox, {validation['predictions']} predictions |
| Fine-tune target weather từ source checkpoint | {target['global_step']} update, {target['processed_images']} ảnh target train |
| Pilot COCO-pretrained qua pilot CLI | {pretrained['global_step']} update, {pretrained['processed_images']} ảnh train |
| CLI từ chối freeze smoke checkpoint và resume đổi LR | {len(guards)}/{len(guards)} đạt |
| Annotation test trong các lượt xác minh | Không truy cập |

Source smoke random initialization: loss hai updates lần lượt
{losses[0]:.4f} và {losses[1]:.4f}. Đây là kiểm tra loss hữu hạn và đường backward,
không phải bằng chứng hội tụ hay cải thiện độ chính xác.

Lượt liên tục xử lý {continuous['processed_images']} image-passes trong
{continuous['training_seconds']:.2f} giây cho data/forward/backward/update,
tương đương {continuous['training_images_per_second']:.3f} ảnh/giây. Wall time
{continuous['invocation_wall_seconds']:.2f} giây còn bao gồm val/checkpoint; setup
model/dataset trước vòng train không nằm trong wall time này.

Pilot COCO-pretrained: {pretrained['training_seconds']:.2f} giây training,
{pretrained['training_images_per_second']:.3f} ảnh/giây; inference validation
{pretrained_val['inference_seconds']:.2f} giây. Số mẫu quá nhỏ và có warm-up/cache
effects; không dùng các con số này để dự báo GPU-hours của toàn campaign.

Source smoke random có AP50:95 val = {validation['AP50_95']:.4f} trên
{validation['images']} ảnh; pretrained pilot có AP50:95 val =
{pretrained_val['AP50_95']:.4f} trên {pretrained_val['images']} ảnh.
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
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
