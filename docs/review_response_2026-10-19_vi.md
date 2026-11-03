# Phản hồi góp ý review — sửa đổi protocol 2026-10-19

Tài liệu này trả lời 6 góp ý theo thứ tự. Với mỗi góp ý: đã sửa gì, số liệu thực tế
và phần nào còn mở. Mọi số liệu bên dưới đều lấy từ artifact trong `runs/` (không
commit vào Git). Không có lượt chạy nào đọc nhãn held-out.

## 1. Zero-shot gap của weather phải đo trước tiên

**Đã sửa.** Thêm `scripts/zero_shot_gap.py` làm bước đầu tiên của pilot. Script so
sánh source model trên **source val** (1.764 ảnh) với **train pool** của từng target.
Source model chưa từng thấy các ảnh này, và đây không phải nhãn held-out. Bootstrap
theo ảnh cho ra CI 95% của gap = mAP_source − mAP(0). Nếu cận dưới của CI không vượt
`--min-gap` (mặc định 0,01 AP), shift đó bị gắn cờ trước khi chạy ladder. Ngoài ra,
`recovery()` trả về *undefined* khi mẫu số mAP_c − mAP(0) không vượt 0,01. Như vậy
Recovery không bao giờ được tính trên một mẫu số gần 0.

Chưa có source model (máy chỉ có CPU), nên mình chạy trước một **probe proxy**: dùng
Faster R-CNN R50-FPN pretrained trên COCO, chưa từng train trên BDD. Probe đánh giá
150 ảnh/domain trên 3 lớp mà COCO và roster reliable cùng có (pedestrian, car,
traffic light), với 1.000 lần bootstrap.

| Domain | AP50:95 (3 lớp) | CI 95% | Gap so với source | CI 95% của gap | Gap tương đối |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source (day-clear, val) | 0,267 | 0,250–0,291 | — | — | — |
| Weather (day-rainy) | 0,246 | 0,226–0,270 | **0,021** | **−0,009 – 0,053** | 8% |
| Lighting (night-clear) | 0,148 | 0,125–0,175 | 0,119 | 0,089–0,152 | 45% |
| Compound (night-rainy) | 0,131 | 0,107–0,157 | 0,136 | 0,106–0,170 | 51% |

**Kết luận:** probe xác nhận lo ngại của reviewer. Với detector tổng quát,
daytime-rainy gần như không khó hơn daytime-clear: CI của gap chứa 0, và gate gắn cờ
*"no clear gap: Recovery denominator at risk"*. Gap của lighting và compound rõ ràng.
Ở mục 6, các proxy dữ liệu (độ sáng, occluded/truncated) cũng không thấy weather khó
hơn.

Giới hạn: đây là proxy COCO, không phải source model của proposal. Gate chính thức
vẫn phải chạy lại bằng `--checkpoint <source best.pt>` ngay sau khi train source.
Nếu gate chính thức cũng gắn cờ weather, có ba lựa chọn đã ghi trong TODO:
(a) chỉ báo cáo weather mô tả (raw change, không có ngưỡng N75/N90);
(b) thay bằng daytime-snowy, vốn đã được proposal giữ làm bản lặp của weather arm;
(c) giữ weather làm *negative control* cho RQ4.

Artifact: `runs/zero_shot_gap_coco_proxy/summary.json` (gồm prediction từng ảnh và
danh sách ảnh).

## 2. Test không đạt 600 ảnh/domain → gộp val vào test cho weather và compound

**Đã sửa đúng theo đề xuất.** Thư mục vật lý giữ nguyên (55.000 / 14.863 / 10.000),
pool train không đổi; thay đổi nằm ở *vai trò* của ảnh:

| Domain | Train pool | Final test | Val dùng để chọn |
| --- | ---: | ---: | ---: |
| Source (day-clear) | 9.804 | 2.650 | 1.764 |
| Lighting (night-clear) | 18.015 | 4.869 | 3.274 |
| Weather (day-rainy) | 1.985 | **933** (537 + 396) | 0 |
| Compound (night-rainy) | 1.738 | **756** (470 + 286) | 0 |

Hiện mọi domain đều có ≥ 600 ảnh test.

- `bdd_pipeline/data.py`: `final_test_rows()` gộp val của weather/compound vào test,
  còn `selection_rows()` **từ chối** val của hai domain này. `read_manifests()` kiểm
  tra toàn bộ số đếm trên ở mỗi lần đọc; đã chạy với dữ liệu thật và đạt.
- Vì weather/compound không còn val, **mọi run target** (fine-tune, ceiling, control)
  dùng checkpoint ở update cuối của schedule cố định. Như vậy mọi shift dùng cùng một
  quy tắc chọn checkpoint, giữ được khả năng so sánh cho RQ4. Chỉ source training
  chọn `best.pt` trên source val. Lighting val chỉ còn dùng để pilot S/E/LR.
- Final plan lên schema 2, ghi roster reliable, các domain holdout-val và cỡ test
  từng domain. `evaluate_frcnn.py` từ chối validation trên weather/compound.
- EDA v2 không còn đọc nhãn val của weather/compound (682 record được bỏ qua và có
  assert kiểm tra).

**Công khai:** trước sửa đổi này, EDA cũ đã đọc nhãn val weather/compound để mô tả.
Một CPU smoke run cũng đã đánh giá trên 1 ảnh val weather (model ngẫu nhiên, 1
update). Không có quyết định mô hình hay lựa chọn nào dựa vào các nhãn đó.

## 3. Reliable class dùng chung cho mọi domain

**Đã sửa.** Roster cố định: `RELIABLE_CLASSES = (pedestrian, car, traffic light,
traffic sign)`. Ngưỡng support 100 cũ đã bỏ. Primary metric là mAP50:95 trên 4 lớp
này. Nếu tập đánh giá thiếu một lớp, metric là `null` và lớp bị thiếu được liệt kê;
roster không bao giờ âm thầm co lại. **Mean 10 lớp được giữ làm sensitivity
analysis.** Số bbox train cho thấy cả 4 lớp đều đủ support ở mọi domain:

| Lớp | Source | Lighting | Weather | Compound |
| --- | ---: | ---: | ---: | ---: |
| car | 109.545 | 165.378 | 19.707 | 15.136 |
| traffic sign | 34.675 | 58.543 | 6.482 | 5.159 |
| traffic light | 20.606 | 53.397 | 5.957 | 5.812 |
| pedestrian | 13.170 | 12.265 | 3.249 | 980 |

## 4. Số liệu smoke test và lý do chọn Faster R-CNN

Cần nói rõ trước: **smoke test không chứng minh được lựa chọn Faster R-CNN, và cũng
không được thiết kế để làm việc đó.** Smoke test chỉ kiểm tra phần mềm: loss hữu hạn,
resume chính xác, evaluator chạy được. Lý do chọn Faster R-CNN nằm ở RQ5 (proposal
6.1). Gần như mọi implementation domain adaptation công khai đều xây trên Faster
R-CNN hoặc FCOS; các nhánh YOLO (MS-DAYOLO, SSDA-YOLO, SF-YOLO) chỉ có cho
YOLOv4/v5. Chọn detector không có implementation DA thì không so sánh được trong
cùng kiến trúc.

Số liệu smoke thực tế (CPU, không có GPU):

| Lượt chạy | Ảnh / update | Kết quả |
| --- | --- | --- |
| Source, random init (chạy liên tục) | 3 ảnh / 2 update | loss 12,567 → 3,829; 0,167 ảnh/s train |
| Source, dừng rồi resume | 3 ảnh / 2 update | 295 tensor khớp tuyệt đối, sai khác lớn nhất 0,0 |
| Pilot COCO-pretrained | 2 ảnh / 1 update | 0,126 ảnh/s train; val AP 0,0 (head mới, 1 update) |
| Fine-tune weather (v1) | 2 ảnh / 1 update | 0,131 ảnh/s |
| **Mới (v2):** fine-tune weather không dùng val | 2 ảnh / 1 update | loss 2,433; chọn `last.pt`; không đọc nhãn test |
| **Mới (v2):** pilot source có val | 2 ảnh / 1 update | loss 13,252; primary = `null` vì 2 ảnh val thiếu pedestrian (đúng thiết kế) |
| Suy luận Faster R-CNN COCO, 1280×720 (probe mục 1) | 600 ảnh | 4,2–4,7 s/ảnh, CPU 12 thread |

## 5. Mirror Kaggle: xác nhận nguồn và phiên bản nhãn

**Nguồn và phiên bản đã được ghi lại:**

- Mirror: Kaggle `solesensei/solesensei_bdd100k`, version 2, cập nhật 2019-05-21.
- Nhãn: **bản BDD100K Labels phát hành 2018** (`bdd100k_labels_release`), schema
  2018. Bản này dùng các tên lớp `person`/`bike`/`motor` và có kèm polygon
  lane/drivable area.
  - Train: 69.863 record, SHA256 `c1998792…a9e`.
  - Val: 10.000 record, SHA256 `c835be65…4af`.
- **Phát hiện quan trọng:** tài liệu chính thức (`github.com/bdd100k/bdd100k`,
  `doc/source/download.rst`) ghi bộ detection 2018 là *deprecated*. Bộ được khuyến
  nghị là **Detection 2020 Labels (`det_20`)**, cùng tập ảnh nhưng đã qua kiểm tra
  chất lượng bổ sung. MD5 chính thức: bản 2018 là `e21be3e7d6a07ee439faf61e769667e4`,
  bản det_20 là `b86a3e1b7edbcad421b7dad2b3987c94`.

**Chưa đối chiếu được với bản chính thức.** Ngày 2026-10-19, `dl.cv.ethz.ch` không
phân giải được tên miền (NXDOMAIN). `bdd-data.berkeley.edu` trả về chứng chỉ TLS sai
tên miền và HTTP 403. Wayback Machine không lưu file nhãn. Phần đã chuẩn bị:

- `scripts/label_provenance.py fingerprint` đã chạy và cố định 200 + 200 record mẫu
  (seed 20261019), gồm hash từng record, thuộc tính và bbox. Artifact:
  `runs/label_provenance/label_fingerprint.json`.
- Khi có file chính thức (đăng nhập portal BDD100K), chạy:
  `python scripts/label_provenance.py compare --official <bdd100k_labels_release.zip>`.
  Script kiểm tra MD5 theo bảng chính thức, rồi so sánh từng record mẫu: có trùng
  byte không, thuộc tính và bbox có giống không.

Cần bạn quyết định: giữ nhãn 2018 (đúng với dataset đã materialize) hay chuyển sang
`det_20` trước khi train. Nếu chuyển, membership ảnh có thể giữ nguyên, nhưng phải
tạo lại annotation, EDA và hash phát triển.

## 6. Bảng occluded/truncated và traffic light/ảnh theo domain

**Đã thêm** vào EDA v2: `runs/eda_revised_v2/tables/domain_occlusion_truncation.csv`,
`domain_objects_per_image.csv` và hình `figures/09_difficulty_class_prior.png`. Cả
hai bảng tính trên bbox hợp lệ của train mới.

**Occluded/truncated (4 lớp reliable gộp lại):**

| Domain | Bbox | Occluded | Truncated | Occluded so với source |
| --- | ---: | ---: | ---: | ---: |
| Source | 177.996 | 50,8% | 6,7% | 1,00 |
| Lighting | 289.583 | 40,3% | 6,1% | 0,79 |
| Weather | 35.395 | 45,4% | 7,7% | 0,89 |
| Compound | 27.087 | 38,3% | 6,4% | 0,75 |

Ở weather, tỷ lệ truncated của car cao hơn (11,2% so với 8,7%), còn lại gần giống
source. Điều này khớp với mục 1. Ban đêm lại có tỷ lệ occluded *thấp hơn*: traffic
light 1,3% so với 5,2%. Nhiều khả năng người gán nhãn khó thấy vật bị che trong bóng
tối, chứ không phải ban đêm "dễ" hơn. Vì vậy thuộc tính này cũng là proxy có nhiễu
theo cách gán nhãn, cần diễn giải thận trọng.

**Số đối tượng trên ảnh (train):**

| Domain | Traffic light/ảnh | So với source | % ảnh có traffic light | Pedestrian/ảnh | Car/ảnh |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 2,10 | 1,00 | 44% | 1,34 | 11,17 |
| Lighting | 2,96 | 1,41 | 65% | 0,68 | 9,18 |
| Weather | 3,00 | 1,43 | 59% | 1,64 | 9,93 |
| Compound | 3,34 | 1,59 | 73% | 0,56 | 8,71 |

Bảng này cho thấy rõ **dịch chuyển phân phối lớp**. Traffic light/ảnh tăng 41–59%;
pedestrian/ảnh ban đêm chỉ còn khoảng một nửa. Vì vậy:

- **Recovery theo từng lớp** đã được implement: `bdd_pipeline/recovery.py` và
  `scripts/recovery_curves.py`. Script vẽ đường reliable mean và 4 đường per-class,
  có seed mean ± SD, ngưỡng N_BE/N75/N90/N95 cho từng lớp, cùng chặn mẫu số tối thiểu.
- Cách đọc: một lớp có prior thay đổi mạnh (traffic light) mà recovery rất nhanh ở N
  nhỏ gợi ý hiệu ứng học lại class prior. Một lớp có prior gần như không đổi (traffic
  sign, car) mà recovery chậm cho thấy cần học đặc trưng mới của đêm/mưa.

## Kiểm tra đã chạy

- Unit tests: **18/18 đạt**. Có 4 test mới: precision bootstrap trùng khớp với
  `COCOeval.accumulate`; vai trò held-out được gộp và selection bị chặn; dataset đọc
  được ảnh từ nhiều split vật lý; công thức recovery, chặn mẫu số, ngưỡng sustained
  và recovery theo lớp.
- Syntax check toàn bộ script. Dry-run trên dữ liệu thật: source có 9.804 train /
  1.764 val, chọn `best.pt`; weather 1.985, compound 1.738 và lighting 18.015 không
  dùng val, chọn update cuối.
- `evaluate_frcnn.py --domain weather` bị từ chối đúng như thiết kế.
- Smoke CPU v2 đạt (`runs/pipeline_verified_v2/`).

## Còn mở

1. Gate zero-shot chính thức bằng source model: cần GPU để train source.
2. Đối chiếu nhãn với bản chính thức và quyết định giữ nhãn 2018 hay chuyển `det_20`.
3. `docs/proposal_dataset_aligned.docx` đã được cập nhật theo sửa đổi ở mục 5, 6.3, 7
   (gồm 7.2, 7.4), 7.3 và 11, thêm 2 dòng rủi ro (shift không có zero-shot gap;
   provenance nhãn). Các mục khác (ví dụ ước tính compute ở mục 10) không đổi.
4. Notebook EDA chưa có section mới; báo cáo HTML v2 là
   `runs/eda_revised_v2/report_vi.html`.
