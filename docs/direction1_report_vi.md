# Hướng 1 trên nhánh `alt-1`: quá trình, logic và kết quả

Tài liệu này mô tả những gì đã làm trên nhánh `alt-1` cho hướng 1 trong góp ý tiếp theo của giảng viên (2026-11-24): các bước làm theo thứ tự nào, mỗi phân tích tính gì và vì sao tính như vậy, và kết quả ra sao.

Nhánh này không huấn luyện thêm model nào và không chấm điểm lại run nào. Cả ba phân tích chỉ đọc các kết quả đã được chấm một lần, cùng annotation của tập train.

Tài liệu liên quan:

- Ghi chú kết quả ngắn hơn, đủ mọi bảng: `docs/results_direction1_vi.md`.
- Luật đọc gốc (tiếng Anh), viết trước khi chạy: `.agent/plans/active/2026-11-24-direction-1-analyses.md`.
- Quyết định và giới hạn được phép phát biểu: mục 2026-11-25 trong `.agent/DECISIONS.md`.
- Bảng và hình: `docs/results/direction1/`.

Quy ước số: dấu phẩy thập phân, dấu chấm phân cách hàng nghìn, khoảng trong ngoặc vuông là khoảng tin cậy 95%. "AP" và "mAP" luôn là AP50:95.

## Mục lục

1. [Câu hỏi](#1-câu-hỏi)
2. [Quá trình](#2-quá-trình)
3. [Dữ liệu và quy ước chung](#3-dữ-liệu-và-quy-ước-chung)
4. [Phân tích 1: mức tăng AP theo instance hay theo ảnh](#4-phân-tích-1-mức-tăng-ap-theo-instance-hay-theo-ảnh)
5. [Phân tích 2: độ phân tán giữa các seed theo N](#5-phân-tích-2-độ-phân-tán-giữa-các-seed-theo-n)
6. [Phân tích 3: source retention như một trục thứ hai](#6-phân-tích-3-source-retention-như-một-trục-thứ-hai)
7. [Kiểm thử và kiểm chứng](#7-kiểm-thử-và-kiểm-chứng)
8. [Giới hạn chung](#8-giới-hạn-chung)
9. [File và cách chạy lại](#9-file-và-cách-chạy-lại)

## 1. Câu hỏi

Hướng 1 gồm ba ý, mỗi ý thành một script:

1. **Mật độ pedestrian (`scripts/analyze_instances.py`).** Pool source có 13.170 pedestrian trên 9.804 ảnh, tức 1,34 mỗi ảnh. Pool lighting (đêm, trời quang) có 12.265 pedestrian trên 18.015 ảnh, tức 0,68 mỗi ảnh. Pedestrian ban đêm recover chậm nhất (0,46 ở N = 1.000). Điều đó có thể một phần do ít nhãn pedestrian hơn, chứ không chỉ do khó về hình ảnh.
   - Câu hỏi: ΔAP của một class giữa hai rung liền nhau đi theo Δ(số instance của class đó) hay theo Δ(số ảnh)?
   - Nếu đi theo instance, ngưỡng tính bằng số ảnh chỉ là artefact của mật độ nhãn trong BDD100K, và đại lượng chuyển được sang dataset khác là số instance mỗi class.
2. **Độ phân tán theo N (`scripts/analyze_seed_spread.py`).** Vẽ độ lệch chuẩn (SD) giữa các seed lấy mẫu theo N.
   - SD giảm như 1/√N nghĩa là chọn ảnh ngẫu nhiên đã gần tối ưu.
   - SD chững lại (plateau) nghĩa là việc chọn ảnh nào vẫn còn quan trọng, tức còn chỗ cho active selection.
   - Đây là câu trả lời bằng số cho câu hỏi "vì sao không dùng active learning".
3. **Source retention (`scripts/analyze_retention.py`).** Với mỗi cặp (phương pháp, N), đặt (recovery trên target, retention trên source) thành một điểm, các điểm tạo thành một mặt đánh đổi.
   - Dự đoán của giảng viên: ALDI++/replay thắng hẳn ở N nhỏ.
   - Ở N lớn, FT thắng trên target nhưng thua trên source.

## 2. Quá trình

### 2.1 Các bước

| Bước | Ngày | Kết quả |
|---|---|---|
| Tạo nhánh `alt-1` từ `team/main` (`ef354cf`) | 2026-11-24 | |
| Viết plan và luật đọc cho cả ba phân tích, trước khi có dòng code nào chạy trên dữ liệu thật | 2026-11-24 | `.agent/plans/active/2026-11-24-direction-1-analyses.md` |
| Viết ba script và 86 unit test trên dữ liệu tổng hợp. Mỗi script có một người viết và hai người review độc lập; từng phát hiện được kiểm lại trước khi sửa | 2026-11-24 đến 25 | Plan được bổ sung các quyết định sau review (mục 2.3) |
| Kiểm chứng lần hai, chỉ đọc: mỗi script một người phản biện, cộng một người đối chiếu plan với code | 2026-11-25 | Không thấy lỗi logic ở phân tích instance và retention. Có vài chỗ diễn đạt chưa đủ và thiếu một test cho `--seed`; đã sửa hết |
| Commit code, rebase lên `team/main` `407d60d` (commit này đưa ba thư mục `scored/` vào git) | 2026-11-25 | `3465e7c` |
| Chạy cả ba script một lần, chỉ CPU, `--seed 0`, sau khi được duyệt | 2026-11-25 | `runs/local/direction1/` |
| Chép output sang `docs/results/direction1/` (byte-identical), viết note kết quả, kiểm chứng từng con số của note với CSV, sửa các chỗ diễn giải quá tay | 2026-11-25 | `e7615d2` |
| Push lên `team/alt-1`, cập nhật trạng thái trong TODO và HANDOFF | 2026-11-25 | `cd70eb0` |

### 2.2 Nguyên tắc đã giữ

- **Luật trước, chạy sau.** Cách đọc từng kết quả được viết và commit cùng code trước khi chạy trên dữ liệu thật. Luật gồm:
  - khi nào một hệ số được coi là "trên 0";
  - khi nào một độ dốc "phủ −1/2";
  - cờ "floor rõ ràng cao hơn nhiễu";
  - định nghĩa "trội hơn" và "đánh đổi".

  Tuy vậy, luật được viết sau khi đã biết kết quả từng rung của RQ1–RQ5. Vì vậy **mọi kết luận ở đây là mô tả, không phải kiểm định xác nhận**.
- **Code trước, duyệt rồi mới chạy.** Ban đầu chỉ viết code ("tạm thời chỉ viết code trước, lúc nào chạy thì báo để mình duyệt"). Việc chạy diễn ra sau khi được duyệt ("Chạy luôn trên máy này").
- **Không đụng vào pipeline.**
  - `scripts/bdd_pipeline/` và `scripts/bdd_adapt/` không bị sửa, vì mỗi run lưu SHA256 của các file đó.
  - Code mới chỉ nằm trong ba file `scripts/analyze_*.py` và ba file test tương ứng.
  - Các script có sẵn (`analyze_campaign.py`, `analyze_rq5.py`, `prepare_revised_bdd100k.py`) được import, không bị sửa.
- **Chỉ dùng kết quả đã train lại trên máy local**: `runs/local/{campaign,rq5,revision}/scored/`.
- **Không mở nhãn test hay val.** Chỉ file annotation train được đọc; script từ chối mọi file annotation khác.

### 2.3 Các quyết định thêm vào sau review code

Review độc lập dẫn tới các quyết định sau. Tất cả đã được ghi vào plan trước khi chạy:

- **Độ dốc chính của phân tích 2 là độ dốc đã hiệu chỉnh pool hữu hạn (`power_fpc`).** Lý do ở mục 5.3.
- **Mô hình phương sai có floor chỉ fit theo từng shift**, không gộp. Khoảng tin cậy của nó lấy từ bootstrap tham số, kèm cờ `floor_above_training_noise`.
- **Khoảng tin cậy của độ dốc được gọi là xấp xỉ**, vì các rung của một seed là subset lồng nhau.
- **Phân tích 1 có thêm cách đọc khi hệ số `within` âm.**
- **Độ tương quan giữa hai biến giải thích được tính sau khi loại phần intercept** của từng đặc tả. Ví dụ với `both+class`, tương quan được tính trong từng class.
- **Phân tích 3 có thêm tỉ lệ replicate không xác định kỳ vọng chính xác** cho mốc 12 epoch.
- **Mẫu số tối thiểu của recovery là ngưỡng chặt ở mọi nơi**: mẫu số phải lớn hơn 0,01, bằng 0,01 là không xác định, giống `bdd_pipeline.recovery`.

### 2.4 Các chỗ sửa trong note sau khi kiểm chứng

Vòng kiểm chứng note tìm ra một số chỗ diễn giải quá tay hoặc thiếu lưu ý. Tất cả đã được sửa:

- **Pedestrian ở compound N = 1.000 không xác định** khi so với ceiling matched-pool: 120/1.000 replicate có mẫu số không vượt quá 0,01.
- **Ghi rõ nhãn E = 3 và E = 12** cho mọi recovery. Recovery của phân tích 3 tính theo "mốc FT toàn pool 3 epoch", không gọi là ceiling.
- **Phần theo từng class chỉ được mô tả**, không kết luận class nào "đi theo" biến nào.
- **Vấn đề nhận diện của phân tích 1 là gây nhiễu, không phải đa cộng tuyến**: tương quan 0,44, VIF 1,2 (mục 4.4).
- **Độ dốc gộp của phân tích 2 không được đọc riêng.**
- **Ghi rõ:**
  - ba hàng có c ≤ 0;
  - giả định nhiễu huấn luyện không phụ thuộc N;
  - khoảng của SD lighting ở N = 10;
  - χ²/df 3,81 của traffic light ở lighting.
- **Kết quả retention được ghi là "khớp với dự đoán (mô tả)".**
- **FT 12 epoch chỉ cao nhất trên target ở lighting.**
- **Pareto front dựa trên ước lượng điểm, và replay ở N = 1.000 chỉ có seed 0.**

## 3. Dữ liệu và quy ước chung

### 3.1 Đầu vào

- **354 bộ kết quả đã chấm**, nằm trong `runs/local/campaign/scored/`, `runs/local/rq5/scored/` và `runs/local/revision/scored/`.
  - Mỗi bộ là một thư mục `<run>/<domain chấm>/` gồm `metrics.json` (AP từng class, thông tin run) và `bootstrap_class_ap.npy` (ma trận 1.000 × 10: AP của 10 class trên 1.000 lần lấy lại ảnh test).
  - Mọi bộ dùng chung bootstrap seed 20261030 và 1.000 lần lấy lại, trên cùng số ảnh test: compound 756, weather 933, lighting 4.869, source 2.650.
  - Script kiểm tra điều này bằng `analyze_rq5.shared_resamples` và dừng lại nếu không khớp. Nhờ đó lần lấy lại thứ t là cùng một tập ảnh test cho mọi run.
- **Annotation train** `data/bdd100k_revised/annotations/bdd100k_labels_images_train.json`: chỉ phân tích 1 đọc file này, và đọc dạng stream một lần.
- **Phân biệt các run bằng trường `run`** trong `metrics.json` (stage, domain, budget, seed, epochs, total_updates), không bao giờ bằng tên thư mục.

### 3.2 Ladder fine-tune

- **Các rung:** N = 10, 25, 50, 100, 250, 500, 1.000, và với lighting thêm 2.000, 5.000, 10.000. Mỗi rung có 3 seed lấy mẫu (0, 1, 2), và subset lồng nhau theo seed: subset N = 10 của seed 0 nằm trong subset N = 25 của seed 0.
- **Full pool:** E = 3, seed 0 và 1. Kích thước pool train: compound 1.738, weather 1.985, lighting 18.015.
- **N = 0:** source model chấm trên shift.
- **Run matched-pool** (1.738 ảnh của lighting và weather, dùng cho RQ4) không phải rung.
- **Lịch huấn luyện cố định:** max(500 update, 3 epoch), LR 1e-3, batch 2. Rung N = 10 vì vậy chạy 500 update, khoảng 100 epoch trên 10 ảnh.

### 3.3 Metric và recovery

- **Metric chính:** reliable mAP50:95, tức trung bình AP của pedestrian, car, traffic light, traffic sign. Phân tích theo class dùng AP của bốn class này. Trung bình 10 class không được dùng.
- **Recovery:** recovery = (mAP(N) − mAP(0)) / (mAP_ref − mAP(0)). Recovery không xác định khi mẫu số không lớn hơn 0,01.
- **Các mốc tham chiếu (reliable mAP):**

| Shift | mAP(0) (source model) | Mốc FT toàn pool 3 epoch | Mẫu số E = 3 | Ceiling FT toàn pool 12 epoch | Mẫu số E = 12 |
|---|---|---|---|---|---|
| compound | 0,2194 | 0,2572 | 0,0378 | 0,2578 | 0,0384 |
| weather | 0,3201 | 0,3355 | 0,0154 | 0,3310 | 0,0109 |
| lighting | 0,2438 | 0,2960 | 0,0523 | 0,2989 | 0,0551 |

Source model trên source test đạt 0,3422; đây là mẫu số của retention. Mẫu số E = 12 của weather (0,0109) chỉ vừa vượt 0,01, nên trong bootstrap khoảng một phần ba số replicate có mẫu số không vượt quá ngưỡng (mục 6.4).

### 3.4 Khoảng tin cậy

**Phân tích 1 và 3** dùng bootstrap phân tầng của dự án:

- Replicate t lấy lần lấy lại ảnh test thứ t của mọi run.
- Các seed lấy mẫu được rút lại có hoàn lại.
- Khoảng tin cậy là khoảng percentile 95% trên các replicate xác định, và số replicate không xác định được báo kèm.

**Phân tích 2 là ngoại lệ.** Khoảng của SD, của floor nhiễu và của độ dốc được tính giải tích (χ², phân phối chuẩn). Khoảng của mô hình phương sai lấy từ một bootstrap tham số có seed cố định.

### 3.5 Hàm dùng lại

Không viết lại logic đã có, để kết quả khớp với các phân tích trước:

- **`analyze_campaign`:**
  - `load` đọc các thư mục scored;
  - `require_one_schedule` và `require_ceiling_seeds` kiểm tra lịch và seed;
  - `read_revision` đọc các run revision;
  - `write_tables` ghi CSV.
- **`analyze_rq5`:**
  - `select` và `arm_runs` lấy run theo stage và domain;
  - `shared_resamples` kiểm tra lần lấy lại;
  - `split_revision` và `merge_replay_seeds` dùng cho run revision của ALDI++/replay;
  - `row_rng` sinh số ngẫu nhiên theo từng hàng.
- **`bdd_pipeline.data`:** `nested_rows` và `domain_rows` dựng lại subset của mỗi run; `sanitize_labels` làm sạch box như lúc train.
- **`bdd_pipeline.recovery.recovery`.**
- **`prepare_revised_bdd100k.stream_records`.**

## 4. Phân tích 1: mức tăng AP theo instance hay theo ảnh

### 4.1 Đếm instance

1. Stream file annotation train một lần. Với mỗi ảnh thuộc pool train của source và ba target, box đi qua `sanitize_labels` (đúng những box mà lúc train model nhìn thấy), rồi đếm số box của từng reliable class.
2. Ảnh không thuộc bốn pool bị bỏ qua. Ảnh thiếu annotation hoặc bị trùng làm script dừng.
3. Dựng lại tập ảnh của mỗi run bằng chính lệnh mà code train dùng: `nested_rows(domain_rows(manifests, "train", domain), budget, seed)`, với budget và seed lấy từ trường `run`. Vì vậy số instance của một run là số instance trong đúng các ảnh run đó đã học.
4. Output:
   - `instance_pool_density.csv`: số ảnh, số instance và số instance mỗi ảnh theo pool và class.
   - `instance_counts.csv`: số ảnh, số instance và số ảnh có class đó, theo shift, seed và rung.

### 4.2 Quan sát

- **Chuỗi rung:** với mỗi shift và seed lấy mẫu, các rung mà seed đó có tạo thành một chuỗi: N = 0, các rung ladder, rồi full pool (chỉ seed 0 và 1 có full pool).
- **Một quan sát** là một cặp rung liền nhau nhân với một class:
  - ΔAP = AP_c(rung sau) − AP_c(rung trước);
  - ΔN = số ảnh thêm vào, ΔI = số instance của class c thêm vào;
  - Δln N = ln(N_sau / N_trước), Δln I = ln(I_sau / I_trước).
- **Bước từ N = 0 lên rung đầu được ghi lại nhưng không đưa vào hồi quy**, vì hai lý do:
  - Δln I không xác định khi xuất phát từ 0;
  - rung N = 10 bị chi phối bởi negative transfer dưới sàn 500 update.
- **Số quan sát:**
  - Compound: seed 0 và 1 có 7 bước, seed 2 có 6, tổng 20. Weather cũng 20.
  - Lighting: 10 + 10 + 9 = 29.
  - Tổng 69 bước × 4 class = 276 quan sát.
  - Dạng log bỏ 1 quan sát có I_trước = 0, còn 275.
  - Số update của cả hai rung được giữ thành cột, vì sàn 500 update làm lịch thay đổi dọc ladder.

### 4.3 Các đặc tả hồi quy

Mỗi đặc tả được fit bằng OLS ở hai dạng. **Dạng log** dùng x_I = Δln I và x_N = Δln N. **Dạng thô** dùng x_I = ΔI và x_N = ΔN, đúng dạng giảng viên viết.

| Đặc tả | Mô hình | Hỏi gì |
|---|---|---|
| `images` | ΔAP ~ 1 + x_N | Chỉ theo ảnh |
| `instances` | ΔAP ~ 1 + x_I | Chỉ theo instance |
| `both` | ΔAP ~ 1 + x_I + x_N | Hồi quy giảng viên đề xuất |
| `both+class` | ΔAP ~ class + x_I + x_N | Như trên, có intercept riêng cho mỗi class |
| `within` | ΔAP ~ ô + x_I, với ô = shift × class × bước | Ở cùng một bước ảnh, seed nào rút được nhiều instance hơn thì có tăng nhiều AP hơn không? |
| `both` theo class | ΔAP ~ 1 + x_I + x_N, từng class riêng | Bản theo class |

- **Cách fit `within`:** trừ trung bình trong từng ô cho cả ΔAP và x_I, rồi hồi quy phần còn lại. Chỉ giữ các ô có ít nhất 2 quan sát. R² báo kèm là within-R².
- **Mỗi hàng báo:**
  - hệ số của x_I và x_N, kèm khoảng tin cậy và hệ số chuẩn hóa;
  - R²;
  - tương quan giữa x_I và x_N sau khi loại phần intercept, và VIF = 1/(1 − r²);
  - RMSE bỏ-một-shift: fit trên hai shift, dự đoán shift thứ ba (không tính cho `within`).
- **Fit không xác định** khi ma trận thiết kế suy biến hoặc biến phụ thuộc có giá trị không hữu hạn.

### 4.4 Vì sao `within` là phép đọc chính

- **Trong kỳ vọng hai biến gần trùng nhau.** Subset lồng nhau giữ mật độ của một class gần như không đổi dọc một chuỗi, nên Δln I ≈ Δln N.
- **Trên dữ liệu thật hai biến không trùng.** Chúng chỉ tương quan 0,44 (VIF 1,2):
  - Δln N gần như cố định theo bước;
  - Δln I dao động do nhiễu lấy mẫu ở các rung nhỏ, nơi có ít instance, nhất là pedestrian.
- **Phần lệch giữa hai biến vì vậy trùng với vị trí trên ladder.** Một đặc tả gộp mà thấy instance "giải thích tốt hơn" có thể chỉ phản ánh việc ΔAP ở rung nhỏ khác ΔAP ở rung lớn. Đây là vấn đề gây nhiễu, không phải đa cộng tuyến.
- **`within` loại được phần nhiễu đó.** Nó giữ cố định shift, class và bước ảnh, nên chỉ còn khác biệt giữa các seed ở cùng một bước. Phần này không lẫn với class, với số ảnh hay với vị trí trên ladder.
- **Ở dạng thô, hai biến tách nhau chủ yếu nhờ mật độ khác nhau giữa các class**, mà mật độ lại lẫn với độ khó của từng class. Vì vậy dạng thô không được dùng để đọc.

### 4.5 Khoảng tin cậy

1. Mỗi replicate t rút lại 3 seed có hoàn lại trong từng shift. Chuỗi của mỗi seed được lấy nguyên, kèm số instance của nó.
2. AP lấy từ lần lấy lại ảnh test thứ t.
3. Mảng seed rút được (`default_rng(--seed)`) dùng chung cho mọi hàng của bảng, nên các hàng được so trên cùng các replicate.
4. Có 1.000 replicate; khoảng tin cậy là khoảng percentile 95% trên các replicate xác định.

### 4.6 Luật đọc

Luật dựa trên hệ số x_I của đặc tả `within` ở dạng log:

- **Khoảng tin cậy trên 0:** "ở cùng bước ảnh, nhiều instance hơn cho class đó nhiều AP hơn".
- **Khoảng tin cậy phủ 0:** "không phát hiện tác động của instance ở cùng bước ảnh".
- **Khoảng tin cậy dưới 0:** "ở cùng bước ảnh, nhiều instance hơn đi kèm mức tăng AP thấp hơn", tức giả thuyết mật độ không được ủng hộ.

Các đặc tả gộp, VIF và RMSE bỏ-một-shift được báo bên cạnh, không kèm kết luận.

### 4.7 Recovery theo class và hình

- `instance_recovery.csv` và `instance_recovery.png` vẽ recovery từng class (trung bình theo seed, mốc là full pool E = 3) theo hai trục: hàng trên là số ảnh, hàng dưới là số instance của class đó. Mỗi shift là một đường.
- Nếu đường pedestrian ban đêm khớp với các shift khác trên trục instance nhưng không khớp trên trục ảnh, giải thích theo mật độ sẽ hiện ra trực quan.
- Đường nào có mẫu số không xác định thì bị bỏ khỏi hình và được ghi tên trong panel.

### 4.8 Kết quả

**Mật độ (số instance mỗi ảnh trong pool train):**

| Pool | pedestrian | car | traffic light | traffic sign |
|---|---|---|---|---|
| source (9.804 ảnh) | 1,34 | 11,17 | 2,10 | 3,54 |
| compound (1.738 ảnh) | 0,56 | 8,71 | 3,34 | 2,97 |
| weather (1.985 ảnh) | 1,64 | 9,93 | 3,00 | 3,27 |
| lighting (18.015 ảnh) | 0,68 | 9,18 | 2,96 | 3,25 |

Hai con số của giảng viên khớp chính xác, kể cả sau bước làm sạch nhãn: source có 13.170 pedestrian trên 9.804 ảnh, lighting có 12.265 trên 18.015 ảnh.

**Hồi quy, dạng log (275 quan sát):**

| Đặc tả | Hệ số Δln(instance) | Hệ số Δln(ảnh) | R² | VIF | RMSE bỏ-một-shift |
|---|---|---|---|---|---|
| images | – | +0,0242 [+0,0180; +0,0299] | 0,117 | – | 0,0075 |
| instances | +0,0116 [+0,0093; +0,0141] | – | 0,184 | – | 0,0073 |
| both | +0,0094 [+0,0049; +0,0128] | +0,0135 [+0,0072; +0,0207] | 0,214 | 1,2 | 0,0072 |
| both+class | +0,0099 [+0,0056; +0,0132] | +0,0129 [+0,0063; +0,0198] | 0,241 | 1,2 | 0,0071 |
| **within (phép đọc chính)** | **+0,0055 [−0,0017; +0,0143]** | – | 0,067 (within) | – | – |

Theo từng class (đặc tả `both`):

- Khoảng của hệ số Δln(instance) nằm trên 0 ở pedestrian (+0,0107) và traffic light (+0,0113).
- Khoảng của hệ số Δln(ảnh) nằm trên 0 ở car (+0,0335) và traffic sign (+0,0241).
- Car có mặt ở khoảng 99% số ảnh và traffic sign ở khoảng 80%. Với hai class này, Δln(instance) bám sát Δln(ảnh): tương quan 0,84 và 0,75, so với 0,19 ở pedestrian. Vì vậy hệ số tách riêng của hai class này kém tin cậy.

Dạng thô có within-R² 0,002, nên không dùng. Bảng đủ cả hai dạng là `instance_regression.csv`.

**Cách đọc:**

- **Theo luật:** khoảng của hệ số `within` phủ 0, nên kết luận là "không phát hiện tác động của instance ở cùng bước ảnh".
  - Cận trên +0,0143 có nghĩa: ở cùng một bước ảnh, gấp đôi số instance của một class (Δln = 0,69) đi kèm tối đa khoảng +0,010 AP cho class đó.
  - Phép đọc này gộp bốn class và chỉ dựa vào khác biệt giữa 3 seed, nên độ nhạy thấp. Nó không loại trừ được tác động của mật độ.
- **Các đặc tả gộp:** mô hình theo instance có R² và RMSE ngoài shift tốt hơn một chút so với mô hình theo ảnh (0,18 và 0,0073, so với 0,12 và 0,0075). Nhưng cả hai đều lẫn với vị trí trên ladder (mục 4.4), nên không được đọc thành kết luận.
- **Về con số 0,46:**
  - 0,46 là recovery của pedestrian ban đêm ở N = 1.000, tính so với ceiling full-pool 3 epoch, tức model train trên cả 18.015 ảnh lighting.
  - So với ceiling matched-pool 3 epoch (1.738 ảnh, chuẩn so sánh giữa các shift của RQ4), cùng run đó đạt 0,75 [0,56; 1,03], gần weather 0,85 [0,62; 1,07]. Ô của compound không xác định.
  - Vậy phần lớn khoảng cách về mức (0,46 so với 0,75) đến từ việc ceiling của lighting được train trên một pool lớn gấp khoảng 10 lần.
  - Dù vậy, so với ceiling matched-pool, pedestrian vẫn thấp nhất trong các class của lighting: car 0,87, traffic light 0,85, traffic sign 0,89, các khoảng tin cậy chồng nhau.
  - Thứ tự giữa các class phụ thuộc ceiling. Theo ceiling 12 epoch, pedestrian ban đêm là 0,54 (full pool) và 0,90 (matched pool), còn traffic light mới là class chậm nhất ở lighting (0,44 và 0,76).
- **Kết luận cho câu hỏi của giảng viên:** dữ liệu hiện có chưa đủ để nói ngưỡng tính theo ảnh là artefact mật độ của BDD100K, và cũng không đủ để loại trừ điều đó. Phần chênh còn lại giữa lighting 0,75 (0,68 pedestrian mỗi ảnh) và weather 0,85 (1,64 mỗi ảnh) đi đúng chiều giả thuyết mật độ, nhưng các khoảng tin cậy chồng nhau.

## 5. Phân tích 2: độ phân tán giữa các seed theo N

### 5.1 SD theo rung

- Với mỗi shift, mỗi key (reliable và bốn class) và mỗi rung ladder, tính SD mẫu của AP giữa các seed lấy mẫu (ddof 1). Ba seed cho ν = 2 bậc tự do.
- Khoảng 95% lấy từ phân phối χ² với ν bậc tự do: [√(ν s² / χ²₀,₉₇₅(ν)); √(ν s² / χ²₀,₀₂₅(ν))].
- Mỗi hàng có thêm hai cột:
  - SD theo đơn vị recovery: SD chia cho mẫu số E = 3, chỉ tính khi mẫu số lớn hơn 0,01;
  - hệ số pool hữu hạn fpc = (M − N)/(M − 1), với M là kích thước pool.

### 5.2 Floor nhiễu huấn luyện

- Hai run full pool E = 3 (seed 0 và 1) train trên cùng một tập ảnh. SD giữa chúng (ν = 1) vì vậy chỉ là nhiễu huấn luyện: thứ tự ảnh, phép lật ảnh, tính không tất định của GPU.
- Bản gộp ba shift lấy trung bình các phương sai, với ν = 3.
- Ở một rung ladder, seed đổi cả subset lẫn phần ngẫu nhiên trong huấn luyện. Vì vậy SD² ở rung ≈ phần do lấy mẫu ảnh cộng phần nhiễu huấn luyện.
- **Giả định nhiễu huấn luyện không phụ thuộc N chưa được kiểm tra**, và được nói kèm mọi kết quả.

### 5.3 Fit luật lũy thừa

ln s = a + b · ln N, fit bằng bình phương tối thiểu có trọng số trên các rung ladder (không gồm full pool). Có hai mức fit:

- **Theo từng shift:** một intercept, một độ dốc.
- **Gộp ba shift:** mỗi shift một intercept, chung một độ dốc.

**Hiệu chỉnh độ lệch và trọng số.**

- Nếu s² ~ σ² χ²_ν / ν, thì E[ln s] = ln σ + (ψ(ν/2) − ln(ν/2))/2 và Var(ln s) = ψ′(ν/2)/4, với ψ là hàm digamma và ψ′ là hàm trigamma.
- Biến phụ thuộc được trừ phần độ lệch đó, và trọng số là 4/ψ′(ν/2).
- Độ dốc có khoảng chuẩn từ (X′WX)⁻¹.
- χ²/df của phần dư chuẩn hóa xấp xỉ 1 khi mô hình đúng; giá trị lớn hơn nhiều là dấu hiệu fit kém.
- Có hai cờ: độ dốc phủ −1/2, và độ dốc dưới 0 (cận trên dưới 0).

**Hiệu chỉnh pool hữu hạn (`power_fpc`, phép đọc chính).**

- Các subset được rút không hoàn lại từ một pool hữu hạn. Với chọn ngẫu nhiên thuần, phương sai của trung bình tỉ lệ với fpc/N, nên s ∝ √(fpc/N), không phải 1/√N.
- Ở các rung chiếm phần lớn pool (ví dụ 1.000 trên 1.738 ảnh của compound), fpc nhỏ đi rõ rệt. Vì vậy ngay cả chọn ngẫu nhiên thuần cũng cho độ dốc chưa hiệu chỉnh khoảng −0,54 đến −0,58 trên các ladder này, chứ không phải −0,5.
- `power_fpc` trừ ½ ln(fpc) khỏi biến phụ thuộc, nên với chọn ngẫu nhiên thuần độ dốc trở về đúng −1/2. Bản `power` không hiệu chỉnh, đúng dạng 1/√N trong góp ý, được giữ để tham khảo.

**Khoảng tin cậy là xấp xỉ.** Fit coi các rung là độc lập, nhưng các rung của một seed là subset lồng nhau, nên SD ở các rung kề nhau tương quan với nhau. Trên dữ liệu tổng hợp có subset lồng nhau, khoảng 95% chỉ phủ đúng khoảng 89% số lần.

### 5.4 Mô hình phương sai có floor

s² = σ²_floor + c · (1/N − 1/M). Đây là dạng pool hữu hạn của giả thuyết "phương sai do lấy mẫu tỉ lệ với 1/N, cộng một floor". Mô hình chỉ fit theo từng shift.

1. **Fit:** OLS, rồi một bước tinh chỉnh có trọng số với w = ν / (2 f²), trong đó f là s² đã fit.
   - f được chặn dưới ở 1% trung bình s² quan sát được, để trọng số không bùng lên khi f gần 0.
2. **Đại lượng báo cáo:**
   - σ_floor = √max(σ², 0);
   - c;
   - N* = 1 / (σ²/c + 1/M), tức budget tại đó hai phần bằng nhau, chỉ tính khi σ² > 0 và c > 0;
   - tỉ số σ_floor trên nhiễu huấn luyện của shift và trên nhiễu huấn luyện gộp.
3. **Khoảng tin cậy:** bootstrap tham số 1.000 lần. Mỗi lần, s² của từng rung được rút lại thành f · χ²_ν / ν quanh giá trị đã fit, rồi fit lại. Bộ sinh số ngẫu nhiên là `row_rng(--seed + 40, chỉ số shift, chỉ số key)`. Khoảng này có cùng giới hạn độc lập như khoảng của độ dốc.
4. **Cờ `floor_above_training_noise`:** cận dưới của σ_floor cao hơn cận trên của SD nhiễu huấn luyện gộp. Đây là định nghĩa của "floor rõ ràng cao hơn nhiễu".

### 5.5 Luật đọc

- **Độ dốc `power_fpc` phủ −1/2 và nằm dưới 0:** SD giảm đúng như khi lấy trung bình trên các ảnh rút ngẫu nhiên.
- **Floor ở mức nhiễu huấn luyện:** phần phân tán còn lại ở N lớn là nhiễu huấn luyện, mà không cách chọn ảnh nào loại được.
- **Floor rõ ràng cao hơn nhiễu:** còn một phần phân tán phụ thuộc subset, tức còn chỗ cho việc chọn ảnh.
- **Giới hạn, luôn nói kèm:** SD trên các subset ngẫu nhiên chỉ cho biết một lần rút ngẫu nhiên lệch bao nhiêu so với lần khác. Nó không chặn được mức mà một bộ chọn có chủ đích đạt tới, vì active selection có thể tốt hơn mọi subset ngẫu nhiên. Kết quả trả lời câu "subset ngẫu nhiên nào được chọn có quan trọng không", không trả lời câu "active learning có vô ích không".

Ghi chú kỹ thuật: script không dùng scipy.

- Phân vị χ² có dạng đóng khi ν = 1 và ν = 2. Với ν ≥ 3, phân vị được tìm bằng chia đôi trên CDF chính xác (hàm gamma không đầy đủ chuẩn hóa, tính bằng hệ thức truy hồi).
- Digamma và trigamma tại bội số của 1/2 được tính bằng hệ thức truy hồi từ 1/2 và 1.
- Các hàm này có test với giá trị đã biết.

### 5.6 Kết quả

**SD của reliable mAP theo rung** (đơn vị 10⁻³ mAP; trong ngoặc là SD theo đơn vị recovery E = 3):

| Shift | 10 | 25 | 50 | 100 | 250 | 500 | 1.000 | 2.000 | 5.000 | 10.000 |
|---|---|---|---|---|---|---|---|---|---|---|
| compound | 5,7 (0,150) | 4,4 (0,115) | 2,5 (0,065) | 3,0 (0,080) | 0,5 (0,013) | 0,5 (0,014) | 0,2 (0,006) | | | |
| weather | 10,6 (0,687) | 8,9 (0,574) | 5,0 (0,322) | 5,1 (0,332) | 1,3 (0,084) | 0,3 (0,017) | 0,2 (0,014) | | | |
| lighting | 1,2 (0,023) | 1,3 (0,024) | 3,3 (0,064) | 1,7 (0,033) | 2,0 (0,038) | 1,0 (0,018) | 0,6 (0,012) | 1,0 (0,019) | 0,8 (0,015) | 0,4 (0,007) |

**Nhiễu huấn luyện** (đơn vị 10⁻³):

- Từng shift (ν = 1):
  - compound 0,56 [0,25; 17,8];
  - weather 0,16 [0,07; 5,1];
  - lighting 0,03 [0,01; 0,97].
- Gộp (ν = 3): 0,34 [0,19; 1,25].
- Với ν = 1, các khoảng này rất rộng.

**Độ dốc ln SD theo ln N:**

| Shift | Số rung | `power_fpc` (chính) | `power` (chưa hiệu chỉnh) | χ²/df | Phủ −1/2 | Dưới 0 |
|---|---|---|---|---|---|---|
| compound | 7 | −0,65 [−0,96; −0,34] | −0,73 [−1,04; −0,42] | 0,39 | có | có |
| weather | 7 | −0,87 [−1,18; −0,56] | −0,93 [−1,24; −0,62] | 0,68 | không (dốc hơn) | có |
| lighting | 10 | −0,14 [−0,33; +0,04] | −0,18 [−0,37; −0,00] | 0,45 | không | không |
| gộp | 24 | −0,40 [−0,54; −0,26] | −0,45 [−0,59; −0,31] | 1,40 | có | có |

Độ dốc theo từng class (`power_fpc`):

- Ở compound và weather, mọi class đều có độ dốc dưới 0 và phủ −1/2.
- Ở lighting:
  - pedestrian −0,39 [−0,57; −0,21] phủ −1/2;
  - traffic light −0,37 [−0,55; −0,19] cũng phủ −1/2, nhưng χ²/df 3,81, tức fit không tốt;
  - traffic sign −0,31 [−0,49; −0,13] nông hơn −1/2;
  - car gần phẳng: −0,09 [−0,27; +0,10].

**Mô hình phương sai, reliable mAP:**

| Shift | σ_floor (×10⁻³) | c | N* | Rõ ràng cao hơn nhiễu |
|---|---|---|---|---|
| compound | 0,0 [0,0; 2,1] | 4,5 × 10⁻⁴ | không xác định (σ² ≤ 0) | không |
| weather | 0,0 [0,0; 4,1] | 1,7 × 10⁻³ | không xác định (σ² ≤ 0) | không |
| lighting | 1,5 [0,9; 2,0] | 1,2 × 10⁻⁵ (khoảng tin cậy phủ 0) | khoảng 5 | không |

- Trong 15 hàng (3 shift × 5 key), chỉ compound traffic light được gắn cờ. Nhưng ở hàng đó c ≈ 0, tức mô hình không tìm được thành phần giảm theo 1/N, nên hàng này không được đọc.
- Hai hàng khác cũng có c ≤ 0 và không được đọc: compound traffic sign và lighting traffic light. Riêng compound traffic sign còn có khoảng percentile của σ_floor không chứa ước lượng điểm.

**Cách đọc:**

- **Compound và weather:**
  - Compound có độ dốc phủ −1/2 và nằm dưới 0, phù hợp với việc lấy trung bình trên các ảnh rút ngẫu nhiên. Weather giảm còn nhanh hơn −1/2.
  - Ở N = 1.000, SD còn 0,2 × 10⁻³ ở cả hai shift, tức ở mức nhiễu huấn luyện đo được.
- **Lighting:**
  - SD gần phẳng, nhưng thấp ngay từ N = 10: 1,2 × 10⁻³, tức 0,023 đơn vị recovery, so với 5,7 × 10⁻³ ở compound và 10,6 × 10⁻³ ở weather.
  - Floor 1,5 [0,9; 2,0] × 10⁻³, khoảng 0,03 đơn vị recovery, gấp khoảng 4 lần ước lượng điểm của nhiễu gộp. Nhưng khoảng tin cậy của nó chồng lên cận trên 1,25 × 10⁻³ của nhiễu gộp, nên theo luật là "không rõ ràng cao hơn nhiễu".
  - N* ≈ 5 nghĩa là theo mô hình, phần floor đã trội hơn phần 1/N ngay từ rung đầu tiên. Khoảng tin cậy của N* rất rộng (0,2 đến 87), và khoảng của c phủ 0.
- **Độ dốc gộp** (−0,40) là trung bình của các shift có độ dốc không tương thích với nhau, nên không đọc riêng.
- **Với câu hỏi "vì sao không dùng active learning":**
  - Compound và weather cho thấy, trong số các subset ngẫu nhiên, chọn ngẫu nhiên đã gần tối ưu.
  - Ở lighting, nếu còn chỗ cho việc chọn ảnh thì chỗ đó rất nhỏ (khoảng 0,03 đơn vị recovery) và chưa phân biệt được với nhiễu huấn luyện.
  - Ở N nhỏ, compound và weather phân tán mạnh: SD 0,15 và 0,69 đơn vị recovery ở N = 10. Phần phân tán này có thể đến từ việc chọn ảnh, hoặc từ nhiễu huấn luyện lớn hơn ở N nhỏ: các rung nhỏ chạy 500 update, khoảng 100 epoch ở N = 10. Hai phần này chưa tách được.
  - Muốn trả lời trực tiếp thì cần một run có active selection.

## 6. Phân tích 3: source retention như một trục thứ hai

### 6.1 Các điểm

Mỗi điểm (shift, phương pháp, N) là trung bình theo seed của các run được chấm trên **cả** test của target **lẫn** test của source. Một seed thiếu một trong hai bộ chấm làm script dừng.

| Phương pháp | N | Seed | Nguồn |
|---|---|---|---|
| source model | 0 | – | campaign |
| FT | các rung ladder; full pool (E = 3) | 0/1/2; full pool 0/1 | campaign |
| FT 12 epoch | full pool | 0/1 | revision |
| ALDI++ (`adapt`) | 0, 10, 100, 1.000 (+5.000 ở lighting) | 0/1 ở N = 0; 0/1/2 | RQ5 |
| replay | 0 (run replay chỉ có ảnh source), 10, 100, 1.000 | 0; thêm 1/2 ở N = 10 và 100 từ revision | RQ5 + revision |
| `adapt_matched`, `replay_matched` | 1.000, 3.000 update (số lượt ảnh có nhãn cân bằng) | 0/1/2 | revision |

Các run revision được phân loại bằng các hàm có sẵn: `read_revision`, `split_revision`, `merge_replay_seeds`. Các hàm này dừng lại khi gặp run không rõ loại, run trùng, hoặc lần lấy lại test không dùng chung.

### 6.2 Hai trục

- **Trục target:**
  - gain = mAP_target − mAP(0);
  - recovery_e3 = gain / (mốc FT toàn pool 3 epoch − mAP(0)), cùng mốc với các bảng RQ5;
  - recovery_e12, tính so với ceiling 12 epoch.
- **Trục source:**
  - retention = mAP_source(model) / mAP_source(source model), với mAP_source(source model) = 0,3422;
  - mức thay đổi tuyệt đối mAP_source(model) − 0,3422.

### 6.3 Khoảng tin cậy

- **Lần lấy lại test:** replicate t lấy lần lấy lại thứ t trên test target và lần lấy lại thứ t trên test source của mọi run.
- **Seed của một điểm** được rút lại một lần cho mỗi replicate, và cùng lần rút đó dùng cho cả hai trục. Vì vậy độ bất định trên hai trục của cùng một điểm đi cùng nhau.
- **Seed của mốc tham chiếu** được rút lại một lần cho mỗi replicate và shift. Mọi điểm của shift chia cho cùng mẫu số đó, giống như các rung của một đường recovery dùng chung lần rút ceiling trong `analyze_campaign.py`.
- **Điểm chính là mốc tham chiếu** (FT ở N = pool đối với E = 3, FT 12 epoch đối với E = 12) dùng đúng lần rút của mốc, nên recovery của nó bằng 1 ở mọi replicate xác định.

### 6.4 Recovery so với ceiling 12 epoch

- Một recovery "không xác định" khi hơn 5% số replicate có mẫu số không vượt quá 0,01. Luật này giống `analyze_rq5.py`.
- Số replicate không xác định phụ thuộc lần rút seed (`--seed`). Script vì vậy báo thêm tỉ lệ kỳ vọng chính xác, tính trên mọi cách rút lại có thứ tự của các seed mốc (đồng khả năng) và mọi lần lấy lại test, không dùng số ngẫu nhiên.
- Với weather: 33,8% số replicate rút được không xác định, kỳ vọng chính xác là 33,3%. Recovery so với ceiling 12 epoch của weather vì vậy không xác định, và kết luận này không phụ thuộc seed. Compound và lighting có 0%.

### 6.5 Pareto front

- Trong mỗi shift, một điểm nằm trên front khi không có điểm nào khác có gain và retention đều ít nhất bằng nó, với ít nhất một trục cao hơn hẳn.
- Front chỉ dựa trên ước lượng điểm và chỉ mang tính mô tả.
- Gain và recovery xếp các điểm của một shift theo cùng một thứ tự, nên front không phụ thuộc vào mốc tham chiếu.

### 6.6 So sánh từng cặp

- **Các cặp:** adapt − FT và replay − FT ở mọi N ≥ 10 mà cả hai nhánh cùng có; adapt_matched − FT và replay_matched − FT ở N = 1.000; adapt − source model và replay − source model ở N = 0.
- **Cách ghép:** theo seed lấy mẫu, chỉ trên các seed hai nhánh cùng có. Các cặp seed được rút lại cùng nhau, một lần cho mỗi replicate, dùng chung cho cả hai trục.
- **Mỗi hàng có:** Δ target mAP (cả theo đơn vị recovery), Δ retention, khoảng tin cậy và số replicate không xác định.
- **Các cờ:**
  - `dominates` ("trội hơn"): cận dưới của cả hai hiệu đều trên 0.
  - `dominated`: cận trên của cả hai hiệu đều dưới 0.
  - `trade_off` ("đánh đổi"): cả hai khoảng tin cậy đều không chứa 0, và hai hiệu ngược dấu.
  - Không cờ nào bật: "không rõ thứ tự".

### 6.7 Thiết kế bộ sinh số ngẫu nhiên

Mỗi hàng có bộ sinh riêng, là con của `SeedSequence(--seed + offset)`:

- điểm: offset 30, khóa (shift, phương pháp, N);
- cặp so sánh: offset 31, khóa (shift, so sánh, N);
- mốc tham chiếu: offset 32, khóa (shift, mốc).

Nhờ vậy khi một hàng có thêm seed, các hàng khác không đổi. Thêm dữ liệu revision cũng không làm xê dịch các hàng mà revision không chạm tới.

### 6.8 Lưu ý thiết kế (luôn nói kèm kết quả)

Mỗi update của ALDI++ và replay có 1 ảnh source có nhãn cộng 1 ảnh target có nhãn; FT chỉ học ảnh target. Retention cao của hai nhánh này vì vậy là do ôn lại source, có sẵn trong thiết kế. Bảng đo đánh đổi của từng công thức huấn luyện như đã chạy, không đo riêng phần adaptation. Nhánh replay có cùng luồng ảnh có nhãn nhưng không có adaptation, nên tách được hai phần, như trong RQ5.

### 6.9 Kết quả

**Các điểm chính** (recovery theo mốc FT toàn pool 3 epoch):

| Shift | FT N = 10 | FT N = 100 | FT N = 1.000 | FT full pool, E = 3 | FT 12 epoch | adapt 10 / 100 / 1.000 | replay 10 / 100 / 1.000 |
|---|---|---|---|---|---|---|---|
| compound: recovery | −1,33 | +0,09 | +0,87 | 1,00 | +1,01 | +0,29 / +0,51 / +0,56 | +0,02 / +0,41 / +0,54 |
| compound: retention | 0,840 | 0,937 | 0,954 | 0,947 | 0,905 | 0,991 / 0,995 / 0,996 | 0,991 / 0,999 / 1,001 |
| weather: recovery | −3,51 | −0,42 | +0,77 | 1,00 | +0,71 | +0,20 / +0,46 / +0,56 | −0,03 / +0,37 / +0,55 |
| weather: retention | 0,831 | 0,949 | 0,982 | 0,984 | 0,957 | 0,993 / 0,997 / 1,000 | 0,991 / 1,000 / 1,005 |
| lighting: recovery | −1,17 | +0,01 | +0,55 | 1,00 | +1,06 | +0,18 / +0,31 / +0,34 | +0,01 / +0,27 / +0,34 |
| lighting: retention | 0,839 | 0,944 | 0,964 | 0,924 | 0,875 | 0,989 / 0,995 / 0,997 | 0,990 / 0,998 / 1,004 |

Khoảng tin cậy của từng điểm nằm trong `retention_points.csv` và trong `docs/results_direction1_vi.md`. Replay ở N = 1.000 chỉ có seed 0.

**So sánh với FT** (Δ target tính bằng AP, Δ retention tính theo tỉ lệ so với source model):

| Shift | N | adapt − FT | replay − FT | Kết luận |
|---|---|---|---|---|
| compound | 10 | +0,0613 AP; +0,150 | +0,0512; +0,151 | cả hai trội hơn |
| compound | 100 | +0,0157; +0,058 | +0,0119; +0,062 | cả hai trội hơn |
| compound | 1.000 | −0,0117; +0,042 | −0,0125; +0,050 | cả hai đánh đổi |
| weather | 10 | +0,0573; +0,162 | +0,0538; +0,160 | cả hai trội hơn |
| weather | 100 | +0,0137; +0,049 | +0,0123; +0,052 | cả hai trội hơn |
| weather | 1.000 | −0,0032; +0,018 | −0,0033; +0,026 | cả hai đánh đổi |
| lighting | 10 | +0,0704; +0,150 | +0,0616; +0,151 | cả hai trội hơn |
| lighting | 100 | +0,0154; +0,051 | +0,0133; +0,054 | cả hai trội hơn |
| lighting | 1.000 | −0,0111; +0,033 | −0,0103; +0,039 | cả hai đánh đổi |
| lighting | 5.000 | −0,0096; +0,027 | – | đánh đổi |

- **Khi cân bằng số lượt ảnh có nhãn ở N = 1.000** (adapt_matched và replay_matched so với FT):
  - compound và lighting vẫn là đánh đổi;
  - ở weather, khoảng tin cậy của Δ target phủ 0 (adapt_matched −0,0021 [−0,0040; +0,0002], replay_matched −0,0015 [−0,0032; +0,0006]), nên không rõ thứ tự.
- **Ở N = 0 (so với source model):**
  - ALDI++ không dùng nhãn target tăng target +0,0111 AP ở compound, +0,0055 ở weather, +0,0091 ở lighting (khoảng tin cậy đều trên 0). Retention không đổi rõ rệt: các khoảng tin cậy phủ 0.
  - Run replay chỉ có ảnh source (seed 0) tăng retention +0,0046 [+0,0021; +0,0076]. Target tăng +0,0031 ở weather và +0,0038 ở lighting, với khoảng tin cậy trên 0, nên ở hai shift này nó trội hơn source model. Ở compound, khoảng tin cậy của target phủ 0.

**Pareto front** (ước lượng điểm):

- compound: FT ở N = 1.000 và 1.738, FT 12 epoch, replay ở N = 0 và 1.000, replay_matched.
- weather: FT ở N = 1.985, replay ở N = 1.000, replay_matched.
- lighting: FT ở N = 2.000, 5.000, 10.000 và 18.015, FT 12 epoch, adapt ở N = 5.000, replay ở N = 0 và 1.000, replay_matched.

**Cách đọc:**

- **Kết quả khớp với dự đoán của giảng viên, ở mức mô tả.** Đây không phải kiểm định, vì chiều target đã biết từ các bảng RQ5 trước khi luật được viết.
  - Ở N = 10 và 100, ALDI++ và replay trội hơn FT ở cả ba shift.
  - Ở N = 1.000 (và lighting N = 5.000), FT cao hơn trên target (+0,003 đến +0,012 AP) nhưng mất source nhiều hơn (retention thấp hơn 0,018 đến 0,050).
- **"Trội hơn" ở N = 10 chủ yếu do FT sụp.** FT với 10 ảnh kém source model trên cả hai trục: recovery −1,17 đến −3,51, retention khoảng 0,84. ALDI++ và replay tránh được cú sụp đó.
- **Retention cao đến từ việc ôn lại source, không phải adaptation.** Replay không có adaptation nhưng giữ source ngang hoặc tốt hơn ALDI++: ở N = 10 hai nhánh chênh nhau tối đa 0,002, ở N = 100 và 1.000 replay cao hơn. Điều này khớp với phép phân tách của RQ5.
- **FT 12 epoch** cho target cao nhất ở lighting (1,06 [1,03; 1,08]). Ở compound nó ngang mốc FT toàn pool 3 epoch (+0,0005 AP, khoảng tin cậy phủ 0). Retention của nó thấp nhất trong các điểm có recovery dương: 0,905 ở compound và 0,875 ở lighting.

## 7. Kiểm thử và kiểm chứng

**Unit test.** Có 86 test mới, tất cả trên dữ liệu tổng hợp tạo trong thư mục tạm, không đọc `runs/` hay `data/`:

- `tests/test_analyze_instances.py`, 27 test, kiểm:
  - số box đếm được là số box mà lúc train giữ lại;
  - chỉ file annotation train được đọc, và record thiếu hoặc trùng làm script dừng;
  - subset của một run trùng với `nested_rows` của code train;
  - bước 0 → N không bao giờ vào hồi quy;
  - OLS tìm lại đúng hệ số đã biết;
  - `within` hấp thụ x_N;
  - tương quan sau khi loại intercept;
  - cách đọc khi hệ số âm.
- `tests/test_analyze_seed_spread.py`, 35 test, kiểm:
  - khoảng χ², digamma và trigamma so với giá trị đã biết;
  - độ dốc của SD đúng 1/√N là −0,5, và với dữ liệu rút từ pool hữu hạn thì độ dốc đã hiệu chỉnh là −0,5;
  - bootstrap tham số;
  - ngưỡng mẫu số chặt;
  - việc truyền `--seed`.
- `tests/test_analyze_retention.py`, 24 test, kiểm:
  - các cờ Pareto trên một ví dụ dựng tay;
  - so sánh từng cặp chỉ dùng seed chung, với cùng một lần rút seed cho cả hai trục;
  - tỉ lệ không xác định kỳ vọng;
  - ngưỡng chặt khi mẫu số đúng bằng 0,01.

Chạy cả bộ test trên máy Windows được 312 test. Tất cả qua, trừ `tests/test_revision_runs.py`: file này import `scripts/local_revision_runs.py`, mà file đó cần `fcntl` (chỉ có trên Linux). Lỗi này có từ trước và không liên quan đến nhánh này.

**Review và kiểm chứng.** Có ba vòng, mỗi vòng do các agent độc lập thực hiện:

1. Review code: mỗi script hai người review độc lập, từng phát hiện được kiểm lại trước khi sửa.
2. Kiểm chứng chỉ đọc: mỗi script một người phản biện, cộng một người đối chiếu plan với code.
3. Kiểm chứng note kết quả: đối chiếu từng con số với CSV, và tìm các chỗ diễn giải vượt quá luật.

Các chỗ sửa sau mỗi vòng được liệt kê ở mục 2.3 và 2.4.

## 8. Giới hạn chung

- **Mô tả, không phải xác nhận.** Luật được viết sau khi đã thấy kết quả từng rung của RQ1–RQ5.
- **Phân tích 1 có độ nhạy thấp.** Phép đọc chính chỉ dựa vào khác biệt giữa 3 seed ở cùng bước ảnh. Các đặc tả gộp lẫn với vị trí trên ladder.
- **Phân tích 2:**
  - Khoảng tin cậy của độ dốc và của mô hình phương sai là xấp xỉ, vì các rung lồng nhau.
  - Nhiễu huấn luyện chỉ đo từ 2 seed mỗi shift.
  - Giả định nhiễu huấn luyện không phụ thuộc N chưa được kiểm tra.
  - Kích thước test khác nhau giữa các shift (756, 933, 4.869 ảnh) có thể góp phần làm SD ở lighting thấp hơn; điều này chưa được kiểm tra.
  - SD trên các subset ngẫu nhiên không cho biết active selection đạt tới đâu.
- **Phân tích 3:**
  - Retention đo đánh đổi của từng công thức huấn luyện như đã chạy, trong đó ALDI++ và replay có ôn lại ảnh source.
  - Replay ở N = 1.000 chỉ có seed 0.
  - Pareto front dựa trên ước lượng điểm.
  - Recovery so với ceiling 12 epoch của weather không xác định.

## 9. File và cách chạy lại

**Code và test trên nhánh:**

| File | Nội dung |
|---|---|
| `scripts/analyze_instances.py` | Phân tích 1 |
| `scripts/analyze_seed_spread.py` | Phân tích 2 |
| `scripts/analyze_retention.py` | Phân tích 3 |
| `tests/test_analyze_instances.py`, `tests/test_analyze_seed_spread.py`, `tests/test_analyze_retention.py` | 86 unit test |
| `.agent/plans/active/2026-11-24-direction-1-analyses.md` | Plan và luật đọc |
| `docs/results_direction1_vi.md` | Note kết quả |
| `docs/results/direction1/` | Bảng và hình (mô tả từng file trong `docs/results/README.md`) |

**Output** (`docs/results/direction1/`):

- `instances/`: `instance_pool_density.csv`, `instance_counts.csv`, `instance_transitions.csv`, `instance_regression.csv`, `instance_recovery.csv`, `instance_summary.json`, `instance_recovery.png`.
- `seed_spread/`: `seed_spread.csv`, `seed_spread_floor.csv`, `seed_spread_fit.csv`, `seed_spread_summary.json`, `seed_spread_reliable.png`, `seed_spread_classes.png`.
- `retention/`: `retention_points.csv`, `retention_paired.csv`, `retention_summary.json`, `retention_compound.png`, `retention_weather.png`, `retention_lighting.png`.

**Chạy lại** từ `scripts/`, chỉ dùng CPU, seed mặc định `--seed 0`:

```
python analyze_instances.py --scored ../runs/local/campaign/scored --dataset-root ../data/bdd100k_revised --output-dir ../runs/local/direction1/instances
python analyze_seed_spread.py --scored ../runs/local/campaign/scored --output-dir ../runs/local/direction1/seed_spread
python analyze_retention.py --ft-scored ../runs/local/campaign/scored --da-scored ../runs/local/rq5/scored --revision-scored ../runs/local/revision/scored --output-dir ../runs/local/direction1/retention
```

`analyze_instances.py` cần thư mục `data/bdd100k_revised/` (manifest và file annotation train); hai script còn lại chỉ cần các thư mục `scored/` đã có trong git. Sau khi chạy lại, chép output sang `docs/results/direction1/`.

**Unit test:**

```
python -m unittest discover -s tests -p "test_analyze_*.py"
```
