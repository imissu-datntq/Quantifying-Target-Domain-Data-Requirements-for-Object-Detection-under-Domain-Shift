# Kết quả RQ1–RQ4 (2026-11-06)

Toàn bộ nghiên cứu được chạy trên một máy local (RTX 3060 12 GB), theo các luật đã ghi trước trong `.agent/DECISIONS.md`. Các bước gồm:

- dựng split;
- train source model;
- chạy gate zero-shot và pilot chọn schedule;
- chạy grid 87 run.

Plan đóng băng của campaign này được chấm **đúng một lần**:

- Plan đóng băng: `runs/local/campaign/final_plan.json`, schema 3, sha256 `a1c5eb9f…9436a44`.
- 87 bộ prediction, 176 lượt chấm.
- Bootstrap dùng 1.000 lần lấy mẫu lại ảnh test, seed 20261030.

Số liệu gốc của mọi bảng và hình nằm trong `runs/local/campaign/analysis_rev2/` (CSV, `summary.json`, hình `recovery_<domain>.png` theo ceiling 3 epoch và `recovery_<domain>_e12.png` theo ceiling 12 epoch). Thư mục `analysis/` là bản 2026-11-06 và không dùng cho báo cáo. Vì `runs/` không nằm trong git, một bản sao của các bảng và hình này được lưu ở `docs/results/campaign/`.

## Cập nhật theo nhận xét của giảng viên (2026-11-21 đến 2026-11-23)

Nhận xét nằm ở `docs/review_2026-11-21_lecturer_vi.md`. Các luật đọc kết quả được ghi trong `.agent/DECISIONS.md` (mục 2026-11-21 và hai phần bổ sung) trước khi các run mới được chấm. Riêng hai luật báo cáo ở mục 2026-11-23 được đặt sau khi đã thấy kết quả, nên chỉ mang tính mô tả: (a) recovery có quá 5% lần bootstrap không xác định thì ghi là không xác định; (b) hai ceiling được trình bày cạnh nhau, không chọn bên nào là tốt hơn.

- **Ceiling 12 epoch đã chạy và chấm xong** (10 run 12 epoch, cùng 2 run 3 epoch bổ sung seed thứ hai cho ceiling pool 1.738; cả 12 run được chấm đúng một lần, trong plan đóng băng của 42 run bổ sung). Không shift nào có ceiling cao hơn đáng kể so với ceiling 3 epoch; xem mục "Ceiling 12 epoch so với 3 epoch". Mọi ngưỡng được báo theo cả hai ceiling.
- **Bỏ ngưỡng N95.** Ngưỡng nào chỉ đạt ở chính bậc ceiling được ghi là "không xác định trong pool".
- **Fit lũy thừa được thay** bằng fit log-tuyến tính trên vùng dương của đường cong.
- **Kết luận về tương tác ở RQ1 được viết lại** thành giới hạn phát hiện.
- **Thêm bảng mẫu số của recovery** (ceiling − zero-shot).
- **Ceiling pool 1.738 ở 3 epoch giờ có 2 seed** cho lighting và weather, nên bảng RQ4 thay đổi nhẹ.

Các bảng mới nằm trong `runs/local/campaign/analysis_rev2/`. Mọi con số của phân tích theo ceiling 3 epoch toàn pool được tái lập giống hệt bản trước.

## Các bước chốt trước khi chạy

| | Giá trị |
|---|---|
| Phần cứng | RTX 3060 12 GB, một máy |
| Split | 7/7 hash SHA256 khớp `data/split_reference/` |
| Source model, source val | 0,3517 |
| Gate zero-shot (lighting / weather / compound) | 0,100 / 0,032 / 0,129 |
| Schedule chốt bằng pilot | S=500, **E=3**, LR 1e-3 |

**E = 3 được chọn với biên rất mỏng.** Luật (DECISIONS 2026-10-30) chọn E nhỏ nhất nằm trong δ = 0,005 so với E tốt nhất. Trên lighting val, E2 kém E5 0,00511, tức nằm **ngoài δ đúng 0,0001**, nên luật chọn E3. Nhóm em áp dụng luật đúng như đã viết, không tranh luận lại. Hệ quả:

- N ≤ 333 nhận 500 update.
- N ≥ 500 nhận 1,5·N update (3 epoch ở batch 2).

## Thiết lập

- **Model:** Faster R-CNN R50-FPN ở 1280×720, khởi tạo từ source model, giữ update cuối.
- **Huấn luyện:** LR 1e-3, batch 2, AMP, warmup 50.
- **Seed:** 3 seed lấy mẫu cho mỗi bậc N; 2 seed cho ceiling.
- **Metric chính:** reliable mAP50:95 (pedestrian, car, traffic light, traffic sign).
- **Recovery:** (mAP(N) − mAP(0)) / (mAP_c − mAP(0)), không cắt; mẫu số tối thiểu 0,01.

**Kiểm tra sanity:**

- Source model trên source test đạt 0,342.
- Theo ceiling 3 epoch, recovery tại toàn pool bằng 1 theo định nghĩa ở cả 3 shift.
- Theo ceiling 3 epoch toàn pool, không lần bootstrap nào cho recovery không xác định ở metric chính. Theo ceiling 12 epoch, weather có 339 trong 1.000 lần không xác định (mục "Ceiling 12 epoch so với 3 epoch").

## RQ1: domain shift làm giảm bao nhiêu?

| Shift | mAP(0) zero-shot | Drop (CI 95%) | Ceiling 3 epoch | Gap còn lại (3 epoch) | Ceiling 12 epoch | Gap còn lại (12 epoch) |
|---|---:|---:|---:|---:|---:|---:|
| lighting | 0,244 | 0,098 (0,092–0,106) | 0,296 | 0,046 | 0,299 | 0,043 |
| weather | 0,320 | 0,022 (0,012–0,032) | 0,336 | 0,007 | 0,331 | 0,011 |
| compound | 0,219 | 0,123 (0,112–0,133) | 0,257 | 0,085 | 0,258 | 0,084 |

"Gap còn lại" là mAP của source model trên source test (0,342) trừ ceiling.

- **Tương tác** drop(compound) − drop(lighting) − drop(weather) = +0,002 (CI −0,012 đến +0,016).
  - Ước lượng điểm gần 0, nhưng phép kiểm này yếu. Nửa độ rộng CI là 0,014 AP, bằng 63% toàn bộ hiệu ứng weather (0,022 AP).
  - Tương tác nhỏ nhất phát hiện được với power 80% là 0,020 AP, tức 91% hiệu ứng weather. CI còn tương thích với một tương tác lớn tới 0,016 AP (72%).
  - Kết luận đúng vì vậy là: **dữ liệu này không phát hiện được tương tác nhỏ hơn khoảng 0,014–0,020 AP.** Đây không phải bằng chứng rằng shift kép cộng tính.

**Mẫu số của recovery** là ceiling − zero-shot. Nếu nó không vượt 0,01 AP thì recovery được coi là không xác định. CI 95% lấy từ bootstrap.

| Shift | Mẫu số theo ceiling 3 epoch (CI 95%) | So với ngưỡng 0,01 | Mẫu số theo ceiling 12 epoch (CI 95%) | So với ngưỡng 0,01 |
|---|---:|---:|---:|---:|
| lighting | 0,052 (0,050–0,054) | 5,2 lần | 0,055 (0,053–0,058) | 5,5 lần |
| compound | 0,038 (0,034–0,042) | 3,8 lần | 0,038 (0,034–0,043) | 3,8 lần |
| weather | 0,015 (0,012–0,018) | 1,5 lần | 0,011 (0,007–0,014) | 1,1 lần |

- **Ô weather gần như rỗng:** mẫu số của weather chỉ gấp rưỡi ngưỡng với ceiling 3 epoch, và nằm sát ngưỡng với ceiling 12 epoch.
- Con số 0,016 AP (1,6 lần) trong nhận xét của giảng viên lấy từ số đã làm tròn (0,336 − 0,320); số chưa làm tròn là 0,0154 AP, tức 1,5 lần ngưỡng.
- Weather theo từng lớp, ceiling 3 epoch: pedestrian 0,022 (2,2 lần), car 0,018 (1,8 lần), traffic light 0,013 (1,3 lần), traffic sign 0,009 (0,9 lần, không xác định).
- Trong 1.000 lần bootstrap với ceiling 3 epoch, mẫu số rơi xuống dưới ngưỡng 115 lần với traffic light và 713 lần với traffic sign.

## RQ2: đường cong recovery

Recovery của reliable mAP so với ceiling 3 epoch toàn pool (bảng theo cả hai ceiling nằm ở mục "Ceiling 12 epoch so với 3 epoch"). Mỗi điểm là trung bình 3 seed, kèm CI 95% (bootstrap phân cấp).

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 10 | −1,17 (−1,23; −1,10) | −3,51 (−4,67; −2,71) | −1,33 (−1,59; −1,07) |
| 25 | −0,78 (−0,83; −0,72) | −2,26 (−3,12; −1,60) | −0,90 (−1,11; −0,71) |
| 50 | −0,36 (−0,43; −0,29) | −1,39 (−2,01; −0,96) | −0,36 (−0,49; −0,22) |
| 100 | 0,01 (−0,03; 0,05) | −0,42 (−0,85; −0,10) | 0,09 (−0,01; 0,20) |
| 250 | 0,33 (0,28; 0,36) | 0,22 (0,05; 0,35) | 0,52 (0,46; 0,58) |
| 500 | 0,44 (0,41; 0,46) | 0,50 (0,39; 0,60) | 0,66 (0,61; 0,71) |
| 1000 | 0,55 (0,52; 0,57) | 0,77 (0,66; 0,86) | 0,87 (0,82; 0,91) |
| 2000 | 0,66 (0,63; 0,69) |  |  |
| 5000 | 0,80 (0,78; 0,82) |  |  |
| 10000 | 0,91 (0,89; 0,93) |  |  |
| toàn pool | 1 (18.015) | 1 (1.985) | 1 (1.738) |

**Kết quả chính:** với N nhỏ, fine-tune làm model tệ hơn zero-shot.

- Recovery âm ở N ≤ 50 ở cả 3 shift, và ở N = 100 với weather.
- Pilot cũng báo trước điều này: N=10 trên lighting val thấp hơn N0 tới 0,05–0,06 AP.
- Nguyên nhân là schedule cố định: 10 ảnh nhận 500 update, tức khoảng 100 epoch.

**Theo từng lớp** (recovery tại N=1000, theo ceiling 3 epoch):

| Shift | pedestrian | car | traffic light | traffic sign |
|---|---:|---:|---:|---:|
| lighting | 0,46 | 0,60 | 0,51 | 0,58 |
| weather | 0,82 | 0,82 | không xác định (105/1.000) | không xác định |
| compound | không xác định (120/1.000) | 0,88 | 0,88 | 0,82 |

- Ô ghi "không xác định (k/1.000)" là ô có quá 5% lần bootstrap không xác định, vì mẫu số của lớp đó nằm sát ngưỡng 0,01.
- Ở lighting, theo ceiling 3 epoch pedestrian hồi phục chậm nhất (0,46 tại N=1000). Theo ceiling 12 epoch thì traffic light chậm nhất (0,44) và pedestrian đạt 0,54 (0,40; 0,69), vì 12 epoch nâng ceiling của traffic light và hạ ceiling của pedestrian. Thứ tự giữa các lớp vì vậy phụ thuộc vào ceiling.
- Ở compound và weather tại N=1000, các lớp còn trích dẫn được đi gần nhau.
- Weather traffic sign **không xác định**: ceiling chỉ hơn zero-shot 0,0089 AP, dưới mẫu số tối thiểu 0,01.

## RQ3: cần bao nhiêu ảnh target?

Ngưỡng tính theo kiểu bền vững: bậc N nhỏ nhất mà từ đó trở lên recovery luôn đạt mức yêu cầu. Khoảng trong ngoặc lấy từ dải CI 95%.

| Shift | N_BE | N75 | N90 |
|---|---|---|---|
| lighting | 100 (100–250) | 5.000 (5.000) | 10.000 (10.000–toàn pool) |
| weather | 250 (250) | 1.000 (1.000–toàn pool) | không xác định trong pool (> 1.000) |
| compound | 100 (100–250) | 1.000 (1.000) | không xác định trong pool (> 1.000) |

**Vì sao N90 của weather và compound không xác định.** Bậc lớn nhất thật sự được đo là N = 1.000, với recovery 0,77 (weather) và 0,87 (compound). Bậc tiếp theo là chính ceiling, có recovery bằng 1 theo định nghĩa, nên "N90 = toàn pool" không phải là một phép đo. Điều duy nhất dữ liệu cho biết là N90 lớn hơn 1.000. Với compound, cận trên CI của recovery tại N = 1.000 chạm 0,91, nên dải CI của N90 bắt đầu từ 1.000. Cột N95 đã được bỏ vì nó rơi vào đúng trường hợp này ở cả ba shift.

**Ba ngưỡng cần đọc thận trọng:**

- **Weather N75 = 1.000.** Recovery N=1000 là 0,77, vừa qua 0,75. Cận dưới CI là 0,66, nên ngưỡng này chưa chắc chắn: dải CI kéo tới toàn pool.
- **Lighting N90 = 10.000.** Recovery N=10.000 là 0,91. Cận dưới CI là 0,895, ngay dưới mức 0,90, nên dải CI kéo tới toàn pool.
- **Lighting N_BE = 100.** Recovery N=100 là +0,01, CI chứa 0. Đây là hòa vốn sát nút, không phải lợi ích rõ ràng.

Mỗi ngưỡng là của **một schedule cụ thể** (S = 500, E = 3), không phải của riêng dữ liệu.

**Fit log-tuyến tính trên vùng dương**, Recovery = α + β·ln N, chỉ dùng để nội suy giữa các bậc. Nó thay cho fit lũy thừa của bản trước: fit đó có R² cao chủ yếu nhờ các điểm âm ở N nhỏ và cho ngưỡng lạc quan (lighting N75 = 2.530, trong khi bậc N = 2.000 mới đạt recovery 0,66).

- **Vùng fit:** từ bậc đầu tiên có cận dưới CI 95% lớn hơn 0 (và giữ như vậy ở mọi bậc lớn hơn) tới bậc lớn nhất dưới ceiling. Bậc ceiling bị loại vì recovery của nó bằng 1 theo định nghĩa. Cả ba shift bắt đầu từ N = 250.
- **Cần ít nhất 4 bậc.** Chỉ lighting đủ (6 bậc, từ 250 đến 10.000). Weather và compound chỉ có 3 bậc (250, 500, 1.000) nên không fit; kết quả của hai shift này là khoảng ladder ở bảng trên.
- **R² và RMSE** chỉ tính trên vùng fit. Ngưỡng từ fit chỉ được báo khi nó là nội suy trong vùng fit và nằm trong khoảng ladder tương ứng.

| Lighting, toàn pool | Giá trị |
|---|---|
| Vùng fit | N = 250–10.000, 6 bậc, đơn điệu |
| β (mỗi đơn vị ln N) | 0,159 |
| R² / RMSE trên vùng fit | 0,9999 / 0,002 |
| N75 từ fit (CI 95%) | 3.570 (3.250–3.960); khoảng ladder (2.000; 5.000] |
| N90 từ fit (CI 95%) | 9.200 (8.280 đến > 10.000); khoảng ladder (5.000; 10.000] |
| Recovery dự đoán tại toàn pool (18.015 ảnh) | 1,007 (theo định nghĩa là 1) |

- **Cận trên của N90 bị chặn.** Chỉ 891 trong 1.000 lần bootstrap cho ngưỡng nằm trong vùng fit, và cận trên của CI rơi ra ngoài bậc 10.000, nên chỉ viết được là "> 10.000".
- **Kiểm tra ngoài mẫu.** Fit chỉ trên ba bậc 250, 500, 1.000 dự đoán recovery 0,663 / 0,812 / 0,924 tại N = 2.000 / 5.000 / 10.000, so với 0,660 / 0,804 / 0,911 đo được. Sai số tăng dần theo N (+0,003; +0,008; +0,013) và là +0,020 tại toàn pool.
- **Đường cong chưa bão hòa theo dữ liệu.** Điểm toàn pool nằm gần như đúng trên đường log-tuyến tính của các bậc nhỏ hơn (dự đoán 1,007): thêm dữ liệu vẫn làm recovery tăng theo cùng độ dốc. Phần còn thiếu do schedule thì nhỏ: so với ceiling 12 epoch, điểm toàn pool 3 epoch đã đạt recovery 0,95 (mục "Ceiling 12 epoch so với 3 epoch").
- **Độ nhạy với cách chọn vùng.** Nếu vùng fit bắt đầu từ N_BE điểm (N = 100, 7 bậc) thì R² giảm còn 0,973, recovery dự đoán tại toàn pool là 1,053, N75 = 3.450 và N90 = 7.820. Bậc N = 100 (recovery 0,01, CI chứa 0) chưa thuộc vùng log-tuyến tính.

Luật chọn vùng fit được viết sau khi đã thấy đường cong, nên fit này chỉ mang tính mô tả.

## Ceiling 12 epoch so với 3 epoch

Nhận xét của giảng viên: ceiling 3 epoch có thể chưa được huấn luyện đủ, khiến mẫu số của recovery thấp và mọi ngưỡng lạc quan. Nhóm em chạy lại ceiling với 12 epoch (bằng số epoch của source model), 2 seed, cho toàn pool của cả ba shift và cho pool 1.738 của lighting và weather. Mọi thiết lập khác giữ nguyên, và model giữ lại vẫn là update cuối.

Luật ghi trước khi chấm: ceiling 12 epoch thay ceiling 3 epoch làm mẫu số **bất kể kết quả**; "cao hơn đáng kể" nghĩa là chênh lệch ghép cặp ≥ 0,005 AP với cận dưới CI trên 0; cả hai bảng ngưỡng đều được công bố.

| Shift | Pool | Ceiling 3 epoch | Ceiling 12 epoch | Chênh lệch ghép cặp (CI 95%) | Cao hơn đáng kể? |
|---|---|---:|---:|---:|---|
| lighting | toàn pool (18.015) | 0,2960 | 0,2989 | +0,0029 (+0,0014; +0,0041) | không |
| lighting | 1.738 | 0,2777 | 0,2756 | −0,0021 (−0,0031; −0,0007) | không |
| weather | toàn pool (1.985) | 0,3355 | 0,3310 | −0,0045 (−0,0067; −0,0025) | không |
| weather | 1.738 | 0,3344 | 0,3300 | −0,0044 (−0,0067; −0,0022) | không |
| compound | toàn pool (1.738) | 0,2572 | 0,2578 | +0,0005 (−0,0021; +0,0037) | không |

- Ceiling 12 epoch của lighting là 108.096 update × 2 ảnh ≈ 216.000 lượt ảnh, so với khoảng 54.000 của ceiling 3 epoch và 118.000 của source model.
- **Không ceiling nào cao hơn đáng kể.** Ở lighting toàn pool, 12 epoch hơn 3 epoch 0,003 AP; chênh lệch có ý nghĩa thống kê nhưng dưới mức 0,005 đã đặt.
- **Ở các pool nhỏ, 12 epoch không giúp gì hoặc làm tệ đi.** Compound không đổi; weather và pool 1.738 của lighting thấp đi 0,0021–0,0045 AP (đều dưới mức 0,005). Điều này phù hợp với việc 12 epoch bắt đầu overfit ở cỡ pool này, nhưng chưa được kiểm chứng, và compound (cũng 1.738 ảnh) không giảm.
- **Theo từng lớp,** hai ô cao hơn đáng kể: traffic light (+0,0077) và traffic sign (+0,0053) của lighting toàn pool. Theo chiều ngược lại (thấp hơn ít nhất 0,005 với cận trên CI dưới 0; tiêu chí đối xứng này chỉ mang tính mô tả) có năm ô: pedestrian của lighting toàn pool (−0,0055), pedestrian của weather (−0,0062 toàn pool; −0,0071 pool 1.738) và traffic sign của weather (−0,0053; −0,0067).
- **Quên source nặng hơn ở 12 epoch:** trên source test, reliable mAP của ceiling 12 epoch thay đổi −0,043 / −0,015 / −0,032 AP so với source model (lighting / weather / compound), so với −0,026 / −0,005 / −0,018 của ceiling 3 epoch.

**Ngưỡng theo hai ceiling** (reliable, toàn pool). Với ceiling 12 epoch, run 3 epoch trên toàn pool trở thành bậc cao nhất của ladder và là một phép đo, không còn bằng 1 theo định nghĩa.

| Shift | Ngưỡng | Theo ceiling 3 epoch | Theo ceiling 12 epoch |
|---|---|---|---|
| lighting | N_BE | 100 | 100 |
|  | N75 | 5.000 | 5.000 |
|  | N90 | 10.000 | toàn pool (18.015) |
| weather | N_BE | 250 | không xác định |
|  | N75 | 1.000 | không xác định |
|  | N90 | không xác định trong pool (> 1.000) | không xác định |
| compound | N_BE | 100 | 100 |
|  | N75 | 1.000 | 1.000 |
|  | N90 | không xác định trong pool (> 1.000) | toàn pool (1.738) |

- **Lighting:** N_BE không đổi (100, dải CI 100–250). N75 giữ bậc 5.000 nhưng sát nút: recovery tại N = 5.000 là 0,76 (0,73; 0,79), nên dải CI của N75 là 5.000–10.000 (với ceiling 3 epoch là 5.000). N90 dời từ 10.000 lên toàn pool, vì recovery tại N = 10.000 giảm từ 0,91 (0,89; 0,93) xuống 0,86 (0,84; 0,89).
- **Compound:** N_BE và N75 không đổi. N90 trước đây không xác định trong pool; với ceiling 12 epoch, bậc toàn pool 3 epoch là một phép đo và đạt recovery 0,99 (0,91; 1,05), nên N90 = 1.738 (dải CI 1.000–1.738).
- **Weather không xác định theo ceiling 12 epoch.** Mẫu số chỉ còn 0,011 AP (1,09 lần ngưỡng), và 339 trong 1.000 lần bootstrap của đường recovery không xác định (bảng mẫu số, với lần lấy lại seed riêng, đếm 330). Luật báo cáo đặt ngày 2026-11-23, sau khi đã thấy kết quả: quá 5% lần bootstrap không xác định thì recovery và ngưỡng không được trích dẫn. Run 3 epoch trên toàn pool còn cao hơn ceiling 12 epoch 0,0045 AP.

**Weather theo AP thô** (reliable mAP, trung bình seed; zero-shot 0,3201). CI chỉ có ở ba bậc được tính trong phân tích RQ5:

| N | mAP | So với zero-shot (CI 95%) |
|---:|---:|---:|
| 10 | 0,2659 | −0,0542 (−0,0641; −0,0443) |
| 25 | 0,2852 | −0,0349 |
| 50 | 0,2986 | −0,0215 |
| 100 | 0,3136 | −0,0065 (−0,0118; −0,0017) |
| 250 | 0,3236 | +0,0035 |
| 500 | 0,3279 | +0,0078 |
| 1.000 | 0,3319 | +0,0118 (+0,0088; +0,0142) |
| toàn pool, 3 epoch | 0,3355 | +0,0154 |
| toàn pool, 12 epoch | 0,3310 | +0,0109 |

Bậc N = 1.000 đã ngang ceiling 12 epoch, và bậc toàn pool 3 epoch cao hơn nó.

Recovery theo hai ceiling (trung bình seed, CI 95%):

| N | lighting, ceiling 3 epoch | lighting, ceiling 12 epoch | compound, ceiling 3 epoch | compound, ceiling 12 epoch |
|---:|---:|---:|---:|---:|
| 100 | 0,01 (−0,03; 0,05) | 0,01 (−0,02; 0,05) | 0,09 (−0,01; 0,20) | 0,09 (−0,01; 0,20) |
| 250 | 0,33 (0,28; 0,36) | 0,31 (0,27; 0,34) | 0,52 (0,46; 0,58) | 0,51 (0,44; 0,59) |
| 500 | 0,44 (0,41; 0,46) | 0,41 (0,38; 0,44) | 0,66 (0,61; 0,71) | 0,65 (0,58; 0,72) |
| 1.000 | 0,55 (0,52; 0,57) | 0,52 (0,50; 0,55) | 0,87 (0,82; 0,91) | 0,85 (0,78; 0,92) |
| 2.000 | 0,66 (0,63; 0,69) | 0,63 (0,60; 0,65) |  |  |
| 5.000 | 0,80 (0,78; 0,82) | 0,76 (0,73; 0,79) |  |  |
| 10.000 | 0,91 (0,89; 0,93) | 0,86 (0,84; 0,89) |  |  |
| toàn pool, 3 epoch | 1 (theo định nghĩa) | 0,95 (0,93; 0,97) | 1 (theo định nghĩa) | 0,99 (0,91; 1,05) |

**Fit log-tuyến tính theo hai ceiling.** Với ceiling 12 epoch, bậc toàn pool 3 epoch thuộc vùng fit, nên compound có đủ 4 bậc.

| Shift | Ceiling | Vùng fit | R² | N75 từ fit (CI 95%) | N90 từ fit (CI 95%) |
|---|---|---|---:|---:|---:|
| lighting | 3 epoch | 250–10.000, 6 bậc | 0,9999 | 3.570 (3.250–3.960) | 9.200 (8.280–> 10.000) |
| lighting | 12 epoch | 250–18.015, 7 bậc | 0,9999 | 4.690 (4.070–5.340) | 12.800 (10.870–14.970) |
| compound | 12 epoch | 250–1.738, 4 bậc | 0,9937 | 680 (530–880) | 1.240 (980–1.630) |

- **Lighting:** N75 nằm trong khoảng 3.570–4.690 ảnh và N90 trong khoảng 9.200–12.800 ảnh, tùy ceiling. Ngưỡng theo ceiling 3 epoch là cận dưới, theo ceiling 12 epoch là cận trên, vì các bậc lớn của ladder vẫn chỉ được huấn luyện 3 epoch.
- Fit của weather theo ceiling 12 epoch không được trích dẫn, vì cùng lý do mẫu số ở trên.
- **Kiểm tra ngoài mẫu theo ceiling 12 epoch.** Lighting: fit trên ba bậc 250, 500, 1.000 dự đoán 0,628 / 0,769 / 0,876 / 0,966 tại N = 2.000 / 5.000 / 10.000 / 18.015, so với 0,625 / 0,762 / 0,863 / 0,948 đo được (sai số +0,003; +0,008; +0,013; +0,019). Compound: dự đoán 0,978 tại N = 1.738, so với 0,986 đo được (−0,008). RMSE trên vùng fit: 0,003 (lighting) và 0,014 (compound).
- **Độ nhạy với cách chọn vùng, ceiling 12 epoch.** Nếu vùng fit bắt đầu từ N_BE điểm (N = 100): lighting có 8 bậc, R² 0,977, N75 = 4.650, N90 = 11.330; compound có 5 bậc, R² 0,975, N75 = 720, N90 = 1.170.

**Kết luận cho nhận xét về ceiling.** Ceiling 3 epoch không bị undertrain đáng kể: 12 epoch chỉ thêm 0,003 AP ở pool lớn nhất và không thêm gì ở các pool nhỏ. Các ngưỡng của lighting và compound vì vậy gần như giữ nguyên: ngưỡng duy nhất đổi bậc là N90 của lighting (từ 10.000 lên toàn pool 18.015); N90 của compound từ "không xác định trong pool (> 1.000)" thành 1.738, vì bậc toàn pool 3 epoch nay là một phép đo. Riêng weather thì cả ba ngưỡng trở thành không xác định theo ceiling 12 epoch.

## RQ4: so sánh giữa các shift với cùng cỡ pool (1.738 ảnh)

Ceiling 3 epoch ở pool 1.738, 2 seed cho mỗi shift:

| Shift | Recovery N=500 (CI 95%) | Recovery N=1000 (CI 95%) | N75 | N90 |
|---|---:|---:|---|---|
| lighting | 0,67 (0,63; 0,72) | 0,85 (0,81; 0,90) | 1.000 | không xác định trong pool (> 1.000) |
| weather | 0,54 (0,42; 0,65) | 0,82 (0,71; 0,92) | 1.000 (dải CI 1.000–1.738) | không xác định trong pool (> 1.000) |
| compound | 0,66 (0,61; 0,71) | 0,87 (0,82; 0,91) | 1.000 | không xác định trong pool (> 1.000) |

Ở cỡ pool 1.738, vùng dương chỉ có 3 bậc (250, 500, 1.000), nên không shift nào đủ điều kiện fit theo ceiling 3 epoch.

- Ở cùng cỡ pool, lighting và compound hồi phục gần như cùng tốc độ.
- Weather chậm hơn ở N=500 nhưng bắt kịp ở N=1000. Cả 3 shift có N75 = 1.000.
- Kết luận RQ4 (theo ceiling 3 epoch): khác biệt giữa các shift ở RQ3 chủ yếu đến từ **độ lớn ceiling và cỡ pool**, không phải tốc độ học.
- **Theo ceiling 12 epoch ở pool 1.738:** lighting có N75 = N90 = 1.000 (dải CI: N75 500–1.000, N90 1.000–1.738; recovery tại N = 1.000 là 0,91 (0,84; 0,97)), compound có N75 = 1.000 và N90 = 1.738, weather không xác định (mẫu số 0,0099 AP, dưới ngưỡng 0,01). Run 3 epoch ở pool 1.738 của lighting đạt recovery 1,07 (0,98; 1,17), tức ngang hoặc hơn ceiling 12 epoch.

## Kết quả phụ

**Quên source (forgetting).** Thay đổi reliable mAP trên source test so với source model (trung bình seed):

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 10 | −0,055 | −0,058 | −0,055 |
| 100 | −0,019 | −0,018 | −0,022 |
| 1000 | −0,012 | −0,006 | −0,016 |
| toàn pool | −0,026 | −0,005 | −0,018 |

Mức quên nặng nhất ở N=10. Ở lighting và compound nó nhẹ nhất ở N = 250 (−0,011 và −0,014) rồi tăng lại khi N lớn vì số update tăng (toàn pool: −0,026 và −0,018); ở weather nó giảm dần tới toàn pool (−0,005).

**Đối chứng khởi tạo COCO** (reliable mAP):

| Shift, N | COCO-init | Source-init | Zero-shot |
|---|---:|---:|---:|
| compound 100 / 1000 | 0,149 / 0,198 | 0,223 / 0,252 | 0,219 |
| weather 100 / 1000 | 0,209 / 0,267 | 0,314 / 0,332 | 0,320 |
| lighting 100 / 1000 | 0,171 / 0,220 | 0,244 / 0,273 | 0,244 |

Pretrain trên source chiếm phần lớn hiệu năng; với 1.000 ảnh target, COCO-init vẫn thấp hơn zero-shot.

## Hạn chế cần nêu

1. **E = 3 được chọn với biên rất mỏng** (E2 nằm ngoài δ đúng 0,0001 trên lighting val). Với E = 2, mọi bậc N ≥ 500 sẽ nhận số update ít hơn một phần ba (N thay vì 1,5·N).
2. **Schedule cố định gây recovery âm ở N nhỏ**, và số update tăng theo N khi N ≥ 500. Đường cong đo "dữ liệu cộng với lượng huấn luyện tương ứng".
3. **Weather:** mẫu số của recovery chỉ 0,015 AP với ceiling 3 epoch (1,5 lần ngưỡng 0,01) và 0,011 AP với ceiling 12 epoch; traffic sign nằm dưới ngưỡng. Ngưỡng N75 = 1.000 theo ceiling 3 epoch có dải CI kéo tới toàn pool; theo ceiling 12 epoch thì không xác định. Luôn báo weather kèm CI và mức thay đổi thô.
4. **Ceiling là ceiling của schedule.** Ceiling 12 epoch chỉ hơn ceiling 3 epoch 0,003 AP ở lighting và không hơn ở các pool nhỏ. Các bậc của ladder vẫn chỉ được huấn luyện 3 epoch, nên ngưỡng theo ceiling 12 epoch là cận trên ở những bậc lớn mà chính bậc đó có thể chưa được huấn luyện đủ.
5. **Nhãn** là BDD100K bản 2018.
6. **Fit log-tuyến tính chỉ mang tính mô tả:** theo ceiling 3 epoch chỉ có cho lighting; theo ceiling 12 epoch có thêm compound (4 bậc, 250–1.738). Luật chọn vùng fit được viết sau khi đã thấy đường cong.
7. **Phép kiểm tương tác ở RQ1 yếu:** nó chỉ phát hiện được tương tác từ khoảng 0,014–0,020 AP, tức 63–91% hiệu ứng weather.

## Trạng thái

Mọi run của nghiên cứu đã chạy và được chấm xong. Mỗi plan đóng băng được chấm đúng một lần: campaign này (87 run), grid RQ5 (46 run) và 42 run bổ sung theo nhận xét của giảng viên (12 run ceiling ở ghi chú này, 30 run ở `docs/results_rq5_vi.md`).
