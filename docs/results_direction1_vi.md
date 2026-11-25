# Kết quả hướng 1: mật độ instance, độ phân tán theo seed và source retention (2026-11-25)

Ghi chú này trả lời ba ý trong phần góp ý tiếp theo của giảng viên (hướng 1):

1. **Mật độ pedestrian.** Source có 13.170 pedestrian trên 9.804 ảnh (1,34 mỗi ảnh); lighting có 12.265 trên 18.015 ảnh (0,68 mỗi ảnh). Pedestrian ban đêm recover chậm nhất (0,46) có thể một phần do ít nhãn hơn chứ không chỉ do khó về hình ảnh. Câu hỏi: AP của một class tăng theo số instance của class đó hay theo số ảnh?
2. **Độ lệch chuẩn theo N.** SD giảm như 1/√N thì chọn ảnh ngẫu nhiên gần tối ưu; SD chững lại thì còn chỗ cho active selection ("vì sao không dùng active learning").
3. **Source retention là một trục.** Vẽ (target recovery, source retention) cho từng (phương pháp, N) thành một mặt đánh đổi. Dự đoán: ALDI++/replay thắng hẳn ở N nhỏ; FT thắng trên target nhưng thua trên source ở N lớn.

**Dữ liệu.** Chỉ dùng các kết quả đã chấm một lần trong `runs/local/{campaign,rq5,revision}/scored/` (được theo dõi trong git từ commit `407d60d`), cùng annotation của tập train. Không chấm lại plan nào và không mở nhãn test. Script: `scripts/analyze_instances.py`, `scripts/analyze_seed_spread.py`, `scripts/analyze_retention.py`. Bản sao các bảng và hình nằm ở `docs/results/direction1/`.

**Luật đọc kết quả** nằm trong `.agent/plans/active/2026-11-24-direction-1-analyses.md`. Luật được ghi và commit trước khi chạy ba phân tích này, nhưng sau khi đã biết kết quả từng rung của RQ1–RQ5. Vì vậy **mọi cách đọc ở đây là mô tả, không phải kiểm định xác nhận**. Metric: reliable mAP50:95 (pedestrian, car, traffic light, traffic sign) trên final test; khoảng tin cậy 95%.

## Tóm tắt

| Ý của giảng viên | Kết quả |
|---|---|
| 1. Instance hay ảnh | Mật độ khớp số của giảng viên. Phép đọc chính (cùng bước ảnh, chỉ khác seed) **không phát hiện tác động của instance**: +0,0055 [−0,0017; +0,0143] AP mỗi đơn vị Δln(instance); độ nhạy thấp, nên điều này không loại trừ tác động của mật độ. 0,46 tính so với ceiling full-pool 3 epoch (18.015 ảnh); so với ceiling matched-pool 3 epoch (1.738 ảnh) là 0,75 [0,56; 1,03], gần weather 0,85 (compound không xác định). Thứ tự giữa các class phụ thuộc ceiling: theo ceiling 12 epoch, traffic light mới chậm nhất ở lighting. |
| 2. SD theo N | Compound giảm phù hợp với −1/2 (độ dốc −0,65 [−0,96; −0,34]); weather giảm nhanh hơn (−0,87 [−1,18; −0,56]); lighting chững lại (−0,14 [−0,33; +0,04]) nhưng ở mức rất thấp ngay từ N = 10. Theo luật, floor của cả ba shift **không rõ ràng cao hơn nhiễu huấn luyện**. Floor của lighting (khoảng 0,03 đơn vị recovery) gấp khoảng 4 lần nhiễu huấn luyện gộp, nhưng dữ liệu chưa phân biệt được. SD trên các subset ngẫu nhiên không cho biết active selection đạt được tới đâu. |
| 3. Retention | Khớp với dự đoán (mô tả, không phải kiểm định). Ở N = 10 và 100, ALDI++ và replay **trội hơn** FT ở cả ba shift (cao hơn cả target lẫn retention). Ở N = 1.000 là **đánh đổi**: FT cao hơn trên target (+0,003 đến +0,012 AP), ALDI++/replay giữ source tốt hơn (+0,018 đến +0,050). Retention cao của ALDI++ đến từ việc ôn lại ảnh source (replay giữ source ngang ALDI++), không phải từ adaptation. |

## 1. Mức tăng AP đi theo số instance hay số ảnh

### Mật độ instance trong pool train

| Shift | pedestrian | car | traffic light | traffic sign |
|---|---|---|---|---|
| source (9.804 ảnh) | 1,34 | 11,17 | 2,10 | 3,54 |
| compound (1.738 ảnh) | 0,56 | 8,71 | 3,34 | 2,97 |
| weather (1.985 ảnh) | 1,64 | 9,93 | 3,00 | 3,27 |
| lighting (18.015 ảnh) | 0,68 | 9,18 | 2,96 | 3,25 |

Số instance mỗi ảnh, đếm từ annotation train sau bước làm sạch nhãn chung của pipeline. Hai con số của giảng viên (13.170 / 9.804 = 1,34 và 12.265 / 18.015 = 0,68) khớp chính xác.

### Thiết kế

- Mỗi quan sát là một bước giữa hai rung liền nhau trên ladder của cùng một seed lấy mẫu (69 bước: 3 shift × 3 seed, 6–10 bước mỗi chuỗi; × 4 class = 276 quan sát; dạng log bỏ 1 quan sát có 0 instance). Bước từ source model lên rung đầu không đưa vào hồi quy.
- Biến phụ thuộc là ΔAP của class giữa hai rung. Biến giải thích là Δln(số instance của class trong subset) và Δln(số ảnh). Subset của từng run được dựng lại đúng như lúc train (subset lồng nhau theo seed).
- **Vấn đề nhận diện:** subset lồng nhau giữ mật độ gần như không đổi, nên trong kỳ vọng Δln(instance) ≈ Δln(ảnh). Trên dữ liệu, hai biến chỉ tương quan 0,44 (VIF 1,2): Δln(ảnh) gần như cố định theo bước, còn Δln(instance) dao động do nhiễu lấy mẫu ở các rung nhỏ, nơi có ít instance (nhất là pedestrian). Phần lệch giữa hai biến vì vậy trùng với vị trí trên ladder; vấn đề là gây nhiễu, không phải đa cộng tuyến. Vì vậy phép đọc chính là đặc tả `within`: hiệu ứng cố định cho từng ô shift × class × bước, nên chỉ còn lại khác biệt giữa các seed ở cùng một bước ảnh. Các đặc tả gộp được báo kèm VIF.
- Khoảng tin cậy: bootstrap phân tầng (lấy lại ảnh test, rút lại seed theo cả chuỗi), 1.000 lần.

### Kết quả (dạng log)

| Đặc tả | Quan sát | Hệ số Δln(instance) | Hệ số Δln(ảnh) | R² | VIF | RMSE bỏ-một-shift |
|---|---|---|---|---|---|---|
| images | 275 | – | +0,0242 [+0,0180; +0,0299] | 0,117 | – | 0,0075 |
| instances | 275 | +0,0116 [+0,0093; +0,0141] | – | 0,184 | – | 0,0073 |
| both | 275 | +0,0094 [+0,0049; +0,0128] | +0,0135 [+0,0072; +0,0207] | 0,214 | 1,2 | 0,0072 |
| both+class | 275 | +0,0099 [+0,0056; +0,0132] | +0,0129 [+0,0063; +0,0198] | 0,241 | 1,2 | 0,0071 |
| **within (phép đọc chính)** | 275 | **+0,0055 [−0,0017; +0,0143]** | – | 0,067 (within) | – | – |
| both, pedestrian | 68 | +0,0107 [+0,0056; +0,0148] | +0,0105 [−0,0058; +0,0258] | 0,366 | 1,0 | 0,0072 |
| both, car | 69 | −0,0035 [−0,0357; +0,0165] | +0,0335 [+0,0085; +0,0615] | 0,181 | 3,4 | 0,0078 |
| both, traffic light | 69 | +0,0113 [+0,0011; +0,0257] | +0,0034 [−0,0223; +0,0247] | 0,213 | 1,7 | 0,0068 |
| both, traffic sign | 69 | +0,0024 [−0,0145; +0,0153] | +0,0241 [+0,0074; +0,0469] | 0,162 | 2,3 | 0,0071 |

Dạng thô (ΔAP theo số instance và số ảnh chưa lấy log) có R² within 0,002 và VIF theo class tới hàng nghìn, nên không dùng để đọc (`instance_regression.csv` có đủ cả hai dạng).

### Cách đọc

- **Theo luật đã ghi:** khoảng tin cậy của hệ số `within` phủ 0, nên kết luận là "không phát hiện tác động của instance ở cùng bước ảnh". Cận trên +0,0143 nghĩa là: ở cùng một bước ảnh, gấp đôi số instance của một class (Δln = 0,69) đi kèm tối đa khoảng +0,010 AP cho class đó. Phép đọc này gộp bốn class và chỉ dựa vào khác biệt giữa 3 seed, nên độ nhạy thấp.
- **Mô tả thêm (không có kết luận):** trong các đặc tả gộp, mô hình theo instance có R² 0,18 và RMSE bỏ-một-shift 0,0073; mô hình theo ảnh có 0,12 và 0,0075. Theo từng class, khoảng tin cậy của hệ số Δln(instance) nằm trên 0 ở pedestrian và traffic light, còn khoảng của hệ số Δln(ảnh) nằm trên 0 ở car và traffic sign. Hai hệ số của cùng một class không được so sánh trực tiếp (pedestrian: +0,0107 và +0,0105), nên không nói class nào đi theo biến nào. Car có mặt ở khoảng 99% ảnh và traffic sign ở khoảng 80%, nên với hai class này Δln(instance) bám sát Δln(ảnh) (tương quan 0,84 và 0,75, so với 0,19 ở pedestrian) và hệ số tách riêng của chúng kém tin cậy. Mọi đặc tả gộp đều lẫn với vị trí trên ladder (xem phần thiết kế).
- **Về con số 0,46** (`docs/results/campaign/recovery_points.csv`):
  - 0,46 là recovery của pedestrian ban đêm ở N = 1.000 so với ceiling full-pool 3 epoch, tức model train trên toàn bộ 18.015 ảnh lighting.
  - So với ceiling matched-pool 3 epoch (1.738 ảnh, chuẩn so sánh giữa các shift của RQ4), cùng run đó đạt 0,75 [0,56; 1,03], gần weather 0,85 [0,62; 1,07] (8/1.000 lần bootstrap không xác định). Compound ở ô này không xác định: 120/1.000 lần bootstrap có mẫu số không vượt quá 0,01.
  - Phần lớn khoảng cách về mức (0,46 so với 0,75) vì vậy đến từ ceiling của lighting ở một pool lớn gấp khoảng 10 lần. Nhưng với ceiling matched-pool 3 epoch, pedestrian vẫn thấp nhất trong các class của lighting (car 0,87, traffic light 0,85, traffic sign 0,89; khoảng tin cậy chồng nhau).
  - Thứ tự giữa các class phụ thuộc ceiling. Theo ceiling 12 epoch, pedestrian ban đêm là 0,54 [0,40; 0,69] (full pool) và 0,90 [0,67; 1,20] (matched pool), còn traffic light mới là class chậm nhất ở lighting (0,44 và 0,76).
  - Phép so sánh ceiling không kiểm tra mật độ instance. Phần chênh còn lại giữa lighting 0,75 (0,68 pedestrian mỗi ảnh) và weather 0,85 (1,64 mỗi ảnh) đi đúng chiều giả thuyết mật độ, nhưng khoảng tin cậy chồng nhau, nên không đủ để kết luận.
- **Kết luận cho câu hỏi của giảng viên:** dữ liệu hiện có **chưa đủ** để nói ngưỡng tính theo ảnh là một artefact mật độ của BDD100K, và cũng không loại trừ được điều đó. Hình `instance_recovery.png` vẽ recovery từng class theo cả trục ảnh lẫn trục instance để so sánh trực quan. Recovery trên trục instance vẫn tính so với ceiling full-pool 3 epoch riêng của từng shift, nên so sánh giữa các shift trên trục này mang cùng giới hạn như trên.

## 2. Độ phân tán giữa các seed theo N

SD của reliable mAP giữa 3 seed lấy mẫu ở mỗi rung (ν = 2), với khoảng tin cậy chi-square. Nhiễu huấn luyện đo từ 2 seed của run full-pool E = 3 (cùng ảnh, chỉ khác ngẫu nhiên trong huấn luyện; ν = 1 mỗi shift, ν = 3 khi gộp).

### SD theo rung (×10⁻³ mAP; trong ngoặc: SD tính theo đơn vị recovery E = 3)

| Shift | 10 | 25 | 50 | 100 | 250 | 500 | 1.000 | 2.000 | 5.000 | 10.000 |
|---|---|---|---|---|---|---|---|---|---|---|
| compound | 5,7 (0,150) | 4,4 (0,115) | 2,5 (0,065) | 3,0 (0,080) | 0,5 (0,013) | 0,5 (0,014) | 0,2 (0,006) | | | |
| weather | 10,6 (0,687) | 8,9 (0,574) | 5,0 (0,322) | 5,1 (0,332) | 1,3 (0,084) | 0,3 (0,017) | 0,2 (0,014) | | | |
| lighting | 1,2 (0,023) | 1,3 (0,024) | 3,3 (0,064) | 1,7 (0,033) | 2,0 (0,038) | 1,0 (0,018) | 0,6 (0,012) | 1,0 (0,019) | 0,8 (0,015) | 0,4 (0,007) |

Nhiễu huấn luyện gộp: 0,3 [0,2; 1,2] ×10⁻³.

### Độ dốc ln SD theo ln N

| Shift | Số rung | Đã hiệu chỉnh pool hữu hạn (chính) | Chưa hiệu chỉnh | χ²/df | Phủ −1/2 | Dưới 0 |
|---|---|---|---|---|---|---|
| compound | 7 | −0,65 [−0,96; −0,34] | −0,73 [−1,04; −0,42] | 0,39 | có | có |
| weather | 7 | −0,87 [−1,18; −0,56] | −0,93 [−1,24; −0,62] | 0,68 | không (dốc hơn) | có |
| lighting | 10 | −0,14 [−0,33; +0,04] | −0,18 [−0,37; −0,00] | 0,45 | không | không |
| gộp | 24 | −0,40 [−0,54; −0,26] | −0,45 [−0,59; −0,31] | 1,40 | có | có |

- **Hiệu chỉnh pool hữu hạn:** các subset được rút không hoàn lại từ một pool hữu hạn, nên ngay cả khi chọn hoàn toàn ngẫu nhiên, SD vẫn giảm dốc hơn −1/2 ở các rung chiếm phần lớn pool. Sau khi hiệu chỉnh hệ số pool hữu hạn, chọn ngẫu nhiên thuần cho đúng −1/2. Độ dốc chưa hiệu chỉnh (đúng dạng 1/√N trong góp ý) được giữ để tham khảo.
- **Theo class** (đã hiệu chỉnh, `seed_spread_fit.csv`): mọi class ở compound và weather đều có độ dốc dưới 0 và phủ −1/2. Ở lighting: pedestrian −0,39 [−0,57; −0,21] và traffic light −0,37 [−0,55; −0,19] phủ −1/2 (traffic light có χ²/df 3,81, đường lũy thừa không khớp tốt); traffic sign −0,31 [−0,49; −0,13] nông hơn −1/2; car gần phẳng (−0,09 [−0,27; +0,10]).

### Mô hình phương sai có floor (từng shift)

s² = σ²_floor + c · (1/N − 1/M), với M là kích thước pool. Khoảng tin cậy lấy từ bootstrap tham số (1.000 lần). "Rõ ràng cao hơn nhiễu" nghĩa là cận dưới của σ_floor cao hơn cận trên của nhiễu huấn luyện gộp.

| Shift, class | σ_floor (×10⁻³) | Nhiễu huấn luyện gộp (×10⁻³) | Rõ ràng cao hơn nhiễu |
|---|---|---|---|
| compound, reliable | 0,0 [0,0; 2,1] | 0,3 [0,2; 1,2] | không |
| weather, reliable | 0,0 [0,0; 4,1] | 0,3 [0,2; 1,2] | không |
| lighting, reliable | 1,5 [0,9; 2,0] | 0,3 [0,2; 1,2] | không |
| compound, pedestrian | 2,2 [0,0; 7,4] | 0,9 [0,5; 3,3] | không |
| weather, pedestrian | 2,8 [0,0; 6,6] | 0,9 [0,5; 3,3] | không |
| lighting, pedestrian | 4,5 [2,0; 7,7] | 0,9 [0,5; 3,3] | không |
| compound, traffic light | 4,1 [2,2; 5,9] | 0,5 [0,3; 1,9] | có (xem lưu ý) |

Trong 15 hàng (3 shift × reliable và 4 class, `seed_spread_fit.csv`), chỉ compound traffic light được gắn cờ. Ba hàng có c ≤ 0, tức mô hình phương sai không tìm được thành phần giảm theo 1/N, dù độ dốc lũy thừa của chính các hàng này vẫn dưới 0: compound traffic light (c ≈ 0, χ²/df của đường lũy thừa 2,39), compound traffic sign (khoảng percentile của σ_floor còn không chứa ước lượng điểm) và lighting traffic light. Ở đó mô hình phương sai không mô tả được dữ liệu, nên không đọc các hàng này, kể cả cờ của compound traffic light.

### Cách đọc

- **Compound:** độ dốc phủ −1/2 và nằm dưới 0, phù hợp với việc lấy trung bình trên các ảnh rút ngẫu nhiên. **Weather:** giảm nhanh hơn −1/2. Ở cả hai shift này, phần phân tán còn lại ở N = 1.000 (0,2 ×10⁻³) nằm ở mức nhiễu huấn luyện đo được. **Độ dốc gộp** (−0,40) là trung bình của các shift có độ dốc không tương thích (weather và lighting), nên không đọc riêng.
- **Lighting:** gần phẳng, nhưng ở mức rất thấp ngay từ N = 10 (SD 1,2 ×10⁻³, tức 0,023 đơn vị recovery; so với 5,7 ×10⁻³ ở compound và 10,6 ×10⁻³ ở weather). Floor 1,5 [0,9; 2,0] ×10⁻³ (khoảng 0,03 đơn vị recovery) gấp khoảng 4 lần ước lượng điểm của nhiễu huấn luyện gộp, nhưng khoảng tin cậy chồng lên cận trên 1,2 ×10⁻³, nên theo luật là "không rõ ràng cao hơn nhiễu". Ở lighting N = 10, SD giữa 3 seed là 0,023 đơn vị recovery (khoảng 95%: 0,012–0,143); ở rung này FT sụp dưới source model ở cả ba seed (recovery trung bình −1,17).
- **Với câu hỏi "vì sao không dùng active learning":**
  - Theo cách đọc độ dốc của giảng viên, compound và weather cho thấy chọn ngẫu nhiên gần tối ưu trong số các subset ngẫu nhiên. Lighting chững lại, nhưng ở mức rất thấp (floor khoảng 0,03 đơn vị recovery), và theo luật floor của plan thì không rõ ràng cao hơn nhiễu huấn luyện. Nếu có chỗ cho active selection ở lighting thì chỗ đó rất nhỏ.
  - Ở N nhỏ, compound và weather phân tán mạnh (SD 0,15 và 0,69 đơn vị recovery ở N = 10). Ở đó việc chọn ảnh có thể quan trọng, hoặc nhiễu huấn luyện ở N nhỏ lớn hơn; hai phần này chưa tách được (xem lưu ý về giả định).
- **Giới hạn:** SD trên các subset ngẫu nhiên không chặn được mức active selection có thể đạt; một subset được chọn có chủ đích có thể nằm ngoài phân bố của các subset ngẫu nhiên. Muốn trả lời trực tiếp thì cần một run có active selection.

### Lưu ý

- **Khoảng tin cậy là xấp xỉ.** Các rung của một seed là subset lồng nhau, nên SD ở các rung kề nhau tương quan. Khoảng tin cậy của độ dốc coi chúng là độc lập (trên dữ liệu tổng hợp lồng nhau độ phủ khoảng 89%, không phải 95%). Khoảng tin cậy của mô hình phương sai cũng vậy.
- **Nhiễu huấn luyện đo từ 2 seed** cho mỗi shift (ν = 1), nên rất rộng; bản gộp (ν = 3) vẫn rộng.
- **Giả định nhiễu huấn luyện không phụ thuộc N** chưa được kiểm tra. Nhiễu chỉ được đo ở run full-pool. Ở một rung, seed đổi cả subset lẫn phần ngẫu nhiên của huấn luyện. Các rung nhỏ chạy 500 update (khoảng 100 epoch ở N = 10), nên nhiễu huấn luyện ở đó có thể lớn hơn nhiều, và SD ở N nhỏ gồm cả phần đó, không chỉ phần do chọn ảnh. Một độ dốc dốc hơn −1/2 như ở weather cũng có thể xuất hiện nếu nhiễu huấn luyện giảm theo N. Mọi so sánh floor với nhiễu dựa trên giả định này.
- **Kích thước tập test khác nhau:** final test của compound có 756 ảnh, weather 933, lighting 4.869. SD giữa các seed gồm cả cách khác biệt giữa các model hiện ra trên một tập test hữu hạn. Đây có thể là một phần lý do lighting phân tán ít hơn; điều này chưa được kiểm tra.

## 3. Source retention như một trục thứ hai

- **Trục target:** recovery so với mốc FT toàn pool 3 epoch, như các bảng RQ5 (mốc tham chiếu, không phải ceiling mAP_c 12 epoch).
- **Trục source:** retention = mAP trên source test / mAP của source model trên source test (0,3422).
- **Phương pháp:** FT (ladder và full pool, E = 3), FT full pool 12 epoch, ALDI++ (`adapt`), replay, và hai nhánh ở N = 1.000 với số lượt ảnh có nhãn cân bằng (`adapt_matched`, `replay_matched`, 3.000 update).
- **Cặp so sánh:** ghép theo seed chung, cùng các lần lấy lại ảnh test cho cả hai trục. "Trội hơn": khoảng tin cậy của cả hai hiệu đều trên 0. "Đánh đổi": hai hiệu ngược dấu và cả hai khoảng tin cậy đều không chứa 0.

### Các điểm chính

| Shift | Phương pháp | N | Recovery (mốc FT 3 epoch) | Retention |
|---|---|---|---|---|
| compound | FT | 10 | −1,33 [−1,60; −1,09] | 0,840 [0,830; 0,850] |
| compound | FT | 100 | +0,09 [−0,01; +0,21] | 0,937 [0,930; 0,945] |
| compound | FT | 1.000 | +0,87 [+0,82; +0,91] | 0,954 [0,949; 0,960] |
| compound | FT | 1.738 | +1,00 | 0,947 [0,941; 0,953] |
| compound | FT 12 epoch | 1.738 | +1,01 [+0,94; +1,10] | 0,905 [0,900; 0,911] |
| compound | adapt | 10 / 100 / 1.000 | +0,29 / +0,51 / +0,56 | 0,991 / 0,995 / 0,996 |
| compound | replay | 10 / 100 / 1.000 | +0,02 / +0,41 / +0,54 | 0,991 / 0,999 / 1,001 |
| weather | FT | 10 | −3,51 [−4,76; −2,66] | 0,831 [0,791; 0,858] |
| weather | FT | 100 | −0,42 [−0,85; −0,11] | 0,949 [0,936; 0,960] |
| weather | FT | 1.000 | +0,77 [+0,65; +0,86] | 0,982 [0,977; 0,987] |
| weather | FT | 1.985 | +1,00 | 0,984 [0,980; 0,988] |
| weather | FT 12 epoch | 1.985 | +0,71 [+0,55; +0,84] | 0,957 [0,952; 0,963] |
| weather | adapt | 10 / 100 / 1.000 | +0,20 / +0,46 / +0,56 | 0,993 / 0,997 / 1,000 |
| weather | replay | 10 / 100 / 1.000 | −0,03 / +0,37 / +0,55 | 0,991 / 1,000 / 1,005 |
| lighting | FT | 10 | −1,17 [−1,23; −1,10] | 0,839 [0,831; 0,848] |
| lighting | FT | 100 | +0,01 [−0,03; +0,05] | 0,944 [0,937; 0,950] |
| lighting | FT | 1.000 | +0,55 [+0,53; +0,57] | 0,964 [0,959; 0,969] |
| lighting | FT | 18.015 | +1,00 | 0,924 [0,918; 0,930] |
| lighting | FT 12 epoch | 18.015 | +1,06 [+1,03; +1,08] | 0,875 [0,867; 0,883] |
| lighting | adapt | 10 / 100 / 1.000 | +0,18 / +0,31 / +0,34 | 0,989 / 0,995 / 0,997 |
| lighting | replay | 10 / 100 / 1.000 | +0,01 / +0,27 / +0,34 | 0,990 / 0,998 / 1,004 |

Khoảng tin cậy của từng điểm nằm trong `retention_points.csv`. Replay ở N = 1.000 chỉ có seed 0. Với weather, recovery so với ceiling 12 epoch không xác định: 33,8% số lần bootstrap có mẫu số không vượt quá 0,01 (kỳ vọng chính xác 33,3%).

### So sánh từng cặp với FT

| Shift | N | So sánh | Seed | Δ target mAP | Δ retention | Kết luận |
|---|---|---|---|---|---|---|
| compound | 10 | adapt − FT | 0 1 2 | +0,0613 [+0,0544; +0,0677] | +0,150 [+0,140; +0,160] | trội hơn |
| compound | 100 | adapt − FT | 0 1 2 | +0,0157 [+0,0118; +0,0195] | +0,058 [+0,050; +0,064] | trội hơn |
| compound | 1.000 | adapt − FT | 0 1 2 | −0,0117 [−0,0142; −0,0090] | +0,042 [+0,037; +0,047] | đánh đổi |
| compound | 10 | replay − FT | 0 1 2 | +0,0512 [+0,0449; +0,0570] | +0,151 [+0,140; +0,162] | trội hơn |
| compound | 100 | replay − FT | 0 1 2 | +0,0119 [+0,0081; +0,0155] | +0,062 [+0,055; +0,069] | trội hơn |
| compound | 1.000 | replay − FT | 0 | −0,0125 [−0,0152; −0,0090] | +0,050 [+0,046; +0,056] | đánh đổi |
| compound | 1.000 | adapt_matched − FT | 0 1 2 | −0,0074 [−0,0098; −0,0046] | +0,034 [+0,029; +0,041] | đánh đổi |
| compound | 1.000 | replay_matched − FT | 0 1 2 | −0,0058 [−0,0081; −0,0035] | +0,045 [+0,039; +0,051] | đánh đổi |
| weather | 10 | adapt − FT | 0 1 2 | +0,0573 [+0,0495; +0,0670] | +0,162 [+0,135; +0,201] | trội hơn |
| weather | 100 | adapt − FT | 0 1 2 | +0,0137 [+0,0091; +0,0188] | +0,049 [+0,039; +0,060] | trội hơn |
| weather | 1.000 | adapt − FT | 0 1 2 | −0,0032 [−0,0052; −0,0011] | +0,018 [+0,013; +0,022] | đánh đổi |
| weather | 10 | replay − FT | 0 1 2 | +0,0538 [+0,0456; +0,0632] | +0,160 [+0,134; +0,199] | trội hơn |
| weather | 100 | replay − FT | 0 1 2 | +0,0123 [+0,0090; +0,0165] | +0,052 [+0,043; +0,063] | trội hơn |
| weather | 1.000 | replay − FT | 0 | −0,0033 [−0,0052; −0,0004] | +0,026 [+0,023; +0,030] | đánh đổi |
| weather | 1.000 | adapt_matched − FT | 0 1 2 | −0,0021 [−0,0040; +0,0002] | +0,011 [+0,006; +0,016] | không rõ thứ tự |
| weather | 1.000 | replay_matched − FT | 0 1 2 | −0,0015 [−0,0032; +0,0006] | +0,020 [+0,016; +0,025] | không rõ thứ tự |
| lighting | 10 | adapt − FT | 0 1 2 | +0,0704 [+0,0674; +0,0732] | +0,150 [+0,142; +0,156] | trội hơn |
| lighting | 100 | adapt − FT | 0 1 2 | +0,0154 [+0,0136; +0,0174] | +0,051 [+0,045; +0,057] | trội hơn |
| lighting | 1.000 | adapt − FT | 0 1 2 | −0,0111 [−0,0122; −0,0094] | +0,033 [+0,028; +0,037] | đánh đổi |
| lighting | 5.000 | adapt − FT | 0 1 2 | −0,0096 [−0,0108; −0,0080] | +0,027 [+0,022; +0,031] | đánh đổi |
| lighting | 10 | replay − FT | 0 1 2 | +0,0616 [+0,0599; +0,0632] | +0,151 [+0,143; +0,159] | trội hơn |
| lighting | 100 | replay − FT | 0 1 2 | +0,0133 [+0,0109; +0,0151] | +0,054 [+0,048; +0,060] | trội hơn |
| lighting | 1.000 | replay − FT | 0 | −0,0103 [−0,0114; −0,0089] | +0,039 [+0,035; +0,044] | đánh đổi |
| lighting | 1.000 | adapt_matched − FT | 0 1 2 | −0,0076 [−0,0089; −0,0057] | +0,025 [+0,020; +0,030] | đánh đổi |
| lighting | 1.000 | replay_matched − FT | 0 1 2 | −0,0064 [−0,0077; −0,0050] | +0,036 [+0,031; +0,040] | đánh đổi |

So sánh ở N = 0 (ALDI++ và replay so với source model) nằm trong `retention_paired.csv`. ALDI++ không học nhãn target tăng target +0,006 đến +0,011 AP mà không đổi retention một cách rõ ràng.

### Pareto front (gain trên target, retention)

Front dựa trên ước lượng điểm, chỉ mang tính mô tả; replay ở N = 1.000 chỉ có seed 0.


- **compound:** FT ở N = 1.000 và 1.738, FT 12 epoch, replay ở N = 0 và 1.000, replay_matched.
- **weather:** FT ở N = 1.985, replay ở N = 1.000, replay_matched.
- **lighting:** FT ở N = 2.000, 5.000, 10.000 và 18.015, FT 12 epoch, adapt ở N = 5.000, replay ở N = 0 và 1.000, replay_matched.

### Cách đọc

- **Kết quả khớp với dự đoán của giảng viên, ở mức mô tả.** Đây không phải kiểm định: chiều target đã biết từ các bảng RQ5 trước khi luật được viết. Ở N = 10 và 100, ALDI++ và replay trội hơn FT ở cả ba shift. Ở N = 1.000 (và lighting N = 5.000), FT cao hơn trên target nhưng mất source nhiều hơn. Khi cân bằng số lượt ảnh có nhãn, compound và lighting vẫn là đánh đổi; ở weather, khoảng tin cậy của Δ target phủ 0 nên không rõ thứ tự.
- **"Trội hơn" ở N = 10 chủ yếu là FT sụp:** FT với 10 ảnh kém source model trên cả hai trục (recovery −1,17 đến −3,51, retention khoảng 0,84). ALDI++ và replay tránh được cú sụp đó.
- **Retention cao đến từ việc ôn lại source, không phải adaptation.** Mỗi update của ALDI++ và replay có 1 ảnh source có nhãn, còn FT chỉ học ảnh target. Replay giữ source ít nhất ngang ALDI++ ở mọi N (chênh tối đa 0,002 ở N = 10, cao hơn ở N = 100 và 1.000). Vì vậy trục retention đo đánh đổi của từng công thức huấn luyện như đã chạy, không phải riêng phần adaptation. Điều này khớp với phép phân tách của RQ5.
- **FT 12 epoch** cho target cao nhất ở lighting (1,06 [1,03; 1,08]); ở compound nó ngang FT toàn pool 3 epoch (+0,0005 AP, khoảng tin cậy phủ 0). Retention của nó thấp nhất trong các điểm có recovery dương (0,905 và 0,875).

## Tái lập

Chạy từ `scripts/` (chỉ dùng CPU):

```
python analyze_instances.py --scored ../runs/local/campaign/scored --dataset-root ../data/bdd100k_revised --output-dir ../runs/local/direction1/instances
python analyze_seed_spread.py --scored ../runs/local/campaign/scored --output-dir ../runs/local/direction1/seed_spread
python analyze_retention.py --ft-scored ../runs/local/campaign/scored --da-scored ../runs/local/rq5/scored --revision-scored ../runs/local/revision/scored --output-dir ../runs/local/direction1/retention
```

Seed mặc định: bootstrap `--seed 0` cho cả ba script. Bảng và hình ở `docs/results/direction1/` là bản sao byte-identical của các thư mục output trên.
