# Kết quả RQ5: ALDI++ so với fine-tune thường (2026-11-07; cập nhật 2026-11-23)

**Câu hỏi:** một phương pháp domain adaptation thật sự (ALDI++, Kay et al., CVPR 2024) có giúp tiết kiệm nhãn target so với fine-tune thường (FT) không? Nếu có, tới N bao nhiêu thì FT đuổi kịp?

**Grid:** 46 run, chạy trên máy local (RTX 3060), xong lúc 2026-11-07 07:11. Luật phân tích được ghi trước mọi kết quả (`.agent/DECISIONS.md`, 2026-11-03). Pilot trên lighting val chốt α = 0,9996 và T = 2.000 update (DECISIONS 2026-11-06).

**Mỗi plan đóng băng được chấm đúng một lần:**

- Grid 46 run: plan `runs/local/rq5/final_plan.json`, sha256 `3df3d788…b4b7`, chấm ngày 2026-11-07.
- 30 run bổ sung: plan `runs/local/revision/final_plan.json` (42 bộ prediction, trong đó 30 bộ của RQ5), sha256 `de135bcc…0282`, chấm ngày 2026-11-23.
- Bootstrap dùng cùng seed (20261030) và cùng danh sách ảnh test với campaign FT (RQ1–RQ4). Vì vậy mọi so sánh DA − FT được ghép cặp trên cùng các lần lấy mẫu lại ảnh test.

Số liệu gốc của mọi bảng và hình nằm trong `runs/local/rq5/analysis_rev2/` (`rq5_*.csv`, `summary.json`, hình `rq5_<domain>.png`). Thư mục `analysis/` là bản 2026-11-07: hình của nó còn ghi "FT full-pool ceiling" và không có các run bổ sung, nên không dùng cho báo cáo. Script phân tích: `scripts/analyze_rq5.py`.

## Cập nhật theo nhận xét của giảng viên (2026-11-21 đến 2026-11-23)

Nhận xét nằm ở `docs/review_2026-11-21_lecturer_vi.md`. Các luật đọc kết quả được ghi trong `.agent/DECISIONS.md` (mục 2026-11-21 và hai phần bổ sung) trước khi các run mới được chấm.

- **Các luật này mang tính mô tả, không phải kiểm định xác nhận.** Chúng được viết sau khi đã thấy kết quả của grid gốc, kể cả tỉ lệ ở seed 0 (0,82–0,95 ở N = 10 và 0,78–0,86 ở N = 100). Dữ liệu mới so với luật chỉ là seed 1 và 2 của nhánh replay và các run cân bằng.
- **Kết quả chính của ghi chú này là phép phân tách bằng nhánh đối chứng `replay`**, không phải vị trí crossover. Luận điểm: ở ngân sách nhãn nhỏ, phần lớn lợi thế của ALDI++ so với fine-tune không phải là adaptation.
- **Vị trí của luận điểm trong nghiên cứu** (theo nhận xét của giảng viên): nó nối trực tiếp với luận điểm của ALDI (Kay et al., CVPR 2024) rằng mức lợi thế được báo cáo của DAOD phụ thuộc vào lựa chọn triển khai và cách dựng baseline. Các đường cong hiệu quả dữ liệu (RQ1–RQ4) là bối cảnh để đo luận điểm; các ngưỡng N_BE, N75, N90 là kết quả thực tiễn thứ cấp. Weather là lần lặp lại thứ ba của phép phân tách, không phải phép đo ngưỡng thứ ba.
- **30 run bổ sung** đã chạy xong và được chấm đúng một lần qua plan đóng băng của đợt bổ sung: replay thêm seed 1 và 2 ở N = 10 và 100; ALDI++ và replay ở N = 1.000 với 3.000 update, 3 seed.
- **Luận điểm chính đủ điều kiện để nêu:** với 3 seed cho nhánh replay, luật đã ghi được thỏa ở cả N = 10 và N = 100. Luận điểm là một phát biểu về lợi thế **so với fine-tune**. Ở N = 10 nó phải đi kèm cách đọc so với mô hình nguồn: replay gần như không được gì và ALDI++ chỉ hơn +0,003 đến +0,011 AP, tức 10 ảnh có nhãn không thêm gì.
- **Kết luận "fine-tune dẫn trước ALDI++ từ N = 1.000"** đứng vững ở lighting và compound sau khi cân bằng số lượt ảnh có nhãn, và được rút lại ở weather.

Các bảng theo luật 2026-11-03 (Δ, N_eq, phân tách seed 0, quên source) giữ nguyên từng con số so với bản 2026-11-07. Các bảng trong ghi chú này được sinh trực tiếp từ file CSV.

## Thiết lập tóm tắt

| Nhánh | Luồng có nhãn (2 ảnh/step) | Luồng không nhãn (2 ảnh/step) | Số update |
|---|---|---|---|
| FT (campaign) | 2 ảnh target có nhãn | không có | max(500, 3 epoch) |
| `adapt` N=0 (UDA) | 2 ảnh source | toàn pool target, nhãn bị ẩn | 2.000 |
| `adapt` N>0 | 1 ảnh source + 1 ảnh trong N ảnh target có nhãn | toàn pool target | max(2.000, 2N) |
| `replay` (đối chứng) | như `adapt` | không có, không distill | như `adapt` |

- **Khởi tạo và model giữ lại:** mọi nhánh khởi tạo từ cùng source model và giữ model cuối. Với `adapt`/`replay`, model cuối là EMA teacher.
- **Subset target:** `adapt` dùng đúng các subset target lồng nhau như FT, nên Δ được ghép theo seed lấy mẫu.
- **Δ(N)** = trung bình theo seed của [mAP_DA(N, s) − mAP_FT(N, s)]. Với N = 0, so với zero-shot mAP(0).
- **Lợi thế có ý nghĩa:** Δ ≥ 0,005 và cận dưới CI 95% > 0.
- **Metric:** reliable mAP50:95 trên final test.

## Δ(N) = ALDI++ − FT (theo luật 2026-11-03)

| Shift | N=0 (UDA) | N=10 | N=100 | N=1000 | N=5000 |
|---|---:|---:|---:|---:|---:|
| lighting | **+0,009** (0,008; 0,011) | **+0,070** (0,068; 0,073) | **+0,015** (0,014; 0,017) | −0,011 (−0,012; −0,009) | −0,010 (−0,011; −0,008) |
| weather | **+0,006** (0,003; 0,007) | **+0,057** (0,050; 0,066) | **+0,014** (0,009; 0,019) | −0,003 (−0,005; −0,001) |  |
| compound | **+0,011** (0,008; 0,014) | **+0,061** (0,054; 0,068) | **+0,016** (0,012; 0,019) | −0,012 (−0,014; −0,009) |  |

In đậm là lợi thế có ý nghĩa theo luật. Weather N=0 vừa qua ngưỡng (0,0055 ≥ 0,005).

**Crossover N× theo luật đã đăng ký: trong khoảng (100, 1.000] ở cả 3 shift.**

- Từ N=1000 trở lên, ALDI++ không còn lợi thế và thấp hơn FT với CI hoàn toàn dưới 0. Lighting N=5000 cũng vậy (−0,010).
- Ở N=1000, FT thấy khoảng 3.000 lượt ảnh target có nhãn còn ALDI++ chỉ khoảng 2.000 (mục Hạn chế, điểm 1), nên con số trên chưa đủ để nói fine-tune dẫn trước.
- Thí nghiệm cân bằng số lượt ảnh đã được chạy (mục "Thí nghiệm cân bằng" bên dưới). Sau khi cân bằng, FT vẫn dẫn trước ở lighting và compound; ở weather không còn khác biệt có ý nghĩa.

**mAP tuyệt đối** (trung bình seed):

| Shift | zero-shot | DA N=0 | DA N=10 | DA N=100 | DA N=1000 | FT N=100 | FT N=1000 | FT toàn pool, 3 epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lighting | 0,244 | 0,253 | 0,253 | 0,260 | 0,262 | 0,244 | 0,273 | 0,296 |
| weather | 0,320 | 0,326 | 0,323 | 0,327 | 0,329 | 0,314 | 0,332 | 0,336 |
| compound | 0,219 | 0,230 | 0,230 | 0,239 | 0,240 | 0,223 | 0,252 | 0,257 |

Cột cuối là trung bình seed của FT trên toàn pool với 3 epoch (mốc của các luật 2026-11-03), không phải mAP_c 12 epoch (0,299 / 0,331 / 0,258) trong `docs/results_local_rq1_rq4_vi.md`.

Lighting N=5000: DA 0,276, FT 0,286.

Đường DA gần như phẳng: từ N=0 tới N=1000 chỉ tăng thêm khoảng 0,003–0,010 AP. Đường FT dốc hơn nhiều sau khi qua vùng overfit ở N nhỏ.

## Ngân sách nhãn tương đương N_eq

N_eq là số ảnh FT cần để đạt cùng mAP với DA. Nội suy tuyến tính theo log N, lấy điểm cắt bền vững (lần cắt lên cuối cùng của đường FT). CI 95% lấy từ bootstrap. Chi tiết "điểm cắt bền vững" không có trong luật 2026-11-03; nó được thêm khi phân tích, vì đường FT xuống dưới mAP(0) ở N nhỏ.

| Shift | DA N=0 | DA N=10 | DA N=100 | DA N=1000 | DA N=5000 |
|---|---|---|---|---|---|
| lighting | 160 (142–184) | 162 (145–183) | 236 (215–287) | 272 (235–344) | 1.556 (1.343–1.894) |
| weather | 349 (240–511) | 242 (184–379) | 451 (313–672) | 575 (413–834) |  |
| compound | 153 (123–182) | 152 (121–180) | 242 (209–332) | 300 (233–410) |  |

**Cách đọc:**

- **Adapt không nhãn (UDA)** đáng giá khoảng **150–350 ảnh có nhãn** khi dùng FT với schedule này. Theo đơn vị recovery, với mẫu số là FT toàn pool 3 epoch trừ zero-shot: 0,17 (lighting), 0,36 (weather), 0,29 (compound). Với mẫu số theo ceiling 12 epoch thì weather không xác định.
- DA với 10 ảnh có nhãn **không hơn** DA không nhãn. 10 ảnh đó gần như không thêm gì.
- Với số update đã đăng ký, DA với 1.000 ảnh có nhãn chỉ tương đương FT với 270–575 ảnh. Con số này lẫn với chênh lệch số lượt ảnh có nhãn (mục Hạn chế, điểm 1), nên chưa thể đọc là ALDI++ dùng nhãn kém hiệu quả hơn FT. N_eq không được tính lại cho thí nghiệm cân bằng. Ở weather, sau khi cân bằng không còn khác biệt có ý nghĩa với FT ở N = 1.000, nên con số 575 (413–834) không được đọc là ALDI++ thua FT.

## Phân tách đóng góp (seed 0)

Có hai nguồn đóng góp:

- **adapt − replay:** giá trị của distillation trên ảnh target không nhãn.
- **replay − FT:** giá trị của việc trộn ảnh source, augmentation mạnh và EMA.

| Shift | N | adapt − replay | replay − FT |
|---|---:|---:|---:|
| lighting | 0 | +0,006 | +0,004 |
|  | 10 | +0,009 | +0,062 |
|  | 100 | +0,002 | +0,012 |
|  | 1000 | 0,000 (CI chứa 0) | −0,010 |
| weather | 0 | +0,003 | +0,003 |
|  | 10 | +0,004 | +0,064 |
|  | 100 | +0,003 | +0,016 |
|  | 1000 | 0,000 (CI chứa 0) | −0,003 |
| compound | 0 | +0,009 | +0,002 (CI chứa 0) |
|  | 10 | +0,010 | +0,046 |
|  | 100 | +0,004 | +0,013 |
|  | 1000 | 0,000 (CI chứa 0) | −0,012 |

Ở N = 0, "replay" là source model train tiếp trên source với augmentation mạnh và EMA. Vì vậy replay − FT ở đây là replay − zero-shot.

### Tỉ lệ đóng góp của replay

Tỉ lệ = (replay − FT) / (adapt − FT): phần lợi thế của ALDI++ so với FT mà một nhánh **không có adaptation** tái tạo được (`rq5_replay_share.csv`).

- Ở N = 10 và N = 100, cả ba nhánh có 3 seed lấy mẫu. Ở N = 0 và N = 1000, nhánh replay với số update gốc chỉ có seed 0.
- CI 95% lấy từ bootstrap: ảnh test được lấy mẫu lại chung cho ba nhánh, và seed được lấy lại theo bộ ba.
- Tỉ lệ không xác định khi |adapt − FT| < 0,005, và không được trích dẫn khi quá 5% số lần bootstrap rơi vào trường hợp đó.

| N | lighting | weather | compound |
|---:|---:|---:|---:|
| 10 (3 seed) | 0,87 (0,85–0,90) | 0,94 (0,91–0,96) | 0,83 (0,80–0,86) |
| 100 (3 seed) | 0,86 (0,78–0,90) | 0,90 (0,81–1,04) | 0,76 (0,62–0,85) |
| 1000 (seed 0; phần thua) | 1,05 (0,95–1,16) | không xác định (926/1.000) | 0,97 (0,82–1,15) |
| 0 (seed 0; không có nhãn) | 0,39 (0,28–0,45) | không xác định (287/1.000) | 0,17 (−0,04 đến 0,35); 1/1.000 không xác định |

- **Ở N = 10 và N = 100, replay tái tạo 76–94% lợi thế ở cả ba shift.** Mọi cận dưới CI đều trên 0,5, và ALDI++ thật sự hơn FT ở cả sáu ô.
- Ở N = 1000, adapt − FT là số âm, nên tỉ lệ nói rằng phần **thua** nằm trọn ở replay. Weather không xác định vì |adapt − FT| chỉ 0,003.
- Trong ngoặc sau "không xác định" là số lần bootstrap mà tỉ lệ không xác định, trên 1.000 lần.
- Ở N = 0 thì ngược lại với lighting và compound: phần lớn lợi ích đến từ distillation. Weather không xác định vì 287 trong 1.000 lần bootstrap có lợi ích dưới 0,005.

### Cùng phép tách, nhưng so với mô hình nguồn

FT ở N ≤ 50 thấp hơn cả mô hình nguồn, nên tỉ lệ trên đọc khác đi khi mốc so sánh là "không fine-tune". Bảng dưới là mAP trừ zero-shot. Cột FT là trung bình 3 seed (`rq5_vs_zero_shot.csv`); replay, ALDI++ và tỉ lệ lấy từ `rq5_replay_share_vs_source.csv` (3 seed ở N = 10 và 100, seed 0 ở N = 1000).

| Shift | N | FT | replay | ALDI++ | replay / ALDI++ (CI 95%) |
|---|---:|---:|---:|---:|---:|
| lighting | 10 | −0,061 | 0,000 | +0,009 | 0,05 (−0,09 đến 0,21) |
|  | 100 | +0,001 | +0,014 | +0,016 | 0,87 (0,80–0,91) |
|  | 1000 | +0,029 | +0,018 | +0,018 | 0,97 (0,92–1,03) |
| weather | 10 | −0,054 | 0,000 | +0,003 | không xác định (908/1.000) |
|  | 100 | −0,007 | +0,006 | +0,007 | 0,80 (0,57–1,07); 29/1.000 không xác định |
|  | 1000 | +0,012 | +0,008 | +0,008 | 1,00 (0,86–1,24); 1/1.000 không xác định |
| compound | 10 | −0,050 | +0,001 | +0,011 | 0,07 (−0,22 đến 0,29) |
|  | 100 | +0,003 | +0,015 | +0,019 | 0,80 (0,72–0,88) |
|  | 1000 | +0,033 | +0,021 | +0,020 | 1,02 (0,92–1,12) |

**Bốn kết luận từ ba bảng:**

1. **Ở N = 10, lợi thế lớn của ALDI++ so với FT là thiệt hại mà FT tự gây ra, không phải adaptation.**
   - FT với 10 ảnh và 500 update làm model tệ hơn mô hình nguồn 0,05–0,06 AP. Replay chỉ cần không gây ra thiệt hại đó là đã tái tạo 83–94% lợi thế.
   - So với mô hình nguồn, replay không được gì (0,000; 0,000; +0,001; CI đều chứa 0). ALDI++ được +0,003 đến +0,011 AP, không hơn mức của nó khi không có nhãn nào (+0,006 đến +0,011 ở N = 0).
   - Nói cách khác, 10 ảnh có nhãn không thêm gì. Phần lợi ích nhỏ còn lại là của distillation trên ảnh không nhãn.
2. **Ở N = 100, hai mốc so sánh cho cùng một kết luận, và đây là bằng chứng sạch nhất.** FT lúc này xấp xỉ mô hình nguồn ở lighting (+0,001) và compound (+0,003), CI của cả hai chứa 0. Ở weather FT vẫn thấp hơn mô hình nguồn 0,007 AP (CI −0,012 đến −0,002), nên ở weather một phần lợi thế so với FT vẫn là thiệt hại của FT. Replay tái tạo 76–90% lợi thế so với FT, và 80–87% lợi ích so với mô hình nguồn.
3. **Distillation trên ảnh không nhãn có giá trị thật nhưng nhỏ:** khoảng 0,003–0,010 AP khi N ≤ 10, giảm còn ≤ 0,004 ở N=100 (ở weather, trung bình 3 seed là +0,001 với CI (0,000; +0,003) chứa 0, theo cột distill_part của `rq5_replay_share.csv`; riêng seed 0 là +0,003 với CI không chứa 0) và về 0 ở N=1000.
4. **Phần thua ở N=1000 không do distillation.** Replay thua FT gần bằng mức đó, và so với mô hình nguồn thì replay bằng ALDI++ (tỉ lệ ≈ 1). Thí nghiệm cân bằng bên dưới cho cùng kết luận.

**Trạng thái của luận điểm chính.** Luật được ghi ngày 2026-11-21, trước khi seed 1 và 2 của nhánh replay được chấm, nhưng sau khi đã thấy tỉ lệ của seed 0; phần xác nhận độc lập vì vậy là hai seed mới. Luật đòi, ở cả ba shift và với 3 seed cho nhánh replay: ALDI++ hơn FT, tỉ lệ lớn hơn 0,5 với cận dưới CI trên 0,5, và không quá 5% lần bootstrap không xác định. Cả N = 10 và N = 100 đều thỏa. Luận điểm được nêu như một phát biểu về lợi thế **so với fine-tune**; ở N = 10 nó phải đi kèm cách đọc so với mô hình nguồn (kết luận 1).

## Thí nghiệm cân bằng lượt ảnh có nhãn ở N = 1.000

ALDI++ và replay được chạy lại ở N = 1.000 với 3.000 update thay vì 2.000, 3 seed, để số lượt ảnh target có nhãn bằng của FT (3.000 lượt). Mọi thiết lập khác giữ nguyên. Số liệu: `rq5_matched_passes.csv`.

| Shift | ALDI++ gốc − FT | ALDI++ cân bằng − FT (CI 95%) | replay cân bằng − FT (CI 95%) | ALDI++ − replay, cân bằng (CI 95%) | Theo luật đã ghi |
|---|---:|---:|---:|---:|---|
| lighting | −0,0111 | −0,0076 (−0,0091; −0,0058) | −0,0064 (−0,0077; −0,0049) | −0,0012 (−0,0021; −0,0002) | "FT dẫn trước" đứng vững |
| weather | −0,0032 | −0,0021 (−0,0038; +0,0002) | −0,0015 (−0,0034; +0,0006) | −0,0005 (−0,0022; +0,0011) | rút lại: không có khác biệt có ý nghĩa |
| compound | −0,0117 | −0,0074 (−0,0097; −0,0047) | −0,0058 (−0,0082; −0,0035) | −0,0016 (−0,0035; +0,0007) | "FT dẫn trước" đứng vững |

- **Thêm 1.000 update thu hẹp khoảng một phần ba khoảng cách (31% / 36% / 36% ở lighting / weather / compound) nhưng không xóa được nó ở lighting và compound.** Thêm 1.000 update giúp ALDI++ +0,0034 / +0,0012 / +0,0043 AP (lighting / weather / compound).
- **Luật đọc kết quả** (ghi trước khi chấm): "FT dẫn trước từ N = 1.000" đứng vững khi chênh lệch ≤ −0,005 và cận trên CI dưới 0; nếu không thì rút lại. Văn bản luật không nói rõ áp dụng gộp hay theo từng shift; ở đây nó được áp dụng cho từng shift, và cách áp dụng này được ghi lại ở DECISIONS 2026-11-23.
- **Distillation không đóng góp gì trên reliable mAP ở N = 1.000.** Sau khi cân bằng, ALDI++ ngang hoặc thấp hơn replay một chút; ở lighting CI của chênh lệch không chứa 0. Theo lớp thì đây là bù trừ: car được +0,003 (lighting) và +0,002 (compound), còn traffic light và traffic sign ở lighting thay đổi −0,004 và −0,005 (`rq5_matched_passes_per_class.csv`).
- So với mô hình nguồn, replay tái tạo toàn bộ lợi ích của ALDI++ sau khi cân bằng: tỉ lệ 1,06 (1,01–1,10) / 1,05 (0,89–1,26) / 1,06 (0,98–1,15) (lighting / weather / compound).
- So với FT, sau khi cân bằng, replay chiếm 0,84 (0,74–0,98) phần thua ở lighting và 0,79 (0,56–1,06) ở compound (40 trong 1.000 lần bootstrap không xác định, sát giới hạn 5%); weather không xác định (999 trong 1.000). Đây là tỉ lệ của một phần thua, không phải bằng chứng cho luận điểm chính (`rq5_replay_share_matched.csv`).
- **Giới hạn của phép cân bằng:** nó cân bằng số lượt ảnh có nhãn mà student thấy. Model được chấm là EMA teacher, và sau 3.000 update teacher vẫn mang khoảng 30% trọng số của khởi tạo từ source (mục Hạn chế, điểm 1). Độ trễ này là một cách giải thích khác cho phần thua còn lại.
- N× theo luật 2026-11-03 (bậc N nhỏ nhất mà từ đó ALDI++ không còn lợi thế có ý nghĩa) không đổi: (100; 1.000] ở cả ba shift. Ở weather, N× chỉ nói rằng lợi thế của ALDI++ mất đi, không nói rằng FT vượt lên: kết luận "FT dẫn trước" đã được rút lại.

## Kết quả phụ

**Quên source.** Thay đổi reliable mAP trên source test so với source model:

| N | FT | adapt | replay |
|---:|---:|---:|---:|
| 10 | −0,055 đến −0,058 | −0,003 đến −0,004 | −0,003 |
| 100 | −0,018 đến −0,022 | −0,001 đến −0,002 | ≈ 0 |
| 1000 | −0,006 đến −0,016 | ≈ 0 | ≈ 0 |
| 5000 (lighting) | −0,020 | −0,010 | |

ALDI++ và replay gần như không quên source, nhờ luôn trộn ảnh source vào batch. Đây là một lợi thế thực tế khi cần giữ hiệu năng trên domain cũ. Với 3.000 update ở N = 1.000 (thí nghiệm cân bằng), thay đổi trên source test của ALDI++ là −0,004 / −0,002 / −0,004 AP (lighting / weather / compound), còn replay là 0,000 / +0,001 / 0,000 (`rq5_forgetting_revision.csv`).

**Theo lớp** (Δ tại N=0):

- Car hưởng lợi nhiều nhất: +0,021 (lighting), +0,028 (compound), +0,009 (weather).
- Traffic light đổi ít nhất: 0,000 (lighting), +0,004 (weather), +0,005 (compound; vừa qua ngưỡng có ý nghĩa).
- Ở N=1000, cả bốn lớp đều âm hoặc ≈ 0.

## Hạn chế cần nêu

1. **Số lần thấy ảnh có nhãn không cân bằng ở N lớn.** Luật T_run = max(T, 2N) được ghi ngày 2026-11-03, trước khi pilot schedule chốt số epoch của FT, và giả định FT chạy 2 epoch. Pilot chốt FT ở **E=3** (2026-11-05), và luật run length không được sửa. Cụ thể:
   - Ở N=1000, FT chạy 1.500 update với 2 ảnh target có nhãn mỗi step, tức khoảng 3.000 lượt ảnh có nhãn.
   - `adapt`/`replay` chạy 2.000 update với 1 ảnh target có nhãn mỗi step, tức khoảng 2.000 lượt, và nửa còn lại của luồng có nhãn là ảnh source.

   Phần thua ở N ≥ 1000 vì vậy **lẫn với chênh lệch lượng huấn luyện có nhãn**. Không thể kết luận ALDI++ về bản chất kém FT ở N lớn; kết luận đúng là: *với lượng update đã đăng ký trước, nó không còn lợi thế từ N=1000.* Vị trí crossover (100, 1.000] là của cấu hình này.

   Thí nghiệm cân bằng ở N = 1.000 (mục riêng ở trên) cho thấy thêm 1.000 update thu hẹp khoảng một phần ba khoảng cách (31% / 36% / 36% ở lighting / weather / compound). Phần thu hẹp này không tách được giữa số lượt ảnh có nhãn và độ trễ của EMA teacher, vì cả hai cùng thay đổi khi tăng số update. Ở lighting N = 5.000 chưa có thí nghiệm cân bằng, nên kết luận ở mức N đó vẫn lẫn với chênh lệch này.

   Phép cân bằng chỉ cân bằng số lượt ảnh có nhãn mà **student** thấy. Model được chấm là EMA teacher: sau 3.000 update nó vẫn mang trọng số 0,9996^3000 ≈ 0,30 của khởi tạo từ source (sau 2.000 update là 0,45). Độ trễ của teacher vì vậy là một cách giải thích khác cho phần thua còn lại, bên cạnh "FT dẫn trước".
2. **Lợi thế ở N nhỏ phụ thuộc vào schedule FT.** FT dùng 500 update cố định nên overfit ở N ≤ 50. Nhánh replay (FT có trộn source, augmentation mạnh và EMA) đã thu hẹp phần lớn khoảng cách ở N=10. Một FT có số update tỉ lệ theo N có thể cũng làm được điều đó, nhưng thí nghiệm này chưa được chạy.
3. **Bản triển khai lại, không phải code gốc.** ALDI++ được viết lại trên torchvision, với ảnh cố định 1280×720, không burn-in và LR 1e-3 (DECISIONS 2026-11-03). Pilot cho thấy lợi ích UDA trên lighting val nhỏ (+0,009 AP), và test xác nhận mức tương tự.
4. **Bảng phân tách theo từng ô (adapt − replay, replay − FT) chỉ dùng seed 0;** CI của nó chỉ tính nhiễu ảnh test. Bảng tỉ lệ dùng 3 seed ở N = 10 và 100. Ở N = 1.000 với số update gốc, replay chỉ có seed 0; 3 seed chỉ có ở thí nghiệm cân bằng.
5. **Weather:** trừ N = 10 (+0,057, chủ yếu do FT overfit), mọi Δ đều nhỏ về AP tuyệt đối (|Δ| ≤ 0,014). Δ(0) = 0,0055 chỉ vừa qua ngưỡng 0,005, và ở N = 1.000 sau khi cân bằng không còn khác biệt có ý nghĩa với FT.
6. **Lần chấm test:** mỗi plan đóng băng được chấm đúng một lần. 46 bộ dự đoán của grid RQ5 được chấm ngày 2026-11-07 qua plan của grid; 30 bộ dự đoán bổ sung được chấm ngày 2026-11-23 qua plan của đợt bổ sung; campaign FT dùng để so sánh được chấm ngày 2026-11-06 qua plan của nó.

## Tóm tắt

Ở ngân sách nhãn nhỏ (hai mức được thử là N = 10 và N = 100), một nhánh đối chứng không có adaptation tái tạo 76–94% lợi thế của ALDI++ so với fine-tune, ở cả ba shift và với 3 seed; điều kiện đã ghi trước để nêu luận điểm này được thỏa. Ở N = 10, lợi thế đó chủ yếu là việc tránh được thiệt hại do fine-tune overfit: so với mô hình nguồn, ALDI++ chỉ hơn +0,003 đến +0,011 AP, không hơn mức khi không có nhãn nào. Ở N = 100, kết luận giữ nguyên với cả hai mốc so sánh. Ở N = 1.000, sau khi cân bằng số lượt ảnh có nhãn, fine-tune vẫn dẫn trước ALDI++ ở lighting và compound và không khác biệt có ý nghĩa ở weather; distillation không đóng góp gì ở mức N này.
