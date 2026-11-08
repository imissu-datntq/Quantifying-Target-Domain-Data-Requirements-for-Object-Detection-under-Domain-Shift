# Kết quả RQ5: ALDI++ so với fine-tune thường (chạy local, 2026-11-07)

**Câu hỏi:** một phương pháp domain adaptation thật sự (ALDI++, Kay et al., CVPR 2024) có giúp tiết kiệm nhãn target so với fine-tune thường (FT) không? Nếu có, tới N bao nhiêu thì FT đuổi kịp?

**Grid:** 46 run, chạy trên máy local (RTX 3060), xong lúc 2026-11-07 07:11. Luật phân tích được ghi trước mọi kết quả (`.agent/DECISIONS.md`, 2026-11-03). Pilot trên lighting val chốt α = 0,9996 và T = 2.000 update (DECISIONS 2026-11-06).

**Chấm test đúng một lần:**

- Plan đóng băng `runs/local/rq5/final_plan.json` gồm 46 bộ prediction, sha256 `3df3d788…b4b7`.
- Bootstrap dùng cùng seed (20261030) và cùng danh sách ảnh test với campaign FT local. Vì vậy mọi so sánh DA − FT được ghép cặp trên cùng các lần lấy mẫu lại ảnh test.

Số liệu gốc nằm trong `runs/local/rq5/analysis/` (`rq5_*.csv`, `summary.json`, hình `rq5_<domain>.png`). Script phân tích: `scripts/analyze_rq5.py`.

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

## Kết quả chính: Δ(N) = ALDI++ − FT

| Shift | N=0 (UDA) | N=10 | N=100 | N=1000 | N=5000 |
|---|---:|---:|---:|---:|---:|
| lighting | **+0,009** (0,008; 0,011) | **+0,070** (0,068; 0,073) | **+0,015** (0,014; 0,017) | −0,011 (−0,012; −0,009) | −0,010 (−0,011; −0,008) |
| weather | **+0,006** (0,003; 0,007) | **+0,057** (0,050; 0,066) | **+0,014** (0,009; 0,019) | −0,003 (−0,005; −0,001) | |
| compound | **+0,011** (0,008; 0,014) | **+0,061** (0,054; 0,068) | **+0,016** (0,012; 0,020) | −0,012 (−0,014; −0,009) | |

In đậm là lợi thế có ý nghĩa theo luật. Weather N=0 vừa qua ngưỡng (0,0055 ≥ 0,005).

**Crossover N×: trong khoảng (100, 1.000] ở cả 3 shift.**

- Từ N=1000 trở lên, ALDI++ không còn lợi thế; nó **kém FT có ý nghĩa thống kê**, CI hoàn toàn dưới 0.
- Lighting N=5000 cũng vậy (−0,010).

**mAP tuyệt đối** (trung bình seed):

| Shift | zero-shot | DA N=0 | DA N=10 | DA N=100 | DA N=1000 | FT N=100 | FT N=1000 | FT ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lighting | 0,244 | 0,253 | 0,253 | 0,260 | 0,262 | 0,244 | 0,273 | 0,296 |
| weather | 0,320 | 0,326 | 0,323 | 0,327 | 0,329 | 0,314 | 0,332 | 0,336 |
| compound | 0,219 | 0,230 | 0,230 | 0,239 | 0,241 | 0,223 | 0,252 | 0,257 |

Lighting N=5000: DA 0,276, FT 0,286.

Đường DA gần như phẳng: từ N=0 tới N=1000 chỉ tăng thêm khoảng 0,003–0,011 AP. Đường FT dốc hơn nhiều sau khi qua vùng overfit ở N nhỏ.

## Ngân sách nhãn tương đương N_eq

N_eq là số ảnh FT cần để đạt cùng mAP với DA. Nội suy tuyến tính theo log N, lấy điểm cắt bền vững (lần cắt lên cuối cùng của đường FT). CI 95% lấy từ bootstrap.

| Shift | DA N=0 | DA N=10 | DA N=100 | DA N=1000 | DA N=5000 |
|---|---|---|---|---|---|
| lighting | 160 (142–184) | 162 (145–183) | 236 (215–287) | 272 (235–344) | 1.556 (1.343–1.894) |
| weather | 349 (240–511) | 242 (184–379) | 451 (313–672) | 575 (413–834) | |
| compound | 153 (123–182) | 152 (121–180) | 242 (209–332) | 300 (234–410) | |

**Cách đọc:**

- **Adapt không nhãn (UDA)** đáng giá khoảng **150–350 ảnh có nhãn** khi dùng FT với schedule này. Theo đơn vị recovery: 0,17 (lighting), 0,36 (weather), 0,29 (compound).
- DA với 10 ảnh có nhãn **không hơn** DA không nhãn. 10 ảnh đó gần như không thêm gì.
- DA với 1.000 ảnh có nhãn chỉ tương đương FT với 270–575 ảnh. Nói cách khác, ở vùng N lớn ALDI++ (theo cách triển khai này) **dùng nhãn kém hiệu quả hơn FT**.

## Phân tách đóng góp (seed 0)

Có hai nguồn đóng góp:

- **adapt − replay:** giá trị của distillation trên ảnh target không nhãn.
- **replay − FT:** giá trị của việc trộn ảnh source, augmentation mạnh và EMA.

| Shift | N | adapt − replay | replay − FT |
|---|---:|---:|---:|
| lighting | 0 | +0,006 | +0,004 |
| | 10 | +0,009 | +0,062 |
| | 100 | +0,002 | +0,012 |
| | 1000 | +0,001 (CI chứa 0) | −0,010 |
| weather | 0 | +0,003 | +0,003 |
| | 10 | +0,004 | +0,064 |
| | 100 | +0,003 | +0,016 |
| | 1000 | 0,000 (CI chứa 0) | −0,003 |
| compound | 0 | +0,009 | +0,002 (CI chứa 0) |
| | 10 | +0,010 | +0,046 |
| | 100 | +0,004 | +0,013 |
| | 1000 | −0,000 (CI chứa 0) | −0,013 |

Ở N = 0, "replay" là source model train tiếp trên source với augmentation mạnh và EMA. Vì vậy replay − FT ở đây là replay − zero-shot.

**Ba kết luận từ bảng này:**

1. **Lợi thế lớn ở N=10 gần như hoàn toàn đến từ việc tránh overfit, không phải từ adaptation.**
   - Replay hơn FT khoảng 0,05–0,06 AP, còn distillation chỉ thêm 0,004–0,010.
   - FT với 10 ảnh và 500 update bị overfit nặng (recovery −1,2 đến −3,5, xem `docs/results_local_rq1_rq4_vi.md`). Trộn source và EMA ngăn được điều đó.
2. **Distillation trên ảnh không nhãn có giá trị thật nhưng nhỏ:** khoảng 0,003–0,010 AP khi N ≤ 10, giảm còn ≤ 0,004 ở N=100 và về 0 ở N=1000.
3. **Phần thua ở N=1000 không do distillation.** Replay thua FT đúng bằng mức đó, và adapt − replay ≈ 0. Nguyên nhân nằm ở thiết kế luồng có nhãn chung của `adapt` và `replay`, xem mục Hạn chế (điểm 1).

## Kết quả phụ

**Quên source.** Thay đổi reliable mAP trên source test so với source model:

| N | FT | adapt | replay |
|---:|---:|---:|---:|
| 10 | −0,055 đến −0,058 | −0,003 đến −0,004 | −0,003 |
| 100 | −0,018 đến −0,022 | −0,001 đến −0,002 | ≈ 0 |
| 1000 | −0,006 đến −0,016 | ≈ 0 | ≈ 0 |
| 5000 (lighting) | −0,020 | −0,010 | |

ALDI++ và replay gần như không quên source, nhờ luôn trộn ảnh source vào batch. Đây là một lợi thế thực tế khi cần giữ hiệu năng trên domain cũ.

**Theo lớp** (Δ tại N=0):

- Car hưởng lợi nhiều nhất: +0,021 (lighting), +0,028 (compound), +0,009 (weather).
- Traffic light gần như không đổi (≤ 0,005).
- Ở N=1000, cả bốn lớp đều âm hoặc ≈ 0.

## Hạn chế cần nêu

1. **Số lần thấy ảnh có nhãn không cân bằng ở N lớn.** Luật T_run = max(T, 2N) được ghi trước với lý do "khớp E=2 của FT". Nhưng trên máy local, pilot chốt FT ở **E=3**, và luật run length không được sửa. Cụ thể:
   - Ở N=1000, FT chạy 1.500 update với 2 ảnh target có nhãn mỗi step, tức khoảng 3.000 lượt ảnh có nhãn.
   - `adapt`/`replay` chạy 2.000 update với 1 ảnh target có nhãn mỗi step, tức khoảng 2.000 lượt, và nửa còn lại của luồng có nhãn là ảnh source.

   Phần thua ở N ≥ 1000 vì vậy **lẫn với chênh lệch lượng huấn luyện có nhãn**. Không thể kết luận ALDI++ về bản chất kém FT ở N lớn; kết luận đúng là: *với lượng update đã đăng ký trước, nó không còn lợi thế từ N=1000.* Vị trí crossover (100, 1.000] là của cấu hình này.
2. **Lợi thế ở N nhỏ phụ thuộc vào schedule FT.** FT dùng 500 update cố định nên overfit ở N ≤ 50. Một FT có số update tỉ lệ theo N, hoặc có trộn source, sẽ thu hẹp phần lớn khoảng cách ở N=10. Bảng phân tách cho thấy điều đó: replay đã gánh phần lớn lợi thế.
3. **Bản triển khai lại, không phải code gốc.** ALDI++ được viết lại trên torchvision, với ảnh cố định 1280×720, không burn-in và LR 1e-3 (DECISIONS 2026-11-03). Pilot cho thấy lợi ích UDA trên lighting val nhỏ (+0,009 AP), và test xác nhận mức tương tự.
4. **Phân tách chỉ dùng seed 0;** CI của nó chỉ tính nhiễu ảnh test, không tính nhiễu seed lấy mẫu.
5. **Weather:** mọi Δ đều nhỏ về AP tuyệt đối. Δ(0) = 0,0055 chỉ vừa qua ngưỡng 0,005.
6. **Lần mở nhãn test:** đây là lần đầu tiên nhãn test được mở cho RQ5. Grid RQ5 trên Kaggle đã được khởi chạy ngày 2026-11-04 nhưng bị bỏ dở khi chuyển sang local, và chưa bao giờ được chấm. Campaign FT dùng để so sánh là bản local, đã chấm ngày 2026-11-06.

## Tóm tắt một câu

ALDI++ đáng giá khoảng 150–350 ảnh có nhãn khi không có nhãn nào, và giữ lợi thế tới N=100. Phần lớn lợi thế đó đến từ việc trộn source và EMA, không phải từ distillation. Từ N=1000, fine-tune thường bắt kịp và vượt (crossover trong (100, 1.000]), với lưu ý rằng FT ở đó được huấn luyện trên nhiều lượt ảnh có nhãn hơn.
