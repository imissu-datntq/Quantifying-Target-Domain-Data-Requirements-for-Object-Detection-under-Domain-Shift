# Nhận xét của giảng viên về báo cáo kết quả (nhận ngày 2026-11-21)

Văn bản dưới đây là bản tổng hợp nhận xét do nhóm chuyển lại, giữ nguyên văn. Báo cáo được nhận xét là `docs/bao_cao_ket_qua_vi.docx` (bản ngày 2026-11-08).

Kế hoạch xử lý: `.agent/plans/active/2026-11-21-lecturer-review-revision.md`. Các luật ghi trước khi chạy thêm: `.agent/DECISIONS.md`, mục 2026-11-21.

## Sáu điểm kỹ thuật

1. Schedule 3 epoch làm ceiling bị undertrain, nên mọi threshold đang optimistic. Ceiling lighting chỉ được tầm 54k image-pass, trong khi source model được gần 120k và schedule 1x tiêu chuẩn là tầm 1,4 triệu. Ceiling thấp -> mẫu số nhỏ -> recovery bị thổi lên -> N75/N90 báo cáo là lower bound của nhu cầu thật. Cần chạy lại ceiling 12 epoch cho compound, weather, lighting. Nếu ceiling 12 epoch cao hơn nhiều thì báo cáo lại threshold; nếu gần bằng thì cũng loại bỏ được phản biện trên.
2. Kết luận RQ5 ở tóm tắt mạnh hơn bằng chứng. Ở 12.3, N = 1.000 fine-tune thấy tầm 3.000 lượt ảnh có nhãn còn ALDI++ chỉ tầm 2.000. Bảng phân tách thì adapt − replay ≈ 0, toàn bộ phần thua nằm ở replay. Nên chạy luôn thí nghiệm đề xuất ở mục 13.
3. Ô Weather cho ra gần như rỗng. Ceiling − zero-shot chỉ 0,016 AP, chỉ gấp 1,6 lần ngưỡng 0,01 mà mình đặt ra để coi là "không xác định". Traffic sign đã rơi xuống dưới (0,0089). Hệ quả: CI của tương tác là (−0,012; +0,016), nửa độ rộng 0,014, trong khi hiệu ứng chính của weather chỉ 0,022. Phép kiểm chỉ phát hiện được tương tác bằng hai phần ba toàn bộ hiệu ứng weather. Do vậy nên ghi "tương tác nhỏ nhất phát hiện được" thay vì kết luận "không có bằng chứng".
4. Báo cáo cả hai lần chạy Kaggle và local như nhau, không để một bên là chính một bên là phụ.
5. R² bị thổi lên nhờ các điểm âm và fit lạc quan (lighting 2.530 so với ladder 5.000). Fit lại chỉ trên vùng dương và đơn điệu.
6. N90 không xác định được cho weather và compound trong phạm vi pool hiện có. Cột N95 bỏ được.

## Diễn đạt lại đóng góp

Đóng góp quan trọng là nhánh điều khiển, không phải các con số. Các em diễn đạt lại:

> "Hầu hết những gì mà thích ứng miền (Domain Adaptation - DAOD) mang lại với small labeled budgets không phải là thích ứng thực sự."

Luận điểm này liên quan trực tiếp đến luận điểm của ALDI (rằng gain của DAOD bị ảnh hưởng bởi các lựa chọn triển khai). Cái này có thể được kiểm chứng bởi người khác, các đường cong hiệu quả dữ liệu trở thành bối cảnh giúp đo lường được, và các ngưỡng được đặt ở vị trí kết quả thực tiễn thứ cấp. Nó cũng làm giảm bớt ảnh hưởng của nhánh thời tiết gần như bằng không. Với tư cách là replication thứ ba của quá trình phân tích thì nó xứng đáng có vị trí của mình, trong khi với tư cách là phép đo ngưỡng thứ ba, nó lại làm giảm giá trị của bài báo.

## Quyết định của nhóm về điểm 4

Nhóm bỏ hẳn lần chạy Kaggle khỏi báo cáo và các ghi chú kết quả thay vì báo cáo hai lần chạy ngang nhau (quyết định ngày 2026-11-21 và 2026-11-23). Lý do: toàn bộ nghiên cứu đã được chạy lại trên máy local, hai lần chạy dùng hai schedule khác nhau (E = 2 và E = 3), và RQ5 chỉ có ở bản local. Cách xử lý này khác với đề nghị của thầy; nhóm sẽ báo lại trực tiếp với thầy.
