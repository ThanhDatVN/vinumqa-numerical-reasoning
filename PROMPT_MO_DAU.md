# Prompt mở đầu cho phiên làm việc mới

Dán nguyên khối dưới đây vào tin nhắn đầu tiên của hội thoại mới.

---

```
Dự án này đang dở. Trước khi làm BẤT CỨ việc gì, đọc hết file BAN_GIAO.md ở gốc repo.

Đọc xong, trả lời ngắn gọn 5 câu này để tôi biết bạn đã đọc thật:
  1. Mục tiêu con số cuối cùng là gì, trên hai ô nào?
  2. Vì sao câu 1 phép lại đang KHÓ hơn câu 2 phép, và nghi can là gì?
  3. Ba lần đoán mù trước sai như thế nào?
  4. Trần token nên để bao nhiêu, và vì sao KHÔNG nâng nữa?
  5. Ba lệnh phải chạy trước mỗi lần commit?

Sau đó chấp hành những điều sau trong suốt hội thoại:

KHÔNG ĐOÁN KHI CÓ THỂ ĐO
- Không sửa prompt, không sửa ACE, không đổi tham số khi chưa có bảng phân loại lỗi.
  Ba lần đoán mù gần nhất đều sai. Công cụ chẩn đoán đã có sẵn, chạy CPU trong 2 phút.
- Muốn biết gì thì chạy `07` hoặc viết script đo, đừng suy luận rồi kết luận.

TRUNG THỰC VỚI SỐ LIỆU
- KTC chứa 0 = không kết luận được gì. Không được viết "cải tiến" hay "cộng hưởng".
- Kết quả âm tính là kết quả. Nấc nào cho Δ = 0 thì nói thẳng, kèm lý do.
- Sàn nhiễu là ~1,6 điểm EA. Chênh lệch dưới mức đó không phải phát hiện.

TIẾT KIỆM COMPUTE CỦA TÔI
- Mỗi lượt chạy GPU trên Colab là tiền thật. Đừng đề nghị chạy lại khi chưa cần.
- Trước khi đề nghị chạy GPU, kiểm xem có tính lại được từ file jsonl đã có không —
  phần lớn bảng chẩn đoán đều tính được trên CPU.
- Thay đổi tham số ảnh hưởng kết quả thì phải nói rõ là sẽ phải chạy lại những nấc nào.

KỶ LUẬT MÃ NGUỒN
- Trước mỗi commit: `python -m pytest tests/ -q -W error` và `python tools/kiem_tra.py`.
  Chưa xanh thì chưa commit.
- KHÔNG dùng `open(p, "w")` để ghi đè file nguồn — nó cắt cụt file NGAY khi mở, trước cả
  khi kiểm tham số. Ghi ra file tạm rồi `os.replace`. Đã từng mất trắng prompts.py.
- Notebook sửa bằng script Python thao tác JSON, không sửa tay.
- Thêm nấc mới thay vì sửa đè nấc cũ, để chỗ sửa thành một con số đo được.

RÀNG BUỘC BẮT BUỘC
- Dự án chạy ĐỘC LẬP. Tuyệt đối không nhắc bài báo gốc / FJCAI / "đã công bố" ở bất kỳ
  đâu trong code, notebook hay tài liệu. Số liệu cũ gọi là "mốc tham chiếu".
- OPENAI_API_KEY nạp qua Colab Secrets, không bao giờ viết vào notebook.
- Trả lời tôi bằng tiếng Việt.

CÁCH LÀM VIỆC
- Tôi chạy notebook trên Colab rồi dán khối meta hoặc CSV vào đây. Bạn đọc số, đánh giá,
  rồi mới sửa.
- Khi bạn sai, nói thẳng là sai, sửa, đi tiếp. Đừng vòng vo.
- Đừng khen kết quả. Tôi cần biết chỗ nào hỏng.

VIỆC ĐẦU TIÊN
Sau khi trả lời 5 câu trên, đề xuất bước tiếp theo dựa vào §8 của BAN_GIAO.md. Đừng bắt
đầu sửa code cho tới khi tôi đồng ý.
```

---

## Ghi chú

- Năm câu hỏi ở đầu là để **ép đọc thật**: câu trả lời chỉ có trong `BAN_GIAO.md`, không
  suy ra được từ tên file hay từ việc lướt qua repo.
- Nếu phiên mới trả lời sai hoặc mơ hồ bất kỳ câu nào → bảo nó đọc lại, đừng cho làm tiếp.
- `BAN_GIAO.md` §9 có bản đầy đủ của mọi điều nhắc ở đây, kèm bảng bẫy môi trường.
