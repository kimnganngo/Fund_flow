# FlowLink — Phân tích mối quan hệ nguồn tiền giữa các tài khoản chứng khoán

Web app Streamlit giúp phát hiện **cùng nguồn tiền**, **tài khoản trung gian**, và **cảnh báo cross-group** bằng cách
kết hợp 2 nguồn dữ liệu:
- (A) **Luồng tiền**: Người nộp/chuyển → Tài khoản, số lệnh, số tiền, nộp/rút, CTCK…
- (B) **Danh sách nhóm (danh tính)**: STT nhóm, Mã TK, Tên, Mối quan hệ với công ty.

## Chạy local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Cloud
1) Push repo lên GitHub
2) Trên Streamlit Cloud: chọn repo + branch + file `app.py`
3) Xong.

## File mẫu
- `sample_data/flow_sample.csv`
- `sample_data/group_sample.csv`

> Lưu ý: Có thể tải 2 file mẫu này trực tiếp trong app (nút "Tải dataset mẫu").
