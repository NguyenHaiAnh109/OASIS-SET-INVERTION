# OASIS với scikit-learn

OASIS xấp xỉ tập tiền ảnh của một hàm. Cho hàm `F`, tập đích `U` và
miền tìm kiếm `Ω`, bài toán là tìm các điểm `x` trong `Ω` sao cho
`F(x)` thuộc `U`. Mỗi điểm được gán nhãn `+1` nếu thỏa điều kiện và
`-1` nếu nằm ngoài tập cần tìm.

| File | Nội dung |
| --- | --- |
| `oasis.py` | Thuật toán OASIS và ví dụ hình vành khăn. |
| `solve_ellipse.py` | Import `oasis()`, giải bài toán elip và vẽ kết quả. |

## Cài đặt và chạy

Dùng Python 3.9 trở lên. Đặt hai file Python trong cùng một thư mục,
mở terminal tại đó rồi chạy:

```bash
python -m pip install numpy scipy scikit-learn matplotlib seaborn
python solve_ellipse.py
```

Để chạy ví dụ hình vành khăn, dùng `python oasis.py`.

## Ví dụ elip

Trong miền `[-2, 2] × [-2, 2]`, tìm các điểm thỏa
`x₁² + 4x₂² ≤ 1`. Tập cần tìm là phần bên trong và trên biên elip,
với bán trục ngang bằng 1 và bán trục dọc bằng 0,5.

Script lấy 50 mẫu ngẫu nhiên ban đầu, sau đó lấy thêm 100 mẫu chủ động.
Biểu đồ dùng Seaborn để phân biệt nhãn và kiểu lấy mẫu. Đường liền là
biên SVM học được. Đường đứt nét là biên thật `x₁² + 4x₂² = 1`.

## Thuật toán hoạt động thế nào?

1. Lấy mẫu ngẫu nhiên trong miền tìm kiếm rồi tính nhãn thật cho từng
   điểm. Tập mẫu ban đầu cần có đủ hai nhãn.
2. Huấn luyện SVM với kernel RBF. Mỗi lần huấn luyện, chương trình thử
   các giá trị `gamma` cho đến khi SVM phân loại đúng tập mẫu hiện tại.
   Tham số `C` giữ cố định ở 1000.
3. Chọn một điểm xuất phát ngẫu nhiên và dùng SLSQP để tìm một điểm
   gần nó trên biên dự đoán của SVM.
4. Tính nhãn thật của điểm mới, thêm vào tập huấn luyện rồi lặp lại
   cho đến khi đạt tổng số mẫu.

## Áp dụng cho bài toán khác

Viết hàm gán nhãn mới rồi truyền vào:

```python
model, points, labels = oasis(label_point, bounds, n_initial, n_total)
```

`bounds` chứa một cặp cận dưới và cận trên cho mỗi chiều.
`n_total` tính cả các mẫu ngẫu nhiên ban đầu. Dùng
`model.predict([[x1, x2]])` để dự đoán nhãn cho một điểm 2D. Nếu vẽ
biên thật, cập nhật công thức và mức contour trong hàm vẽ theo bài toán.
