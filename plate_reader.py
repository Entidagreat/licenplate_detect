from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2

# Tải font
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b, g, r, a = 0, 255, 0, 0  # Màu xanh lá

# Khởi tạo EasyOCR
reader = Reader(['en','vi'])  # Có thể thêm 'vi' nếu cần

# Khởi tạo camera (0 là camera mặc định, có thể thay bằng 1, 2... nếu có nhiều camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    # Đọc từng khung hình từ camera
    ret, img = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera!")
        break

    # Thay đổi kích thước khung hình (tùy chọn, có thể bỏ nếu muốn giữ nguyên)
    img = cv2.resize(img, (800, 600))

    # Tiền xử lý ảnh
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 200)

    # Tìm contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    number_plate_shape = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approximation) == 4:  # Hình chữ nhật
            number_plate_shape = approximation
            break

    # Xử lý kết quả
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    if number_plate_shape is None:
        text = "Không thấy bảng số xe"
        draw.text((150, 500), text, font=font, fill=(b, g, r, a))
    else:
        # Cắt vùng biển số
        (x, y, w, h) = cv2.boundingRect(number_plate_shape)
        number_plate = grayscale[y:y + h, x:x + w]

        # Nhận diện ký tự bằng EasyOCR
        detection = reader.readtext(number_plate)

        if len(detection) == 0:
            text = "Không thấy bảng số xe"
            draw.text((150, 500), text, font=font, fill=(b, g, r, a))
        else:
            # Ghép tất cả các phần tử thành một chuỗi duy nhất
            sorted_detection = sorted(detection, key=lambda x: x[0][0][1])  # Sắp xếp theo y
            combined_text = "".join([det[1] for det in sorted_detection]).replace(" ", "")
            text = "Biển số: " + combined_text
            cv2.drawContours(img, [number_plate_shape], -1, (255, 0, 0), 3)
            draw.text((200, 500), text, font=font, fill=(b, g, r, a))

    # Chuyển lại ảnh và hiển thị
    img = np.array(img_pil)
    cv2.imshow('Plate Detection', img)

    # Thoát vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()