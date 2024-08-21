# scripts/data_preparation.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(data_dir, img_size=(224, 224), batch_size=32):
    # Khởi tạo ImageDataGenerator với augmentation cho tập huấn luyện
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Chuẩn hóa giá trị pixel về [0, 1]
        rotation_range=20,  # Xoay ngẫu nhiên các ảnh trong khoảng 20 độ
        width_shift_range=0.2,  # Dịch ngang ngẫu nhiên trong khoảng 20% chiều rộng ảnh
        height_shift_range=0.2,  # Dịch dọc ngẫu nhiên trong khoảng 20% chiều cao ảnh
        shear_range=0.2,  # Biến đổi xiên ngẫu nhiên
        zoom_range=0.2,  # Phóng to hoặc thu nhỏ ngẫu nhiên trong khoảng 20%
        horizontal_flip=True,  # Lật ngang ngẫu nhiên ảnh
        fill_mode='nearest',  # Điền các điểm trống sau khi biến đổi bằng giá trị gần nhất
        validation_split=0.2  # Tách 20% dữ liệu cho tập kiểm thử
    )

    # Tạo generator cho tập huấn luyện
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,  # Kích thước ảnh đưa vào mô hình (224x224)
        batch_size=batch_size,
        class_mode='binary',  # Mã hóa nhãn nhị phân cho bài toán phân loại 2 lớp
        subset='training',
        color_mode="rgb"  # Đảm bảo ảnh được load dưới dạng RGB với 3 kênh màu
    )

    # Tạo generator cho tập kiểm thử
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',  # Mã hóa nhãn nhị phân cho bài toán phân loại 2 lớp
        subset='validation',
        color_mode="rgb"  # Đảm bảo ảnh được load dưới dạng RGB với 3 kênh màu
    )
    
    return train_generator, validation_generator
