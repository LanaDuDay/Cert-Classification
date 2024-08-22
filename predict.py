import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt thông báo từ TensorFlow

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Tắt cảnh báo của absl

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import joblib
import time

# 1. Tải và chuẩn bị ảnh
def prepare_image(image_path, target_size=(512, 512)):
    # Tải ảnh và thay đổi kích thước
    img = load_img(image_path, target_size=target_size)
    # Chuyển đổi ảnh thành mảng numpy
    img_array = img_to_array(img)
    # Thay đổi kích thước ảnh
    img_array = np.expand_dims(img_array, axis=0)
    # Tiền xử lý ảnh
    img_array = img_array / 255.0
    return img_array

# Đường dẫn đến ảnh
image_path = r'C:\Users\huy\Desktop\Cuong-Classification\test\test.jpg'
img_array = prepare_image(image_path)

# 2. Tải mô hình đã lưu và PCA
model = load_model('pca_certificate_vs_transcript_classifier.h5')
pca = joblib.load('pca_model.pkl')

# 3. Trích xuất đặc trưng của ảnh và áp dụng PCA
features = img_array.reshape(img_array.shape[0], -1)  # Chuyển đổi thành (số ảnh, số pixel)
features_pca = pca.transform(features)  # Chuyển đổi bằng PCA

# 4. Đo thời gian dự đoán
start_time = time.time()  # Bắt đầu đếm thời gian
predictions = model.predict(features_pca)
end_time = time.time()  # Kết thúc đếm thời gian

# 5. Tính thời gian dự đoán
prediction_time = end_time - start_time
print(f'Time taken for prediction: {prediction_time:.4f} seconds')

# 6. Xử lý kết quả dự đoán
predicted_class = np.argmax(predictions, axis=1)

# 7. In kết quả dự đoán
class_names = ['Certificate', 'Scoreboard']  # Thay đổi theo lớp của bạn

# Kiểm tra chỉ số dự đoán có hợp lệ không
if 0 <= predicted_class[0] < len(class_names):
    predicted_label = class_names[predicted_class[0]]
    print(f'Predicted label: {predicted_label}')
else:
    print('Predicted class index is out of range')
