# model.py
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Chuẩn bị dữ liệu và chuyển đổi thành vector đặc trưng
datagen = ImageDataGenerator(rescale=1./255)

# Tạo generator cho dữ liệu
generator = datagen.flow_from_directory(
    r'C:\Users\huy\Desktop\Cuong-Classification\train',
    target_size=(512, 512),
    batch_size=12,
    class_mode=None,  # Không cần nhãn cho PCA
    shuffle=False
)

def extract_features(generator):
    features = []
    for batch in generator:
        features.append(batch)
        if len(features) * generator.batch_size >= generator.samples:
            break
    return np.concatenate(features, axis=0)

# Lấy các vector đặc trưng
features = extract_features(generator)
features = features.reshape(features.shape[0], -1)  # Chuyển đổi thành (số ảnh, số pixel)

# 2. Áp dụng PCA
pca = PCA(n_components=10)
features_pca = pca.fit_transform(features)

# 3. Tạo nhãn cho dữ liệu huấn luyện
labels = generator.classes

# Chia dữ liệu PCA thành dữ liệu huấn luyện và kiểm tra
X_train, X_val, y_train, y_val = train_test_split(
    features_pca, labels, test_size=0.2, random_state=42
)

# 4. Tạo mô hình học máy đơn giản
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(features_pca.shape[1],)))
model.add(layers.Dense(2, activation='softmax'))  # 2 đầu ra (Bằng cấp và Bảng điểm)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Huấn luyện mô hình
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=12,
    epochs=10
)

# 6. Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_val, y_val, batch_size=12)
print('Test accuracy:', test_acc)

# 7. Lưu mô hình và PCA
model.save('pca_certificate_vs_transcript_classifier.h5')

# Lưu PCA
import joblib
joblib.dump(pca, 'pca_model.pkl')
