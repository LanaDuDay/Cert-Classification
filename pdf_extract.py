import os
import fitz  # PyMuPDF
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    # Mở tệp PDF
    doc = fitz.open(pdf_path)
    
    image_paths = []
    
    # Duyệt qua từng trang của PDF
    for i in range(len(doc)):
        page = doc.load_page(i)  # Lấy trang
        pix = page.get_pixmap(dpi=dpi)  # Chuyển đổi trang thành ảnh với độ phân giải DPI
        image_path = f"{output_folder}/page_{i + 1}.png"
        pix.save(image_path)  # Lưu ảnh
        image_paths.append(image_path)
    
    return image_paths

def prepare_image(image_path, target_size=(512, 512)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def classify_pdf_pages(image_paths, model, pca):
    class_names = ['Certificate', 'Scoreboard']
    for image_path in image_paths:
        img_array = prepare_image(image_path)
        features = img_array.reshape(img_array.shape[0], -1)
        features_pca = pca.transform(features)
        
        # Đo thời gian dự đoán cho từng trang
        start_time = time.time()
        predictions = model.predict(features_pca)
        end_time = time.time()
        
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_names[predicted_class[0]]
        
        print(f'Page {image_paths.index(image_path) + 1}: {predicted_label} (Time taken: {end_time - start_time:.4f} seconds)')

# Đường dẫn PDF và thư mục output
pdf_path = r"C:\Users\huy\Desktop\Bằng - Bảng điểm\Bằng - Bảng điểm\CAO THANH HOA.pdf"
output_folder = r'C:\Users\huy\Desktop\Cuong-Classification\train\Output'

# Đảm bảo thư mục output tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Đo thời gian chuyển đổi PDF thành hình ảnh
start_time = time.time()
image_paths = convert_pdf_to_images(pdf_path, output_folder)
end_time = time.time()
print(f"Time taken to convert PDF to images: {end_time - start_time:.4f} seconds")

# Tải mô hình và PCA
model = load_model('pca_certificate_vs_transcript_classifier.h5')
pca = joblib.load('pca_model.pkl')

# Phân loại từng trang
start_time = time.time()
classify_pdf_pages(image_paths, model, pca)
end_time = time.time()
print(f"Total time taken for classification: {end_time - start_time:.4f} seconds")
