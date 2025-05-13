import os
import cv2
import dlib
import numpy as np
import pandas as pd

# 1. Yüz algılama ve yüz hatları çıkarma modelleri
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 2. Özellik çıkarma fonksiyonu
def extract_features(landmarks):
    feature_vector = []
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    feature_vector.append(eye_distance)

    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
    chin = (landmarks.part(8).x, landmarks.part(8).y)
    nose_chin_distance = np.linalg.norm(np.array(nose_tip) - np.array(chin))
    feature_vector.append(nose_chin_distance)

    left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
    right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
    mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
    feature_vector.append(mouth_width)

    return feature_vector

# 3. Fotoğrafların bulunduğu klasör
data_dir = "data/"
output_data = []

# 4. Veriseti oluştur
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    
    if not os.path.isdir(label_dir):
        continue
    
    for image_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            dlib_rectangle = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            landmarks = landmark_detector(gray_image, dlib_rectangle)
            features = extract_features(landmarks)
            
            # Label (millet) ve özellikleri birleştir
            output_data.append([label] + features)

# 5. Veriseti CSV'ye kaydet
columns = ["Label", "Eye_Distance", "Nose_Chin_Distance", "Mouth_Width"]
df = pd.DataFrame(output_data, columns=columns)
df.to_csv("dataset.csv", index=False)

print("Veriseti başarıyla oluşturuldu ve 'dataset.csv' dosyasına kaydedildi!")
