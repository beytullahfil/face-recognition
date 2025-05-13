import cv2
import dlib
import os
import pandas as pd
import numpy as np

# 1. Yüz algılayıcı ve landmark modeli yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 2. Özellik çıkarma fonksiyonu
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None  # Yüz bulunamadı
    landmarks = predictor(gray, faces[0])
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

    # Yüz oranı
    face_width = np.linalg.norm(points[16] - points[0])  # Sağ-sol yüz
    face_height = np.linalg.norm(points[8] - points[27])  # Çene-burun
    face_ratio = face_width / face_height

    # Gözler arası mesafe
    eye_distance = np.linalg.norm(points[42] - points[39])

    # Burun-Göz Mesafesi
    nose_to_left_eye = np.linalg.norm(points[27] - points[36])
    nose_to_right_eye = np.linalg.norm(points[27] - points[45])

    # Dudak dolgunluğu
    upper_lip = np.linalg.norm(points[62] - points[51])
    lower_lip = np.linalg.norm(points[66] - points[57])
    lip_thickness = upper_lip + lower_lip

    # Özellikler
    return {
        "face_width_height_ratio": face_ratio,
        "eye_distance": eye_distance,
        "nose_to_left_eye": nose_to_left_eye,
        "nose_to_right_eye": nose_to_right_eye,
        "lip_thickness": lip_thickness,
    }

# 3. Veri klasörünü işle
data_dir = "data"  # Veri klasörünün yolu
output_data = []

for label in os.listdir(data_dir):  # Her millet klasörünü gez
    label_path = os.path.join(data_dir, label)
    if not os.path.isdir(label_path):
        continue
    for image_name in os.listdir(label_path):  # Her resim dosyasını gez
        image_path = os.path.join(label_path, image_name)
        try:
            features = extract_features(image_path)
            if features:
                features["label"] = label  # Millet etiketini ekle
                output_data.append(features)
        except Exception as e:
            print(f"Hata: {image_path}, {e}")

# 4. Özellikleri CSV dosyasına kaydet
df = pd.DataFrame(output_data)
df.to_csv("yuz_ozellikleri.csv", index=False)
print("Özellik çıkarma işlemi tamamlandı ve CSV'ye kaydedildi!")
