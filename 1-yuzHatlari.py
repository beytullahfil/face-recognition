import cv2
import dlib
import numpy as np


# 1. Yüz algılama modeli yükle
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 2. Yüz hatları çıkarıcı modelini yükle
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3. Özellik çıkarma fonksiyonu
def extract_features(landmarks):
    """
    Yüz hatlarından mesafeleri ve özellikleri çıkarır.
    """
    feature_vector = []
    
    # Örnek 1: Gözler arasındaki mesafe
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    feature_vector.append(eye_distance)
    
    # Örnek 2: Burun ucuyla çene arasındaki mesafe
    nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
    chin = (landmarks.part(8).x, landmarks.part(8).y)
    nose_chin_distance = np.linalg.norm(np.array(nose_tip) - np.array(chin))
    feature_vector.append(nose_chin_distance)
    
    # Örnek 3: Ağız genişliği
    left_mouth = (landmarks.part(48).x, landmarks.part(48).y)
    right_mouth = (landmarks.part(54).x, landmarks.part(54).y)
    mouth_width = np.linalg.norm(np.array(left_mouth) - np.array(right_mouth))
    feature_vector.append(mouth_width)
    
    # Daha fazla mesafe/özellik eklenebilir
    return feature_vector

# 4. Görüntüyü yükle
image_path = "test_image.jpg"  # Test için bir yüz fotoğrafı kullan
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 5. Yüzleri algıla ve özellikleri çıkar
faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
features_list = []  # Tüm yüzlerden çıkarılan özellikler burada tutulacak

for (x, y, w, h) in faces:
    # Yüz çevresine dikdörtgen çiz
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Dlib'e uygun şekilde yüz bölgesini formatla
    dlib_rectangle = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
    
    # Yüz hatlarını çıkar
    landmarks = landmark_detector(gray_image, dlib_rectangle)
    
    # Özellikleri çıkar ve listeye ekle
    features = extract_features(landmarks)
    features_list.append(features)
    
    # Yüz noktalarını görselleştir (isteğe bağlı)
    for n in range(0, 68):
        x_point = landmarks.part(n).x
        y_point = landmarks.part(n).y
        cv2.circle(image, (x_point, y_point), 2, (255, 0, 0), -1)

# 6. Çıkarılan özellikleri görüntüle
print("Çıkarılan Özellikler:", features_list)

# 7. Sonucu görüntüle
cv2.imshow("Yüz ve Yüz Hatları", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
