import cv2
import dlib
import joblib
import numpy as np

# 1. Yüz algılama ve yüz hatlarını çıkarma modelleri
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 2. Daha önce eğittiğimiz modeli yükle
model = joblib.load("millet_tahmin_modeli.pkl")

# 3. Özellik çıkarma fonksiyonu
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

# 4. Yeni fotoğrafta yüz özelliklerini çıkar
def predict_nationality(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü yüklenemedi!")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        dlib_rectangle = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = landmark_detector(gray_image, dlib_rectangle)
        features = extract_features(landmarks)
        features = np.array(features).reshape(1, -1)  # Modelin beklediği formatta

        # Modelle tahmin yap
        prediction = model.predict(features)
        print(f"Tahmin Edilen Millet: {prediction[0]}")

        # Yüzü kutu içine al ve tahmini göster
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, prediction[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("Tahmin", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 5. Test: Yeni bir fotoğraf ile tahmin yap
predict_nationality("test_image.jpg")
