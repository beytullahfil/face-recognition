import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Veri Yükleme
data = pd.read_csv("yuz_ozellikleri.csv")

# Özellikler ve etiketleri ayır
X = data.drop("label", axis=1)  # Özellik sütunları
y = data["label"]  # Millet etiketleri

# Etiketleri encode et (örneğin: "African" -> 0, "American" -> 1, ...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Sınıf etiketlerini .txt dosyasına kaydet
class_names = label_encoder.classes_
with open("labels.txt", "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# 2. Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Model Oluşturma
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),  # Özellik boyutuna göre giriş katmanı
    tf.keras.layers.Dense(128, activation='relu'),  # Gizli katman
    tf.keras.layers.Dropout(0.2),  # Regularization için dropout
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Çıkış katmanı (sınıf sayısı kadar)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Modeli Eğit
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# 5. Model Performansını Test Et
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_accuracy:.2f}")

# 6. Modeli TFLite Formatına Dönüştür
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite modeli kaydet
with open("millet_tahmini_modeli.tflite", "wb") as f:
    f.write(tflite_model)

print("Model TFLite formatında kaydedildi.")
