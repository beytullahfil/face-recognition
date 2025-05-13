import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Verisetini Yükle
data = pd.read_csv("dataset.csv")

# 2. Özellikler ve Etiketleri Ayır
X = data[["Eye_Distance", "Nose_Chin_Distance", "Mouth_Width"]]  # Özellikler
y = data["Label"]  # Etiketler (millet)

# 3. Eğitim ve Test Verilerini Ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli Oluştur ve Eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Modeli Test Et
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Doğruluğu: {accuracy:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# 6. Modeli Kaydet
joblib.dump(model, "millet_tahmin_modeli.pkl")
print("Model 'millet_tahmin_modeli.pkl' olarak kaydedildi!")
