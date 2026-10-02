import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("กำลังโหลดข้อมูลและเทรนโมเดล...")

# 1. โหลดข้อมูล Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. แบ่งข้อมูล Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. สร้างและเทรนโมเดล
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. บันทึกโมเดลเป็นไฟล์ iris_model.pkl
joblib.dump(model, "iris_model.pkl")

print("สร้างไฟล์โมเดล 'iris_model.pkl' สำเร็จเรียบร้อยแล้ว!")