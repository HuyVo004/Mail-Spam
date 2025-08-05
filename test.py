import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Tải dữ liệu NLTK
nltk.download("stopwords")
nltk.download("wordnet")

# Khởi tạo công cụ xử lý ngôn ngữ
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Hàm tiền xử lý văn bản
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# Đọc dữ liệu
raw_mail_data = pd.read_csv("mail_data.csv")
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), "")

# Gán nhãn: spam = 0, ham = 1
mail_data.loc[mail_data["Category"] == "spam", "Category"] = 0
mail_data.loc[mail_data["Category"] == "ham", "Category"] = 1

# Tiền xử lý nội dung
mail_data["Message"] = mail_data["Message"].apply(preprocess)

# Tách đặc trưng và nhãn
X = mail_data["Message"]
Y = mail_data["Category"].astype("int")

# In phân bố lớp
print("Phân bố lớp:")
print(Y.value_counts(), "\n")
print(" Tỉ lệ phần trăm:")
print(Y.value_counts(normalize=True) * 100)

# Chia tập train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Tính toán class weights
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(Y_train), y=Y_train
)
weight_dict = dict(zip(np.unique(Y_train), class_weights))
print("\nTrọng số lớp được áp dụng:", weight_dict)

# Khởi tạo mô hình SVM với trọng số lớp
model = SVC(kernel="linear", class_weight=weight_dict)
model.fit(X_train_features, Y_train)

# Dự đoán và đánh giá
train_predictions = model.predict(X_train_features)
test_predictions = model.predict(X_test_features)

train_accuracy = accuracy_score(Y_train, train_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)

print("\n Accuracy on training data:", train_accuracy)
print(" Accuracy on test data:", test_accuracy)
print(
    "\nClassification Report:\n",
    classification_report(Y_test, test_predictions, target_names=["Spam", "Ham"]),
)

# Lưu mô hình và vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(feature_extraction, vec_file)

print("Đã lưu xong model.pkl và vectorizer.pkl")

# Kiểm tra dự đoán với input từ người dùng
user_input = input("\nNhập nội dung email cần kiểm tra: ")
processed_input = preprocess(user_input)

if processed_input.strip() == "":
    print("Không thể phân tích email: nội dung không hợp lệ hoặc không có từ có nghĩa.")
else:
    input_data_features = feature_extraction.transform([processed_input])
    prediction = model.predict(input_data_features)
    print("Dự đoán:", "Ham mail" if prediction[0] == 1 else "Spam mail")
