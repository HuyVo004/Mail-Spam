import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Tải dữ liệu cần thiết từ NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Khởi tạo lemmatizer và stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def preprocess(text):
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ dấu câu
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tách từ và loại bỏ stop words, lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Bước 1: Đọc dữ liệu
raw_mail_data = pd.read_csv('mail_data.csv')

# Bước 2: Thay thế giá trị null bằng chuỗi rỗng
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Bước 3: Gán nhãn: spam = 0, ham = 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Bước 4: Tiền xử lý nội dung email
mail_data['Message'] = mail_data['Message'].apply(preprocess)

# Bước 5: Tách dữ liệu và nhãn
X = mail_data['Message']
Y = mail_data['Category'].astype('int')

# Bước 6: Chia tập train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Bước 7: Chuyển đổi văn bản thành vector đặc trưng TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Bước 8: Khởi tạo và huấn luyện mô hình SVM
model = SVC(kernel='linear')
model.fit(X_train_features, Y_train)

# Bước 9: Đánh giá mô hình
# Trên tập huấn luyện
train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)
print('Accuracy on training data:', train_accuracy)

# Trên tập kiểm tra
test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)
print('Accuracy on test data:', test_accuracy)

# Bước 10: Lưu mô hình và vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(feature_extraction, vec_file)

print("Đã lưu xong model.pkl và vectorizer.pkl")

# Bước 11: Kiểm tra dự đoán với email nhập từ người dùng (tuỳ chọn)
user_input = input("Nhập nội dung email cần kiểm tra: ")
input_mail = [preprocess(user_input)]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
print('Dự đoán:', 'Ham mail' if prediction[0] == 1 else 'Spam mail')
