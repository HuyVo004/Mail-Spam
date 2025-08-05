from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Flask app
app = Flask(__name__)

# Tải mô hình và vectorizer đã huấn luyện
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Chuẩn bị NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Hàm tiền xử lý nội dung email
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_mail = request.form['email']
    processed_mail = preprocess(input_mail)
    vectorized_input = vectorizer.transform([processed_mail])
    prediction = model.predict(vectorized_input)

    result = "Đây là **email bình thường** (Ham)." if prediction[0] == 1 else "Đây là **email spam**!"

    return render_template('index.html', prediction=result, email=input_mail)

if __name__ == '__main__':
    app.run(debug=True)
