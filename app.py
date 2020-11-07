import nltk
import re
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
model = keras.models.load_model("model.h5")
encoder = pickle.load(open("Encoder.pkl","rb"))

nltk.download('stopwords')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ["POST"])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        ps = PorterStemmer()
        corpus = []
        review = re.sub('[^a-zA-Z]',' ',data)
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        voc_size = 5000
        onehot_message = [encoder(words,voc_size) for words in corpus]
        
        sent_length = 20
        Padding_messages = pad_sequences(onehot_message,padding="pre",maxlen=sent_length)

        vect = Padding_messages.toarray()
        vect.resize(1,20)
        my_prediction = model.predict(vect)
        return render_template('result.html',prediction = my_prediction)



    from flask import render_template

 

if __name__ == "__main__":
    app.run(debug=True)

