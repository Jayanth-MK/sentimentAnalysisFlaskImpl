import re
import string

from flask import *
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from utils.helperFunctions import *


app = Flask(__name__)


@app.route('/', methods =['GET', 'POST'])
def welcome():
    return render_template('home.html')

@app.route('/predict', methods =['POST'])
def predict():
    data = request.form['tweet']
    data = processData(data)
    data = vetorize(data)
    print(data)
    predictions = prediction(data)
    print(predictions)
    return render_template('index.html',predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)


