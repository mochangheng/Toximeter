from flask import Flask,request
app = Flask(__name__)

from flask_cors import CORS
CORS(app)
import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import re
import pickle
import pickle
from aylienapiclient import textapi
import json
from watson_developer_cloud import ToneAnalyzerV3
import numpy as np

client = textapi.Client("19448172", "24cba108f6f2ab7de91f826ff1b30835")

tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='IdQFvGyHgBd1CdfCojl99N8feqafZn7NbwhlHPJNrSM9',
    url='https://gateway.watsonplatform.net/tone-analyzer/api'
)

with open('tokenizer.txt','rb') as f:
    tokenizer = pickle.load(f)
    f.close()
import string
import spacy
from spacy.lang.en import English

nlp = spacy.load('en')
with open('english_stopwords.txt','r') as f:
    raw = f.read()
    f.close()

STOPLIST = raw.split('\n')
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "“", "”"]

def create_model():
    global model
    model = load_model('version-2.h5')
    model._make_predict_function()
def isWord(token):
    if token in STOPLIST or token in SYMBOLS:
        return False
    exp = "^[a-zA-Z']*$"
    pattern = re.compile(exp)
    if pattern.search(token):
        return True
    else:
        return False

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def tokenizeText(sample):
    tokens = nlp(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if isWord(tok)]
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens

def process_text(text):
    text = cleanText(text)
    text = tokenizeText(text)
    text = " ".join(text)
    return text


@app.route('/',methods=['GET'])
def analyze():
    text = request.args.get('text')
    if not text:
        return 'Missing parameter "Text"'

    x = process_text(text)
    x = tokenizer.texts_to_sequences([x])
    x = pad_sequences(x, maxlen=100)
    toxicity = model.predict(x)

    sentiment = client.Sentiment({'text': text})
    if sentiment['polarity']=='negative':
        sentiment['polarity_confidence'] = 1-sentiment['polarity_confidence']

    tone_analysis = tone_analyzer.tone(
        {'text': text},
        'application/json'
    ).get_result()

    tones = tone_analysis['document_tone']['tones']
    if len(tones)>0:
        max_i = 0
        max_score = 0
        for i,element in enumerate(tones):
            if element['score']>max_score:
                max_i = i
                max_score = element['score']
        tone = {'tone':tone_analysis['document_tone']['tones'][max_i]['tone_id'],'score':np.round(tone_analysis['document_tone']['tones'][max_i]['score']*100)}
    else:
        tone = {'tone':'Neutral','score':100}
    #tone = {'tone':tone_analysis['document_tone']['tones'][-1]['tone_id'],'score':tone_analysis['document_tone']['tones'][-1]['score']}
    #tone = tone_analysis['document_tone']['tones'][-1]['tone_id']
    results=dict(toxicity=np.round(toxicity[0][0]*100),sentiment=np.round(sentiment['polarity_confidence']*100),tone=tone)
    return json.dumps(results)

if __name__ == "__main__":
    create_model()
    app.run(host='0.0.0.0',port=8000)
