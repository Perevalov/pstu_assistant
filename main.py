import tensorflow as tf
import pickle
import os,json,sys
from os.path import  join
from keras.models import Model, load_model
from assistant import Assistant, Chatterbox,utils

classes_map = {'DOC':0, 'ENTER':1, 'ORG':2, 'RANG':3, 'HOST':4,'GREET':5}
classes_map_greet = {'QUE':0, 'GREET':1}
idx_to_intent = {0:'DOC', 1:'ENTER', 2:'ORG',3:'RANG', 4:'HOST',5:'GREET'}

print("Загрузка моделей")

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

lang_vectorizer = pickle.load(open(join(PROJECT_PATH,"bin/lang_vectorizer"), 'rb'))
lang_classifier = pickle.load(open(join(PROJECT_PATH,"bin/lang_classifier"), 'rb'))

ru_vectorizer = pickle.load(open(join(PROJECT_PATH,"bin/ru_vectorizer"), 'rb'))
ru_classifier = pickle.load(open(join(PROJECT_PATH,"bin/ru_classifier"), 'rb'))

en_vectorizer = pickle.load(open(join(PROJECT_PATH,"bin/en_vectorizer"), 'rb'))
en_classifier = pickle.load(open(join(PROJECT_PATH,"bin/en_log_reg"), 'rb'))

models = {'lang_vectorizer':lang_vectorizer,'lang_classifier':lang_classifier, 'ru_vectorizer':ru_vectorizer,
          'ru_classifier': ru_classifier,'en_vectorizer':en_vectorizer,'en_classifier': en_classifier}

tk = pickle.load(open(join(PROJECT_PATH,"bin/ru_tokenizer_dssm"), "rb"))

model = load_model(join(PROJECT_PATH, "bin/rnn.model"),
                   custom_objects={"MaskedGlobalMaxPooling1D": utils.MaskedGlobalMaxPooling1D,
                                   "tf": tf,
                                   "dummy_loss": utils.dummy_loss})

q_model = Model(inputs=[model.get_layer("QInput").input],
                outputs=[model.get_layer("QL2norm").output])
a_model = Model(inputs=[model.get_layer("AInput").input],
                outputs=[model.get_layer("AL2norm").output])

with open(join(PROJECT_PATH,"data/conversations.txt")) as f:
    content = f.readlines()
content = [x.strip() for x in content]

questions = []
answers = []

for i in range(len(content)-3):
    if len(json.loads(content[i]))%2 == 0 and len(json.loads(content[i])) > 0:
        for j in range(0,len(json.loads(content[i])),2):
            questions.append(json.loads(content[i])[j]['text'])
            answers.append(json.loads(content[i])[j+1]['text'])

chatterbox = Chatterbox(tk, q_model, a_model)
chatterbox.fit(answers)

del questions
del answers
del content

print(chatterbox.answer("Привет. Как дела?"))

assistant = Assistant(models,chatterbox,idx_to_intent)

