from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import operator
import numpy as np
import pickle
import sys
import os
from os.path import dirname, join, abspath
import random 
from yargy import Parser, rule, and_
from yargy.predicates import gram, is_capitalized, dictionary
sys.path.insert(0, '..')
from common import preprocessing
from chatter import chatter
classes_map = {'DOC':0, 'ENTER':1, 'ORG':2, 'PRIV':3, 'RANG':4, 'HOST':5,'GREET':6}
classes_map_greet = {'QUE':0, 'GREET':1}
WAS_GREETING = False
idx_to_intent = {0:'DOC', 1:'ENTER', 2:'ORG', 3:'PRIV', 4:'RANG', 5:'HOST',6:'GREET'}

print("Загрузка моделей")

PROJECT_PATH = "/home/alex/pstu_assistant"

lang_vectorizer = pickle.load(open(join(PROJECT_PATH,"bin/lang_vectorizer"), 'rb'))
lang_classifier = pickle.load(open(join(PROJECT_PATH,"bin/lang_classifier"), 'rb'))

vectorizer = pickle.load(open(join(PROJECT_PATH,"bin/ru_vectorizer"), 'rb'))
log_reg = pickle.load(open(join(PROJECT_PATH,"bin/ru_log_reg"), 'rb'))

en_vectorizer = pickle.load(open(join(PROJECT_PATH,"bin/en_vectorizer"), 'rb'))
en_log_reg = pickle.load(open(join(PROJECT_PATH,"bin/en_log_reg"), 'rb'))



def fallback(raw_text,lang):
    if lang == 'ru':
        answers = chatter.chatter.answer(raw_text)
        a = ''
        for i in range(len(answers)):
            i = random.randint(0,len(answers)-1)
            if len(answers[i]) > 2:
                return answers[i]
        return "Простите, я Вас не поняла"
    else:
        return "Sorry, I don't understand you"

def get_subintent(raw_text,preprocessed,intent,lang):
    data = None
    with open(join(PROJECT_PATH,"knowledge base/{0}/{1}.json").format(lang,intent)) as f:
        data = f.read()
    data = json.loads(data)
    probas = {}
    for subintent in data:
        count = 0
        for keyword in data[subintent]['keywords']:
            RULE = rule(dictionary({keyword}))
            parser = Parser(RULE)
            for match in parser.findall(preprocessed):
                count = count + len(match.tokens)
        probas[subintent] = count/len(data[subintent]['keywords'])
    print("Вероятности субинтентов",probas)
    if any(list(probas.values())) > 0.0:
        subintent = max(probas.items(), key=operator.itemgetter(1))[0]
        return data[subintent]['response'][0]
    else:
        return fallback(raw_text,lang)

def classify_lang(raw_text):
    preprocessed = preprocessing.preprocess_multilang_list([raw_text])
    v = lang_vectorizer.transform(preprocessed)
    probas = lang_classifier.predict_proba(v)
    
    if probas[0][0] > probas[0][1]:
        print('RUS')
        return 'RUS'
    else:
        print('ENG')
        return 'ENG'
    
def get_answer(raw_text):
    
    with open(join(PROJECT_PATH,"log/dialog"), "a") as myfile:
        
        lang = classify_lang(raw_text)
        if lang == 'RUS':
            preprocessed = preprocessing.preprocess_list([raw_text])
            print("Продобработанный текст:",preprocessed)

            v = vectorizer.transform(preprocessed)
            probas = log_reg.predict_proba(v)
            
            print("Вероятности интентов",probas[0])

            if max(probas[0])<0.6:
                answer = fallback(raw_text,'ru')
                print("Ответ: ",answer)
                myfile.write(raw_text+";"+answer+"\n")
                return answer
            else:
                if list(probas[0]).index(max(probas[0])) == 3: #объединяем интенты (тк некорректна выборка)
                    intent = 2
                else:
                    intent = idx_to_intent[np.argmax(probas[0])]
                answer = get_subintent(raw_text,str(preprocessed),intent,'ru')
                print("Ответ: ",answer)
                myfile.write(raw_text+";"+answer+"\n")
                return answer

        else:
            preprocessed = preprocessing.preprocess_eng_list([raw_text])
            print("Продобработанный текст:",preprocessed)

            v = en_vectorizer.transform(preprocessed)
            probas = en_log_reg.predict_proba(v)
            print("Вероятности интентов",probas[0])

            if max(probas[0])<0.6:
                answer = fallback(raw_text,'en')
                print("Ответ: ",answer)
                myfile.write(raw_text+";"+answer+"\n")
                return answer
            else:
                if list(probas[0]).index(max(probas[0])) == 3: #объединяем интенты (тк некорректна выборка)
                    intent = 2
                else:
                    intent = idx_to_intent[np.argmax(probas[0])]
                answer = get_subintent(raw_text,str(preprocessed),intent,'en')
                print("Ответ: ",answer)
                myfile.write(raw_text+";"+answer+"\n")
                return answer

