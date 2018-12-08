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
import random 
from yargy import Parser, rule, and_
from yargy.predicates import gram, is_capitalized, dictionary
sys.path.insert(0, '..')
from Common import preprocessing
classes_map = {'DOC':0, 'ENTER':1, 'ORG':2, 'PRIV':3, 'RANG':4, 'HOST':5}
classes_map_greet = {'QUE':0, 'GREET':1}
WAS_GREETING = False
idx_to_intent = {0:'DOC', 1:'ENTER', 2:'ORG', 3:'PRIV', 4:'RANG', 5:'HOST'}

print("Загрузка моделей")
vectorizer = pickle.load(open("../bin/vectorizer", 'rb'))
vectorizer_greet = pickle.load(open("../bin/vectorizer_greet", 'rb'))
log_reg = pickle.load(open("../bin/log_reg", 'rb'))
log_reg_greet = pickle.load(open("../bin/log_reg_greet", 'rb'))

def fallback(text):
    return "I dont understand, sorry. Can you reask in different way please"

def get_subintent(preprocessed,intent):
    data = None
    with open("../knowledge base/en/{0}.json".format(intent)) as f:
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
        return fallback(preprocessed)

def chit_chat(preprocessed):
    preprocessed = str(preprocessed)
    data = None
    with open("../knowledge base/en/CHITCHAT.json") as f:
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
    print("Вероятности subgreeting",probas)
    if any(list(probas.values())) > 0.0:
        subintent = max(probas.items(), key=operator.itemgetter(1))[0]
        return data[subintent]['response'][0]
    else:
        return fallback(preprocessed)  

def get_answer(raw_text,WAS_GREETING):
    preprocessed = preprocessing.preprocess_eng_greetings_list([raw_text])
    print("== Продобработанный текст:",preprocessed)
    
    #классифицируем это вопрос по делу или "как дела че делаешь"
    v_ = vectorizer_greet.transform(preprocessed)
    probas = log_reg_greet.predict_proba(v_)
    print("== Вероятности greeting или нет",probas[0])
    
    if not WAS_GREETING:
        #классифицируем greeting или нет
        WAS_GREETING = True
        if probas[0][0] < probas[0][1]+0.1: #если Greeting
            answer = chit_chat(raw_text)
            print("== Ответ: ",answer)
            #os.system("echo "" " + answer + " "" | RHVoice-test -p slt")
            return answer
    else:
        if probas[0][0] < probas[0][1]-0.7: #если Greeting
            answer = chit_chat(raw_text)
            print("== Ответ: ",answer)
            #os.system("echo "" " + answer + " "" | RHVoice-test -p slt")
            return answer

    #если по делу
    v = vectorizer.transform(preprocessed)
    probas = log_reg.predict_proba(v)
    print("== Вероятности интентов",probas[0])

    if max(probas[0])<0.43:
        answer = fallback(preprocessed)
        print("== Ответ: ",answer)
        return answer
        #os.system("echo "" " + answer + " "" | RHVoice-test -p slt")
    else:
        if list(probas[0]).index(max(probas[0])) == 3: #объединяем интенты (тк некорректна выборка)
            intent = 2
        else:
            intent = idx_to_intent[np.argmax(probas[0])]
        answer = get_subintent(str(preprocessed),intent)
        print("== Ответ: ",answer)
        return answer
        #os.system("echo "" " + answer + " "" | RHVoice-test -p slt")


