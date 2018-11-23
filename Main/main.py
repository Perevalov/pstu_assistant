from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import json
import operator
import numpy as np
import sys
import os
from yargy import Parser, rule, and_
from yargy.predicates import gram, is_capitalized, dictionary
sys.path.insert(0, '..')
from Common import preprocessing
classes_map = {'DOC':0, 'ENTER':1, 'ORG':2, 'PRIV':3, 'RANG':4, 'HOST':5}

idx_to_intent = {0:'DOC', 1:'ENTER', 2:'ORG', 3:'PRIV', 4:'RANG', 5:'HOST'}

df = pd.read_csv('..//Data//translation.csv', delimiter=';', engine='python',encoding='utf8')

questions = np.array(df.question)
questions = preprocessing.preprocess_eng_list(questions)

vectorizer = TfidfVectorizer(min_df=3,ngram_range=(1,1))
X = vectorizer.fit_transform(questions)
print("Размерность:",X.shape)

classes = np.array(df['class'])
y = list(map(lambda x: classes_map[x],classes))

log_reg = OneVsRestClassifier(LogisticRegression(random_state=0,C=10,solver='lbfgs',)).fit(X, y)

def get_subintent(preprocessed,intent):
    data = None
    with open("/home/alex/pstu_assistant/knowledge base/en/{0}.json".format(intent)) as f:
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
    subintent = max(probas.items(), key=operator.itemgetter(1))[0]
    
    return data[subintent]['response'][0]

def get_answer(raw_text):
    preprocessed = preprocessing.preprocess_eng_list([raw_text])
    print("Продобработанный текст:",preprocessed)
    v = vectorizer.transform(preprocessed)
    probas = log_reg.predict_proba(v)
    print("Вероятности интентов",probas[0])
    if max(probas[0])<0.6:
        answer = "I dont understand, sorry"
        print("Ответ: ",answer)
        os.system("echo "" " + answer + " "" | RHVoice-test")
    else:
        intent = idx_to_intent[np.argmax(probas[0])]
        answer = get_subintent(str(preprocessed),intent)
        print("Ответ: ",answer)
        os.system("echo "" " + answer + " "" | RHVoice-test")


#get_answer("Hello! I have a question: is it Possible to send the original passport by courier?")
get_answer("Can you please tell me what exams are required for admission to the ASU?")
get_answer("What is the most efficient way to loop through dataframes with pandas?")