from os.path import join
import json
import operator
import logging
import random
import re
import numpy as np
from sklearn.neighbors import NearestNeighbors
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.insert(0, '..')
from common import preprocessing
from config import PROJECT_PATH

RUS = 'ru'
ENG = 'en'


class Assistant(object):
    """
    Класс Assistant - это сущность, которая отвечает на вопросы

    """
    def __init__(self,models,chatterbox,idx_to_intent):
        self.THRESHOLD = 0.5
        self.lang_vectorizer = models['lang_vectorizer']
        self.lang_classifier = models['lang_classifier']
        self.ru_vectorizer = models['ru_vectorizer']
        self.ru_classifier = models['ru_classifier']
        self.en_vectorizer = models['en_vectorizer']
        self.en_classifier = models['en_classifier']
        self.idx_to_intent = idx_to_intent
        self.chatterbox = chatterbox
        logging.basicConfig(filename="sample.log", level=logging.INFO)
        logging.info("Ассистент инициализирован")

    def fallback(self,raw_text,lang):
        """
        Фолбэк - срабатывает тогда, когда мы не можем определить интент пользователя.
        В большинстве случаев отвечает DSSM болталка или просто отвечает, что не понял вас.
        :param raw_text: Необработанный текст, введенный пользователем
        :param lang: Код языка
        :return: Текстовый ответ
        """

        if lang == RUS:
            print("Fallback")
            answers = self.chatterbox.answer(raw_text)
            for i in range(len(answers)):
                j = random.randint(0,len(answers)-1)
                if len(answers[j]) > 2:
                    logging.info("fallback: {0}".format(answers[j]))
                    return answers[j]
            logging.info("Простите, я Вас не поняла")
            print("Простите, я Вас не поняла")
            return "Простите, я Вас не поняла"
        else:
            logging.info("fallback: " + "Sorry, I don't understand you")
            print("fallback: " + "Sorry, I don't understand you")
            return "Sorry, I don't understand you"

    def get_subintent(self,raw_text,preprocessed,intent,lang):

        """
        Метод для определения субинтента по количеству вхождений ключевых слов

        :param raw_text: Необработанный текст, введенный пользователем
        :param preprocessed: Предобработанный текст
        :param intent: Интент, определенный на предыдущем этапе
        :param lang: Код языка
        :return:
        """

        data = None

        with open(join(PROJECT_PATH,"knowledge base/{0}/{1}.json").format(lang,intent)) as f:
            data = f.read()

        data = json.loads(data)
        probas = {}

        for subintent in data:
            count = 0
            for keyword in data[subintent]['keywords']:
                regex = re.compile(keyword)
                count = count + len(regex.findall(preprocessed))
            probas[subintent] = count/len(data[subintent]['keywords'])

        logging.info("Вероятности субинтентов {0}".format(probas))
        print("Вероятности субинтентов {0}".format(probas))

        if any(list(probas.values())) > 0.0:
            subintent = max(probas.items(), key=operator.itemgetter(1))[0]
            j = random.randint(0, len(data[subintent]['response'])-1)
            logging.info("Cубинтент {0}".format(data[subintent]['response'][j]))
            print("Cубинтент {0}".format(data[subintent]['response'][j]))
            return data[subintent]['response'][j]
        else:
            return self.fallback(raw_text,lang)

    def classify_lang(self,raw_text):
        """
        Метод для классификации языка по входному, необработанному тексту

        :param raw_text:  Необработанный текст, введенный пользователем
        :return: Код языка (Русский или английский)
        """
        preprocessed = preprocessing.preprocess_multilang_list([raw_text])
        v = self.lang_vectorizer.transform(preprocessed)
        probas = self.lang_classifier.predict_proba(v)

        if probas[0][0] > probas[0][1]:
            logging.info("RUS language detected")
            print("RUS language detected")
            return RUS
        else:
            logging.info("ENG language detected")
            print("ENG language detected")
            return ENG

    def get_answer(self,raw_text):
        """
        Главный метод, отвечающий за ответ на вопрос пользователя

        :param raw_text: Необработанный текст, введенный пользователем
        :return: Ответ в виде текста
        """

        lang = self.classify_lang(raw_text)
        if lang == RUS:
            preprocessed = preprocessing.preprocess_list([raw_text])
            logging.info("Продобработанный текст: {0}".format(preprocessed))
            print("Продобработанный текст: {0}".format(preprocessed))
            v = self.ru_vectorizer.transform(preprocessed)
            probas = self.ru_classifier.predict_proba(v)
            logging.info("Вероятности интентов: {0}".format(probas[0]))
            print("Вероятности интентов: {0}".format(probas[0]))
            if max(probas[0]) < self.THRESHOLD:
                answer = self.fallback(raw_text,RUS)
                logging.info("Ответ: "+ answer)
                print("Ответ: "+ answer)
                return answer
            else:
                intent = self.idx_to_intent[np.argmax(probas[0])]
                answer = self.get_subintent(raw_text,str(preprocessed),intent,RUS)
                logging.info("Ответ: "+answer)
                print("Ответ: "+answer)
                return answer

        else:
            preprocessed = preprocessing.preprocess_eng_list([raw_text])
            logging.info("Продобработанный текст: {0}".format(preprocessed))
            print("Продобработанный текст: {0}".format(preprocessed))
            v = self.en_vectorizer.transform(preprocessed)
            probas = self.en_classifier.predict_proba(v)
            logging.info("Вероятности интентов {0}".format(probas[0]))
            print("Вероятности интентов {0}".format(probas[0]))
            if max(probas[0]) < self.THRESHOLD:
                answer = self.fallback(raw_text,ENG)
                logging.info("Ответ: "+answer)
                print("Ответ: "+answer)
                return answer
            else:
                if list(probas[0]).index(max(probas[0])) == 3: #объединяем интенты (тк некорректна выборка)
                    intent = 2
                else:
                    intent = self.idx_to_intent[np.argmax(probas[0])]
                answer = self.get_subintent(raw_text,str(preprocessed),intent,ENG)
                logging.info("Ответ: {0}".format(answer))
                print("Ответ: {0}".format(answer))
                return answer


class Chatterbox(object):
    """
    Класс - болталка
    """
    def __init__(self, tokenizer, q_model, a_model, **kwargs):
        """
        Constructor

        Args:
            tokenizer(keras.Tokenizer): fitted instance of Keras tokenizer class
            q_model(keras.Model):       question encoding tower of DSSM in the form of Keras model
            a_model(keras.Model):       answer encoding tower of DSSM in the form of Keras model

        Return:
            self
        """

        # call the ancestor constructor
        super(Chatterbox, self).__init__(**kwargs)
        # assign passed parameters to the class fields
        self._tk = tokenizer
        self._q_model = q_model
        self._a_model = a_model

    def _preprocess(self, bank):
        """
        Preprocess the corpus of sentences into the suitable format
        to feed into the tower.

        Args:
            bank(np.ndarray): array of sentences to preprocess

        Return:
            bank_padded(np.ndarray): preprocessed and tokenized sentences
        """

        bank_processed = []
        # iterate over the corpus
        for i in range(len(bank)):
            # filter out specific symbols, punctuations,
            # transform numbers into special <num> token, put everything to lower case
            q = " ".join([w if not w.isdigit() and not bool(re.search(r'[a-zA-Z]', w))
                          else "<num>" for w in text_to_word_sequence(bank[i])])
            bank_processed.append(q)
        bank_processed = np.array(bank_processed)

        # transform filtered sentences into sequences of tokens
        bank_tokenized = self._tk.texts_to_sequences(bank_processed)

        # pad sequences to the desired length (taken from the model input dimensions)
        bank_padded = pad_sequences(bank_tokenized, maxlen=self._q_model.layers[0].input_shape[1])

        return bank_padded

    def fit(self, answers_bank):
        """
        Fit the replier model to the predefined set of answers.
        All other answers will be chosen from the passed array.

        Args:
            answers_bank(np.ndarray): array of sentences from which
                                      the model can answer

        Return:
            self
        """

        self._answers_bank = answers_bank
        # preprocess and then encode answers with the answer tower of DSMM
        _answers_bank_encoded = self._a_model.predict(self._preprocess(self._answers_bank))

        # nearest neighbors model to find the most suitable answer
        self._replier = NearestNeighbors(n_neighbors=5,
                                         metric="minkowski",
                                         p=2,
                                         n_jobs=-1,
                                         algorithm="kd_tree")
        # fir nearest neighbors model on the encoded set of answers
        self._replier.fit(_answers_bank_encoded)

        return self

    def answer(self, questions, n_answers=5):
        """
        Provide an answer to the given questions.

        Args:
            questions(str, list, np.ndarray): questions to the model
            n_answers(int):                   number of top answers to return

        Return:
            answers(np.ndarray): array of answers to the given questions
        """

        if type(questions) == str:
            questions = np.array([questions])
        elif type(questions) == list:
            questions = np.array(questions)
        elif type(questions) == np.ndarray:
            pass
        else:
            raise ValueError("Wrong format of question")

        questions = self._preprocess(questions)

        questions_encoded = self._q_model.predict(questions)

        res = self._replier.kneighbors(questions_encoded, n_neighbors=n_answers)[1]

        return self._answers_bank[res[0][0]], self._answers_bank[res[0][1]], self._answers_bank[res[0][2]]
