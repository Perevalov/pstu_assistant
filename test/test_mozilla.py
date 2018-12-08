import os,pyaudio,math,wave,audioop,sys
from tkinter import *
from math import sin, cos
from collections import deque
from pocketsphinx.pocketsphinx import *
import threading, main
from deepspeech.model import Model
import numpy as np
import scipy.io.wavfile as wav

question = ''

class SpeechDetector:
    def __init__(self):

        # Конфигурация микрофона
        self.CHUNK = 1024  # CHUNKS - число байт, считываемое каждый раз с микрофона
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1 #кол-во каналов 1-моно 2-стерео
        self.RATE = 16000 #частота
        self.SILENCE_LIMIT = 2
        self.PREV_AUDIO = 2
        self.THRESHOLD = 3500
        self.num_phrases = -1
        
        model = Model('/home/alex/models/output_graph.pb', 26, 9, '/home/alex/models/alphabet.txt', 500)
        model.enableDecoderWithLM('/home/alex/models/alphabet.txt', 
                                  '/home/alex/models/lm.binary', 
                                  '/home/alex/models/trie', 1.50,1, 2.10)
        self.model = model
        print('decoder ready')
        
        
    def save_speech(self, data, p):

        waveFile = wave.open("File.wav", 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(p.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(data))
        waveFile.close()

        return 'File.wav'

    def decode_phrase(self, wav_file):
        #декодируем из файла в массив н-лучших

        fs, audio = wav.read(wav_file)
        return self.model.stt(audio, fs)


    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        audio2send = []
        cur_data = ''  # current chunk of audio data
      
        while flag == True:
            cur_data = stream.read(self.CHUNK)
            audio2send.append(cur_data)

        filename = self.save_speech(audio2send, p)
        stream.stop_stream()
        stream.close()
        p.terminate()
        r = self.decode_phrase(filename)
        text = str(r)
        global question
        question = text
        main.get_answer(question,True)
        global w
        w.insert(1.0,question)
        print(text)

clicks = 0
x = 0
flag = False
sd = SpeechDetector()
text = ''

def click_button():
    global flag,x,sd
    
    x = 0
    flag = True
    t = threading.Thread(target=sd.run)
    t.start()

def click_button_stop():
    global flag,sd,w
    flag = False
    cvs.delete("all")
    w.insert(1.0, text)

def print_dot():
    global x, flag
    y1 = sin(x)
    y2 = cos(x)
    if flag == True:
        cvs.create_oval(15 * x + 10, 15 * y1 + 25, 15 * x + 10, 15 * y1 + 25, width=1, outline="red")
        cvs.create_oval(15 * x + 10, 15 * y2 + 25, 15 * x + 10, 15 * y2 + 25, width=1, outline="blue")
        x += 0.03
        if 15 * x + 10 > 600:
            x = 0
            cvs.delete("all")

        root.after(2, print_dot)


root = Tk()
root.geometry("600x300")
root.configure(background='white')

btn = Button(root, text="Начать запись", background="#555", foreground="#ccc",
             padx="20", pady="8", font="16", command=click_button, width = 100)
btn.pack()

btn = Button(root, text="Остановить запись", background="#555", foreground="#ccc",
             padx="20", pady="8", font="16", command=click_button_stop, width = 100)
btn.pack()

frameLabel = Frame(root,width=600,height=150,bg='white')
frameLabel.pack()
w = Text( frameLabel, wrap='word', font='Arial 12 italic')
w.pack()


cvs = Canvas(root, width=600, height=50, bg="white")
cvs.place(x=0, y=250)
root.mainloop()
