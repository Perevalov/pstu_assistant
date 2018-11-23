import os,pyaudio,math,wave,audioop,sys
from tkinter import *
from math import sin, cos
from collections import deque
from pocketsphinx.pocketsphinx import *

clicks = 0
x = 0
flag = True
cur_data = 0
audio2send = []
p = pyaudio.PyAudio()

config = Decoder.default_config()
config.set_string('-hmm', '/home/alex/diploma/models/acoustic/en-en')
config.set_string('-lm', '/home/alex/diploma/models/lm/en.lm.bin')
config.set_string('-dict', '/home/alex/diploma/models/dic/en.dict')
decoder = Decoder(config)
stream = None

def save_speech(data, p):
    # сохраняем аудио-поток в файл

    waveFile = wave.open("File.wav", 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(16000)
    waveFile.writeframes(b''.join(data))
    waveFile.close()

    return 'File.wav'

def decode_phrase(wav_file):
    decoder.start_utt()
    stream = open(wav_file,"rb")
    while True:
        buf = stream.read(1024)
        if buf:
           decoder.process_raw(buf,False,False)
        else:
            break
        decoder.end_utt()
        words = []
        [words.append(seg.word) for seg in decoder.seg()]
        return [n.hypstr for n in decoder.nbest()]

def click_button():
    global flag,x,cur_data,p,stream
    flag = True
    x = 0	
    #print_dot()

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    print("Старт")
    cur_data = stream.read(1024)


def click_button_stop():
    global flag,audio2send,p,stream
    flag = False
    cvs.delete("all")
    print(cur_data)
    audio2send.append(cur_data)
    filename = save_speech(audio2send, p)
    r = decode_phrase(filename)
    print("* Закончили слушать")
    stream.stop_stream()
    stream.close()
    p.terminate()


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

#poetry = "Вот мысль, которой весь я предан, Итог всего, что ум скопил. Лишь тот, кем бой за жизнь изведан, Жизнь и свободу заслужил."

frameLabel = Frame(root,width=600,height=150,bg='white')
frameLabel.pack()
w = Text( frameLabel, wrap='word', font='Arial 12 italic')
#w.insert( 1.0, poetry )
w.pack()


cvs = Canvas(root, width=600, height=50, bg="white")
cvs.place(x=0, y=250)
root.mainloop()
