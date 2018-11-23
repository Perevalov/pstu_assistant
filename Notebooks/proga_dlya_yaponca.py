import os,pyaudio,math,wave,audioop,sys
from tkinter import *
from math import sin, cos
from collections import deque
from pocketsphinx.pocketsphinx import *

clicks = 0
x = 0
flag = True


class SpeechDetector:
    def __init__(self):

        # Конфигурация микрофона
        self.CHUNK = 1024  # CHUNKS - число байт, считываемое каждый раз с микрофона
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # кол-во каналов 1-моно 2-стерео
        self.RATE = 16000  # частота
        self.SILENCE_LIMIT = 2
        self.PREV_AUDIO = 2
        self.THRESHOLD = 3500
        self.num_phrases = -1

        # Конфигурируем декодер
        config = Decoder.default_config()
        config.set_string('-hmm', '/home/alex/diploma/models/acoustic/en-en')
        config.set_string('-lm', '/home/alex/diploma/models/lm/en.lm.bin')
        config.set_string('-dict', '/home/alex/diploma/models/dic/en.dict')

        # Создаем декодер
        self.decoder = Decoder(config)

    def save_speech(self, data, p):
        # сохраняем аудио-поток в файл

        waveFile = wave.open("File.wav", 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(p.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(data))
        waveFile.close()

        return 'File.wav'

    def decode_phrase(self, wav_file):
        # декодируем из файла в массив н-лучших

        self.decoder.start_utt()
        stream = open(wav_file, "rb")
        while True:
            buf = stream.read(1024)
            if buf:
                self.decoder.process_raw(buf, False, False)
            else:
                break
        self.decoder.end_utt()
        words = []
        [words.append(seg.word) for seg in self.decoder.seg()]
        return [n.hypstr for n in self.decoder.nbest()]

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        audio2send = []
        rel = int(self.RATE / self.CHUNK)
        slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)

        # Прикрепляем спереди аудио длиной 2сек
        prev_audio = deque(maxlen=self.PREV_AUDIO * rel)
        started = False

        while True:

            cur_data = stream.read(self.CHUNK)
            slid_win.append(math.sqrt(abs(audioop.avg(cur_data, 4))))

            if sum([x > self.THRESHOLD for x in slid_win]) > 0:
                if started == False:
                    started = True
                audio2send.append(cur_data)

            elif started:
                print("Конец записи, идёт распознавание")
                #self.speak(" Конец записи, идёт распознавание ")
                filename = self.save_speech(list(prev_audio) + audio2send, p)
                r = self.decode_phrase(filename)
                #self.speak(phrase)
                os.remove(filename)
                stream.stop_stream()
                stream.close()
                p.terminate()

                return str(r[0])
                # Reset all
                started = False
                slid_win = deque(maxlen=self.SILENCE_LIMIT * rel)
                prev_audio = deque(maxlen=self.PREV_AUDIO * rel)
                audio2send = []
                print("Слушаю ...")
                #self.speak(" Слушаю ")
            else:
                prev_audio.append(cur_data)

        print("* Закончили слушать")
        stream.stop_stream()
        stream.close()
        p.terminate()

def click_button():
    global flag,x
    flag = True
    x = 0
    sd = SpeechDetector()
    #sd.setup_mic()
    s = sd.run()
    print_dot()
def click_button_stop():
    global flag
    flag = False
    cvs.delete("all")

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
