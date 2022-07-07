from tkinter import *
import tkinter as tk
import tkinter.font as tkFont
from PIL import ImageGrab, Image
import numpy as np
import cv2
import win32gui
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

image_folder = "img/"
model = load_model('models/emnist.h5')
word_dict = {10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',
            19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
            28:'S',29:'T',30:'U',31:'V',32:'W',33:'X', 34:'Y',35:'Z',36:'a',
            37:'b',38:'c',39:'d',40:'e',41:'f',42:'g',43:'h',44:'i',45:'j',
            46:'k',47:'l',48:'m',49:'n',50:'o',51:'p',52:'q',53:'r',54:'s',
            55:'t',56:'u',57:'v',58:'w',59:'x',60:'y',61:'z'}

def get_contour_precedence(contour, rows):
    tolerance_factor = 1
    origin = cv2.boundingRect(contour)
    return ((origin[0] // tolerance_factor) * tolerance_factor) * rows + origin[1]

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.result = ""

        self.title("Writing Space")
        self.canvas = tk.Canvas(self, width = 1000, height = 1000, bg = "white", cursor="cross")
        self.label = tk.Label(self, text = "Result", font = ("Arial",25))
        self.button_clear = tk.Button(self, text = "Clear Writing Space", command = self.clear_all, width = 20, font = (tkFont.Font(size=15)))
        self.button_clear2 = tk.Button(self, text="Convert to text" , command = self.Writing_To_Text, width = 20, font = (tkFont.Font(size=15)))

        self.canvas.grid(row=1, column=0,columnspan=2, pady=2, sticky='nsew')
        self.label.grid(row = 0, column = 0, columnspan = 2, pady=10)
        self.button_clear.grid(row=2, column=0, pady=2)
        self.button_clear2.grid(row = 2, column = 1,pady = 2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
    
    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text='')
        self.result = ""

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=5
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

    def Writing_To_Text(self):
        self.result = ''
        filename = f'img_0.png'
        
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        ImageGrab.grab(rect).save(image_folder + filename)

        image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contour0 = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt,3,True) for cnt in contour0[0]]
        contours.sort(key=lambda x:get_contour_precedence(x, image.shape[0]))
        
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(image,(x,y), (x+w, y+h), (255,0,0),1)
            character = th[y:y+h,x:x+w]
            resized_character = cv2.resize(character, (18, 18))
            padded_character = np.pad(resized_character, ((5, 5), (5, 5)), "constant", constant_values=0)

            character = padded_character.reshape(1, 28, 28, 1)
            character = character / 255.0

            pred_char = model.predict([character])[0]
            final_pred_char = np.argmax(pred_char)
            if(final_pred_char>=10):
                final_pred_char = word_dict[final_pred_char]


            data = str(final_pred_char)
            self.result+=data

        self.label.configure(text="You wrote: " + self.result, font = ("Arial",25))
        
app = App()
mainloop()




