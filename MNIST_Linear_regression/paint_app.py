from tkinter import *
from tkinter.colorchooser import askcolor
import numpy as np
from PIL import Image, ImageGrab
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import random
from train_optimize import *
import threading

class Paint(object):

    DEFAULT_PEN_SIZE = 35
    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)

        self.eraser_button = Button(self.root, text='classify number', command=self.classify_number)
        self.eraser_button.grid(row=0, column=2)

        self.c = Canvas(self.root, bg='black', width=600, height=600)
        self.c.grid(row=1, columnspan=3)

        self.setup()

        self.progress_bar = self.c.create_rectangle(40, 40, 40, 80, fill='green', width=0, tags="progress_bar")
        self.progress_text = self.c.create_text(10, 25, text="0%", anchor='nw', fill='white', tags="progress_text")

        self.c.create_text(50, 10, text="Traning model", anchor='nw', fill='white')
        self.c.create_text(50, 100, text="Cost:", anchor='nw', fill='white')
        self.c.create_text(90, 100, text="0", anchor='nw', fill='white', tags="costs")
 
        
        self.model_thread = threading.Thread(target=self.create_model)
        self.model_thread.start()

        self.root.mainloop()
    
    def update_progress_bar(self, percentage):  
        new_width = 40 + (percentage / 100) * (self.c.winfo_width() - 80)
        self.c.coords("progress_bar", 40, 40, new_width, 80)
        self.c.itemconfig("progress_text", text=f"{percentage}%")
        self.c.update()
    
    def create_model(self):

        data = np.load('./mnist_bin_zeros_ones.npz')
        X, y = data['X'], data['y']

        unique_integers = random.sample(range(0, 14780), 780)

        X_test = np.zeros((780, 784))
        y_test = np.zeros((780, 1))

        for i in range(len(unique_integers)):
            X_test[i,:]=X[unique_integers[i], :]
            y_test[i] = y[unique_integers[i]]

        X = np.delete(X, unique_integers, axis=0)
        y = np.delete(y, unique_integers)

        w, b = initialize_parameters(X.shape[1])

        for i in range(2000):
            w, b, cost = optimize(w, b, X.T, y, 0.003, print_cost = False)
            self.update_progress_bar(round((i / 2000) * 100, 2))
            if i % 100 == 0:
                self.c.itemconfig("costs", text=f"{cost}")
            
        params = {"w": w,
                  "b": b}

        trained_w = params['w']
        trained_bias = params['b']

        trained_w_and_bias = np.vstack((trained_w, np.array([[trained_bias]])))

        np.savetxt("weights.csv", trained_w_and_bias, delimiter='.', fmt='%.10f')

        self.c.delete("all")

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = 15
        paint_color = 'black' if self.eraser_on else self.color
        self.line_width = 100 if self.eraser_on else 35
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y
    

    def classify_number(self):
        # Get the canvas content as an image
        x0 = self.c.winfo_rootx()
        y0 = self.c.winfo_rooty()
        x1 = x0 + self.c.winfo_width()
        y1 = y0 + self.c.winfo_height()
        img = ImageGrab.grab(bbox=(x0, y0, x1, y1))

        # Resize the image to 600x600 pixels
        img = img.resize((28, 28), Image.ANTIALIAS)
        img.save('canvas_image.png')

        picture_to_feed_model = np.zeros((28, 28))

        # Create a list comprehension to extract pixel_color values and transpose the matrix
        picture_to_feed_model = np.array([[img.getpixel((i, j))[0] for j in range(28)] for i in range(28)]).T

        # Apply thresholding using a list comprehension
        threshold = 80
        picture_to_feed_model = np.array([[1 if pixel_color >= threshold else 0 for pixel_color in row] for row in picture_to_feed_model])

        # Flatten the 2D array to a 1D array
        picture_to_feed_model = picture_to_feed_model.ravel()

        df = pd.read_csv('weights.csv', header=None)
        data_array = df.values

        z = np.dot(data_array[:-1].T, picture_to_feed_model)+data_array[-1]
        A = sigmoid(z)

        # Threshold 40 is just a guess
        # Must do some investegation to find correct threshold
        guess = 100
        if A < 0.40:
            guess = 0
        else:
            guess = 1

        self.c.delete("result_text")

        # Display the value of A on the canvas
        result_text = f"Model guess: {int(guess):.4f}"  # You can format it as needed
        self.c.create_text(10, 10, text=result_text, anchor='nw', fill='white', tags="result_text")
    
        
    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    Paint()