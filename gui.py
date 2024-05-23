import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, 'r') as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel('model_a1.json', 'model.weights.h5')

EMOTIONS_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def Detect(file_path):
    print(f"Detect function called with file_path: {file_path}")
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    print(f"Faces detected: {len(faces)}")
    try:
        if len(faces) == 0:
            label1.configure(foreground='#011638', text='No face detected.')
            return
        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            roi = roi.astype('float') / 255.0  # Normalize the image
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            print(f"ROI shape: {roi.shape}")
            pred = model.predict(roi)
            print(f"Prediction probabilities: {pred}")
        pred_emotion = EMOTIONS_LIST[np.argmax(pred)]
        print(f'Predicted Emotion: {pred_emotion}')
        label1.configure(foreground='#011638', text=pred_emotion)
    except Exception as e:
        print(f"Error in detection: {e}")
        label1.configure(foreground='#011638', text='Unable to detect.')

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    global file_path
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(size=((top.winfo_width()//2.3), (top.winfo_height()//2.3)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error in uploading image: {e}")
        label1.configure(foreground='#011638', text='Unable to load image.')

upload = Button(top, text='Upload Image', command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
heading = Label(top, text='Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()