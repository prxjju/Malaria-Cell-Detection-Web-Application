import os

from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

import cv2
import numpy
import tensorflow

path = "C:/Users/asus/Desktop/Projects/Malaria Cell Detection Project/"
model = tensorflow.keras.models.load_model(path)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index2')
def webapp():
    return render_template('index2.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == "POST":

        empty_list = []

        img = request.files['myimage']
        basepath = os.path.dirname(__file__)
        image_path = os.path.join(basepath, secure_filename(img.filename))
        img.save(image_path)

        image = cv2.imread(image_path)
        img_resized = cv2.resize(image, dsize=(28,28))
        img_scaled = img_resized/255
        empty_list.append(img_scaled)
        array = numpy.array(empty_list)

        output = model.predict(array)
        result = numpy.argmax(output)
        
        empty_list.pop()
        
        #if result==1:
            #string =  "Prediction-> 1 || Conclusion-> You are Infected! "
            #return string    
        #else:
            #string = "Prediction-> 0 || Conclusion-> You are not Infected! "
            #return string
        
        return render_template('index2.html',data=result)
    return None

if __name__ == "__main__":
    app.run(debug = True) 