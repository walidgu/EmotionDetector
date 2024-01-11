from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import cv2
import pickle

#app name
app = Flask(__name__)

#Load model
model = load_model('model.h5', compile=False)

#load label dictionnary
with open('labels.pickle', 'rb') as handle:
    dictionary_labels = pickle.load(handle)

#Image upload
target_img = os.path.join(os.getcwd() , 'static/images')

#Get camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Setting camera parameters
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Parameters for face detection
coord_x, coord_y = 0, 0
x_2, y_2 = 0, 0

#Face detector for camera
face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

#Text parameters on camera
text = "Waiting"
# font 
font = cv2.FONT_HERSHEY_SIMPLEX                
# fontScale 
fontScale = 2        
# Red color in BGR 
color = (0, 255, 0)         
# Line thickness of 2 px 
thickness = 4        

#Home Page
@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def gen_frames():  # generate frame by frame from camera
    counter = 0
    while True:
        #Read image
        success, frame = cap.read() 
        if success:
            try:
                #Turn image to gray
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_for_model = gray_image.copy()

                #Detect face
                face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))   

                for (x, y, w, h) in face:
                    coord_x, coord_y = x, y
                    x, y = x-50, y-50
                    w, h = w + 50, h + 50
                    x_2, y_2 = int((x + w)*1.2), int((y + h)*1.2)
                    #Draw rectangle on face
                    frame = cv2.rectangle(frame, (x, y), (x_2, y_2), (0, 255, 0), 4)
                    break
                        
                #Cropping face to give to the emotion detector model
                img_rgb = cv2.cvtColor(img_for_model, cv2.COLOR_BGR2RGB)
                cropped_image = img_rgb[coord_y:y_2,coord_x:x_2]

                #We run model every 30 images to avoid lag
                if counter % 30 == 0:
                    try:
                        #Resize cropped image to give it as an input
                        image_x = cv2.resize(cropped_image, (48,48), interpolation = cv2.INTER_AREA)
                        img_gray_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
                        img_flatten_x = img_gray_x.flatten()
                        img_shaped_x = img_flatten_x.reshape(48, 48, 1).astype('float32')
                        img_normalized_x = img_shaped_x /255.
                        y_perso = model.predict(np.array([img_normalized_x]))
                        # Detected emotion
                        text = dictionary_labels[np.argmax(y_perso, axis=1).tolist()[0]].capitalize() 
                    except ValueError:
                        pass

                #Coord of text just above the rectangle            
                org = (coord_x - 30, coord_y - 60)  
                #Adding text
                frame = cv2.putText(frame, text, org, font, fontScale,  
                                color, thickness, cv2.LINE_AA, False) 
                #Incrementing counter
                counter += 1
                
                #Serialize Image to give it in the HTTP response
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict',methods=['GET','POST'])
def predict():
    print(request.method)
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = 'static/'+filename
            file.save(file_path)
            img = cv2.imread(file_path) 

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            for (x, y, w, h) in face:
                    coord_x, coord_y = x, y
                    x, y = x-50, y-50
                    w, h = w + 50, h + 50
                    x_2, y_2 = int((x + w)*1.2), int((y + h)*1.2)
                    break
            
            print('la')
            img_rgb = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)
            cropped_image = img_rgb[coord_y:y_2,coord_x:x_2]
            cv2.imshow('video', cropped_image)     
            image_x = cv2.resize(cropped_image, (48,48), interpolation = cv2.INTER_AREA)
            img_gray_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2GRAY)
            img_flatten_x = img_gray_x.flatten()
            img_shaped_x = img_flatten_x.reshape(48, 48, 1).astype('float32')
            img_normalized_x = img_shaped_x /255.
            y_perso = model.predict(np.array([img_normalized_x]))
            # Detected emotion
            predicted_emotion = dictionary_labels[np.argmax(y_perso, axis=1).tolist()[0]].capitalize()
            print(predicted_emotion)
            return render_template('predict.html', emotion=predicted_emotion, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"
    return 'Error'

        
if __name__ == '__main__':
    app.run()