from flask import Flask, render_template, Response

import os
import numpy as np
import cv2
from keras.preprocessing import image
import time
from time import sleep
import argparse
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

app = Flask(__name__)

from keras.models import model_from_json
# load model
model = model_from_json(open("Emotion Analysis in a video in separate\\fer.json", "r").read())
# load weights
model.load_weights('C:\\Users\\welcome\\Downloads\\fer.h5')

faceCascade = cv2.CascadeClassifier('FaceDetection\Cascades\haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('FaceDetection\Cascades\haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('FaceDetection\Cascades\haarcascade_smile.xml')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

frame = 0

cap = cv2.VideoCapture(0) # process real time web-cam
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height


def gen_frames():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames_for_face_eye_smile_detection():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                eyes = eyeCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=5,
                    minSize=(5, 5),
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey),(ex + ew, ey + eh), (0, 255, 0), 2)

                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.5,
                    minNeighbors=15,
                    minSize=(25, 25),
                )

                for (xx, yy, ww, hh) in smile:
                    cv2.rectangle(roi_color, (xx, yy),(xx + ww, yy + hh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_for_emotion_detection_combine():
    while True:
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            img = cv2.resize(img, (640, 360))
            img = img[0:308,:]
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                
            for (x,y,w,h) in faces:
                if w > 130:
                    #highlight detected face
                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
                    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
                    
                    img_pixels = image.img_to_array(detected_face)
                    img_pixels = np.expand_dims(img_pixels, axis = 0)
                        
                    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
                        
                    predictions = model.predict(img_pixels) #store probabilities of 7 expressions
                    max_index = np.argmax(predictions[0])
                        
                    #background of expression list
                    overlay = img.copy()
                    opacity = 0.4
                    cv2.rectangle(img,(x+w+10,y-25),(x+w+150,y+115),(64,64,64),cv2.FILLED)
                    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                        
                    #connect face and expressions
                    cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),(255,255,255),1)
                    cv2.line(img,(x+w,y-20),(x+w+10,y-20),(255,255,255),1)
                        
                    emotion = ""
                    for i in range(len(predictions[0])):
                        emotion = "%s %s%s" % (emotions[i], round(predictions[0][i]*100, 2), '%')
                            
                        color = (255,255,255)
                            
                        cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames_for_emotion_detection_seperate():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            continue
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = faceCascade.detectMultiScale(gray, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x+w, y+h),(255, 0, 0), thickness=7) # cropping region of interest i.e. face area from  image
                roi_gray = gray[y:y+w, x:x+h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                # find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy','sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(frame, predicted_emotion, (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(frame, (1000, 700))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


class FaceCV(object):
    CASE_PATH = "FaceDetection\Cascades\haarcascade_frontalface_default.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"


    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',self.WRN_WEIGHTS_PATH,cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, ""and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,help="depth of network")
    parser.add_argument("--width", type=int, default=8,help="width of network")
    args = parser.parse_args()
    return args


def gen_frames_for_ageGender_Detection_new():
    args = get_args()
    depth = args.depth
    width = args.width

    faceage = FaceCV(depth=depth, width=width)

    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(faceage.face_size, faceage.face_size)
            )
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), faceage.face_size, faceage.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = faceage.crop_face(frame, face, margin=40, size=faceage.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i,:,:,:] = face_img
            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                results = faceage.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()
            # draw results
            for i, face in enumerate(faces):
                label = "{}, {}".format(int(predicted_ages[i]),"Female" if predicted_genders[i][0] > 0.5 else "Male")
                faceage.draw_label(frame, (face[0], face[1]), label)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/index3')
def index3():
    return render_template('index3.html')

@app.route('/index4')
def index4():
    return render_template('index4.html')

@app.route('/index5')
def index5():
    return render_template('index5.html')

@app.route('/index6')
def index6():
    return render_template('index6.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames_for_face_eye_smile_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(gen_frames_for_emotion_detection_seperate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(gen_frames_for_ageGender_Detection_new(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed5')
def video_feed5():
    return Response(gen_frames_for_emotion_detection_combine(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
