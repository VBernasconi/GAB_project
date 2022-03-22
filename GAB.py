"""
Author: V.Bernasconi
Last update: 13.10.2021
"""
from flask import Flask, Blueprint, render_template, Response, request, redirect, url_for, send_from_directory

import os
import cv2
import mediapipe as mp
import threading
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import imageio
import random
import sqlite3

from time import sleep
from datetime import datetime

from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=5)

PATH_PKL = ".."
PATH_HAND_DATABASE = "../res/openpose_std_res_output_hands/"
PATH_IMG_DATABASE = "../git-DVStudies/OAI_harvester/biblhertz_images/"
sys.path.append(PATH_PKL)
from python_scripts import KPmanager

#Definition of variables used with global scope
global capture, rec, stop_thread, hand_detected, current_time, kps, folder_path, frames_folder
stop_thread=False   #thread for countdown when recording
hand_detected=False #hand detection when recording
capture=0           #sequence captured
rec=0               #camera recording
current_time=""     #current time when recording
kps=[]              #hand keypoints storage
folder_path=""      #specific sequence folder
frames_folder=""    #specific frames folder for hands sequence

app = Flask(__name__)

camera = cv2.VideoCapture(0)    #launch webcam

#create directories to store recorded sequence info
try:
    os.mkdir('./static/hands_from_sequence')
    os.mkdir('./static/videos')
except OSError as error:
    pass

#define joints dictionary
joints = []
prev_i = 0
for i in range(1, 21, 4):
    if i >1:
        joints.append([prev_i, 0, i])
    prev = 0
    for j in range(i, i+3):
        joints.append([prev,j,j+1])
        prev = j
    prev_i = i

#define bones dictionary
bone_lst = []
for i in [1,5,9,13,17]:
    bone_lst.append([0, i])
    prev_jt = i
    for j in range(i+1, i+4):
        bone_lst.append([prev_jt, j])
        prev_jt = j

#imports for KNN and fit
with open(PATH_PKL+'/python_scripts/painted_imgs_XY_JT_UNIT_PLAN.pkl','rb') as f: XY_JT_UNIT_PLAN_paintings = pickle.load(f)
with open(PATH_PKL+'/python_scripts/painted_imgs_XY_name_list.pkl','rb') as f: painted_hands_list = pickle.load(f)
knn.fit(XY_JT_UNIT_PLAN_paintings)


"""
coutdown()
    When a hand is detected, the sequence recording is started. The countdown()
    function allows to limit the recording process to 5 seconds
"""
def countdown():
    global stop_thread, my_timer, recorded, cTime, hand_detected, rec

    my_timer = 5
    hand_detected = True
    print("Hand detected, countdown started")
    for x in range(my_timer):
        my_timer = my_timer - 1
        cTime = my_timer
        sleep(1)
        if stop_thread == True:
            cTime = 5
            print("Countdown stopped")
            hand_detected = False
            return
    print("Video has been recorded")
    hand_detected = False
    rec=False
    recorded = True

"""
get_hands_KNN(kps):
    @param kps : keypoints from recorded hands in a sequence

    The function compute the joints angle from the keypoints and use a K-nearest Neighbour model
    to get closest image in the set of images for recorded paintings
"""
def get_hands_KNN(kps):
    XY_JT_UNIT_PLAN = KPmanager.getJointsAngles2Plan(kps, joints, unitVector = True, bone_lst=bone_lst)

    images = []
    images_path = []

    file1 = open(folder_path+"/img_list"+current_time+".txt","w")

    prev_img = ""
    prev_prev_img = ""

    for i in range(len(XY_JT_UNIT_PLAN)):
        index = knn.kneighbors([XY_JT_UNIT_PLAN[i]], return_distance=False)[0]

        file = painted_hands_list[index[0]]
        if file == prev_img:
            file = painted_hands_list[index[1]]
        if file == prev_prev_img:
            file = painted_hands_list[index[2]]
        if file == prev_prev_img or file == prev_img:
            index_paint = knn.kneighbors([XY_JT_UNIT_PLAN_paintings[index[0]]], return_distance=False)[0]
            common_index = [value for value in index if value in index_paint]
            file = painted_hands_list[common_index[0]]

        prev_prev_img = prev_img
        prev_img = file

        outputfolder = os.path.realpath(PATH_HAND_DATABASE)
        img = os.path.join(outputfolder, file)

        file1.writelines(img+"\n")
        images_path.append(file)

        img_ = imageio.imread(img)
        img_ = cv2.resize(img_, (128, 128))
        images.append(img_)

    file1.close()

    gif_path = 'hands_from_sequence/'+current_time+'/KNN_movie_'+current_time+'.gif'
    imageio.mimsave('static/'+gif_path, images)

    return gif_path, images_path

"""
getBBox(kp, pix):
    @param kp : a set of keypoints for a single hand
    @param pix : a margin arround the keypoints to crop the bounding box

    The function returns the extremes coordinates from a set of keypoints in order
    to crop a given image around the detected hand
"""
def getBBox(kp, pix):

    x_min = min(kp, key=lambda x: x[0])
    y_min = min(kp, key=lambda x: x[1])

    x_max = max(kp, key=lambda x: x[0])
    y_max = max(kp, key=lambda x: x[1])

    height = y_max[1] - y_min[1]
    pix = height*pix

    return (max(0, int(x_min[0]-pix)), max(0, int(y_min[1]-pix))), (max(0, int(x_max[0]+pix)), max(0, int(y_max[1]+pix)))

"""
getImgDisplay(images):
    @param images : the list of names for painted hands images

    The function prepare four different lists of information in order to display
    on the html page the following information
    - the sequence of hands recorded on the webcam
    - the sequence of corresponding painted hands
    - the sequence of actual paintings holding the painted hands
    - the sequence of metadata available for each painting

    The function returns a zip of these four lists in order to be used in a for-loop
"""
def getImgDisplay(images):

    global frames_folder

    rec_hands_img = ['hands_from_sequence/'+current_time+'/hands_frame/'+hand_img_ for hand_img_ in frames_folder]

    #master_img = [file_.split("/",1)[1].split("_", 1)[0]+'.jpg' for file_ in images] #for kmeans images
    master_img = [file_.split("_", 1)[0]+'.jpg' for file_ in images] #for knn images

    conn = sqlite3.connect('biblhertz.db')
    cursor = conn.cursor()

    art_metadata = []
    for img_name in master_img:
        img_id = img_name.split(".", 1)[0]
        cursor.execute("SELECT title, artist_name, date_begin FROM Objects WHERE img_digital=?", (img_id,))
        rows = cursor.fetchall()
        for (title, artist_name, date_begin) in rows:
            art_card = title+", "+artist_name+", "+date_begin
            art_metadata.append(art_card)
            break

    return zip(rec_hands_img, images, master_img, art_metadata)

"""
get_file():
    handle route to get and display images stored in a different folder on local computer
"""
@app.route(os.path.realpath(PATH_HAND_DATABASE)+'<path:filename>')
def get_file(filename):
    return send_from_directory(PATH_HAND_DATABASE, filename, as_attachment=True)
"""
get_masterfile():
    handle route to get and display images stored in a different folder on local computer (here for original paintings)
"""
@app.route(os.path.realpath(PATH_IMG_DATABASE)+'<path:filename>')
def get_masterfile(filename):
    return send_from_directory(PATH_IMG_DATABASE, filename, as_attachment=True)
"""
results():
    call different functions to create the gif from the recorded sequence, get
    corresponding images and metadata
"""
@app.route("/results")
def results():
    gif_path, images = get_hands_KNN(kps)
    img_display = getImgDisplay(images)
    return render_template('index.html', gif_path='/'+gif_path, img_display=img_display)

"""
tasks():
    once the record button is pressed, the function GET the data from the webcam
    and handle it:
        - check if hand is detected on the camera
        - start recording
        - start a new thread for countdown execution
        - store keypoints at each frame
"""
@app.route("/tasks", methods=['POST', 'GET'])
def tasks():
    global switch, camera, recorded, rec, stop_thread, current_time, kps, folder_path, frames_folder
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if request.method == 'GET':
        rec= True
        if(rec):
            now = datetime.now()

            print("now =", now)

            current_time = now.strftime("%Y-%m-%d_%H-%M")

            try:
                os.mkdir('./static/hands_from_sequence/'+current_time)
            except OSError as error:
                pass

            folder_path = 'static/hands_from_sequence/'+current_time

            #below code store recorded hands
            try:
                os.mkdir('./'+folder_path+'/hands_frame')
            except OSError as error:
                pass

            #below code allows you to store the recording of the user's hand gesture
            """
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(folder_path+'/hand_output'+current_time+'.avi', fourcc, 20.0, (640, 480))
            """

            mpHands = mp.solutions.hands
            hands = mpHands.Hands()
            mpDraw = mp.solutions.drawing_utils

            pTime = 5
            cTime = 5

            stop_thread = True
            recorded = False

            kps = []
            frames_folder=[]
            i = 0

            while not recorded:
                success, img = camera.read()
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    		    #out.write(img)
                results = hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    if stop_thread:
                        stop_thread = False
                        #out = cv2.VideoWriter(folder_path+'/hand_output'+current_time+'.avi', fourcc, 20.0, (640, 480))
                        countdown_thread = threading.Thread(target = countdown)
                        countdown_thread.start()
                    """
                    else:
                        out.write(img)
                    """

                    for handLms in results.multi_hand_landmarks:
                        kp = []
                        kp_bbox = []
                        for id_, lm in enumerate(handLms.landmark):
                            kp.append([lm.x,lm.y])
                            h, w, c = img.shape
                            cx, cy = int(lm.x*w), int(lm.y*h)
                            kp_bbox.append([cx, cy])

                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                        kps.append(kp)
                        p1,p2 = getBBox(kp_bbox, 0.3)
                        hand_img = img[p1[1]:p2[1], p1[0]:p2[0]]
                        frame_name = 'frame_'+current_time+'_'+str(i)+'.jpg'
                        cv2.imwrite(folder_path+'/hands_frame/'+frame_name, hand_img)
                        frames_folder.append(frame_name)
                        i+=1
                else:
                    if not stop_thread:
                        stop_thread = True
                        countdown_thread.join()
                        #out.release()
                        kps = []

            print("Number of keypoints ", len(kps))
            camera.release()
            # After we release our webcam, we also release the output
            #out.release()
            cv2.destroyAllWindows()

            return render_template('index.html', loading_state="True")

    else:
        return render_template('index.html')

    return redirect(url_for('results'))

"""
gen_frames()
    render webcam capture on the html page
"""
def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if(rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame,1), "Recording process started...", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (142,31,94), 2)
                frame = cv2.flip(frame, 1)
            if(hand_detected):
                frame = cv2.putText(cv2.flip(frame,1), "hand is detected", (5,38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (142,31,94), 2)
                frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

"""
video_feed():
    handle webcam display on .html page
"""
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

"""
index():
    home page, check if webcam is on.
"""
@app.route("/")
def index():
    global camera
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)))
