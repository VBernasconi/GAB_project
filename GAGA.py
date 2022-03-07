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
import itertools

from time import sleep
from datetime import datetime

sys.path.append("../..")
from python_scripts import KPmanager

from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=5)

#Definition of variables used with global scope
global capture, rec, stop_thread, hand_detected, current_time, kps, folder_path, frames_folder
global mov_TASK, time_TASK
global time_image_path

mov_TASK = 0
time_TASK = 0
stop_thread=False   #thread for countdown when recording
hand_detected=False #hand detection when recording
capture=0           #sequence captured
rec=0               #camera recording
current_time=""     #current time when recording
kps=[]              #hand keypoints storage
folder_path=""      #specific sequence folder
frames_folder=""    #specific frames folder for hands sequence

time_image_path = ""

app = Flask(__name__)

camera = cv2.VideoCapture(0)    #launch webcame

#create directories to store recorded sequence info
try:
    os.mkdir('./static/mov_task/hands_from_sequence')
    os.mkdir('./static/mov_task/videos')
    os.mkdir('./static/time_task/hand_picture')
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

#imports for KNN
with open('../../python_scripts/painted_imgs_XY_JT_UNIT_PLAN.pkl','rb') as f: XY_JT_UNIT_PLAN_paintings = pickle.load(f)
with open('../../python_scripts/painted_imgs_XY_name_list.pkl','rb') as f: painted_hands_list = pickle.load(f)
knn.fit(XY_JT_UNIT_PLAN_paintings)

#imports for time retrieval
with open('../century_indexes.pkl','rb') as f: century_indexes = pickle.load(f)
with open('../century_XYs.pkl','rb') as f: century_XYs = pickle.load(f)
with open('../df_handsyear_CLEANED.pkl','rb') as f: df_handsyear_CLEANED = pickle.load(f)

knn_centuries = []
for xys_ in century_XYs:
    if len(xys_)>=5:
        knn_century = NearestNeighbors(radius=2.05)
        knn_century.fit(xys_)
        knn_centuries.append(knn_century)

"""
adjust_radius(radius_)
    adjust the radius for the NearestNeighbors computation
"""
def adjust_radius(radius_):
    knn_centuries = []
    for xys_ in century_XYs:
        if len(xys_)>=5:
            knn_century = NearestNeighbors(radius=radius_)
            knn_century.fit(xys_)
            knn_centuries.append(knn_century)
    return knn_centuries

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
                if mov_TASK :
                    frame = cv2.putText(cv2.flip(frame,1), "Recording process started...", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127,104,19), 2)
                    frame = cv2.flip(frame, 1)
                    if(hand_detected):
                        frame = cv2.putText(cv2.flip(frame,1), "hand is detected", (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127,104,19), 2)
                        frame = cv2.flip(frame, 1)
                if time_TASK :
                    if(hand_detected):
                        frame = cv2.putText(cv2.flip(frame,1), "Hand is detected", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127,104,19), 2)
                        frame = cv2.flip(frame, 1)
                        frame = cv2.putText(cv2.flip(frame,1), str(my_timer), (250,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127,104,19), 2)
                        frame = cv2.flip(frame, 1)
                    else :
                        frame = cv2.putText(cv2.flip(frame,1), "Place one hand in front of the camera", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (127,104,19), 2)
                        frame = cv2.flip(frame, 1)

            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
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
get_image(cluster):
    Retrieve an image of a painted hand from a cluster given as paramter
    For the time being, the retrieve is random within the cluster provided
    @TODO: get closer painted hand within the cluster
"""
def get_image(cluster):
    outputfolder = r"/home/bernasconi/Documents/Bernasconi/PhD_clusters/kmeans_clustering/"

    all_files = os.listdir(outputfolder+str(cluster)) # dir is your directory path
    number_files = len(all_files)

    file = random.sample(all_files, 1)[0]
    file = os.path.join(outputfolder+str(cluster), file)

    return file

"""
get_hands_KMEAN(kps):
    @param kps : keypoints from recorded hands in a sequence

    The function compute the joints angle from the keypoints and use a pre-trained model
    to classify the hands recorded. It produces a list of corresponding clusters, inside
    which a random image will be picked with a get_image() call
"""
def get_hands_KMEAN(kps):
    XY_JT_UNIT_PLAN = KPmanager.getJointsAngles2Plan(kps, joints, unitVector = True, bone_lst=bone_lst)
    with open("../../python_scripts/kmeans_COCO_HANDS_dir_unit_PLAN.pkl", "rb") as f:
        kmeans_COCO_HANDS_unit_PLAN = pickle.load(f)

    labels = kmeans_COCO_HANDS_unit_PLAN.predict(XY_JT_UNIT_PLAN)
    print('Number of labels ', len(labels))

    images = []
    images_path = []

    file1 = open(folder_path+"/img_list"+current_time+".txt","w")

    for label in labels:
        img = get_image(label)
        file1.writelines(img+"\n")
        file_name = os.path.basename(img)
        images_path.append(str(label)+'/'+file_name)

        img_ = imageio.imread(img)
        img_ = cv2.resize(img_, (128, 128))
        images.append(img_)

    file1.close()

    gif_path = 'mov_task/hands_from_sequence/'+current_time+'/movie_'+current_time+'.gif'
    imageio.mimsave('static/'+gif_path, images)

    return gif_path, images_path

"""
get_hands_time_KNN(kps):
    @param kps : keypoints from recorded hands in a sequence

    The function compute the joints angle from the keypoints and use a K-nearest Neighbour model
    to get closest image in the set of images for recorded paintings
"""
def get_hands_time_KNN(kps):
    XY_JT_UNIT_PLAN = KPmanager.getJointsAngles2Plan(kps, joints, unitVector = True, bone_lst=bone_lst)

    images = []

    file1 = open('static/'+folder_path+"/img_list"+current_time+".txt","w")
    outputfolder = r"/home/bernasconi/Documents/Bernasconi/res/openpose_std_res_output_hands"

    knn_res_centuries = []
    i = 1
    for knn_ in knn_centuries:
        knn_res_ = knn_.radius_neighbors([XY_JT_UNIT_PLAN[0]], return_distance=False)[0]
        knn_res_centuries.append(knn_res_)
        for knn_res_index in knn_res_:
            file = painted_hands_list[century_indexes[i][knn_res_index]]
            """img_path = os.path.join(outputfolder, file)"""
            img_simple = file.split(".", 1)[0]
            images.append([file, img_simple])
        i+=1
    radius = 3.0
    while len(images)<5:
        i = 1
        radius+=0.05
        knn_res_centuries = []
        images = []
        knn_adjusted_ = adjust_radius(radius)
        for knn_ in knn_adjusted_:
            knn_res_ = knn_.radius_neighbors([XY_JT_UNIT_PLAN[0]], return_distance=False)[0]
            knn_res_centuries.append(knn_res_)
            for knn_res_index in knn_res_:
                file = painted_hands_list[century_indexes[i][knn_res_index]]
                """img_path = os.path.join(outputfolder, file)"""
                img_simple = file.split(".", 1)[0]
                images.append([file, img_simple])
            i+=1

    """for index_ in indexes :
        file = painted_hands_list[index_]
        outputfolder = r"/home/bernasconi/Documents/Bernasconi/res/openpose_std_res_output_hands"
        img_path = os.path.join(outputfolder, file)

        images.append([file, img_path])"""

    #master_img = [file_.split("/",1)[1].split("_", 1)[0]+'.jpg' for file_ in images] #for kmeans images
    master_img = [file_.split("_", 1)[0]+'.jpg' for [file_, path_] in images] #for knn images

    art_metadata = []
    print(images)

    for img_name in master_img:
        img_id = img_name.split(".", 1)[0]
        img_spec = df_handsyear_CLEANED[df_handsyear_CLEANED['img_digital']==img_id]
        art_card = img_spec[['title', 'artist_name', 'date_begin', 'img_digital']].values[0]
        art_metadata.append(art_card)


    """conn = sqlite3.connect('biblhertz.db')
    cursor = conn.cursor()

    art_metadata = []
    for img_name in master_img:
        img_id = img_name.split(".", 1)[0]
        cursor.execute("SELECT title, artist_name, date_begin FROM Objects WHERE img_digital=?", (img_id,))
        rows = cursor.fetchall()
        for (title, artist_name, date_begin) in rows:
            art_card = [title, artist_name, date_begin]
            art_metadata.append(art_card)
            break

    cursor.close()
    conn.close()"""

    # zipping lists of lists
    img_list = [list(itertools.chain(*i)) for i in zip(images, art_metadata)]
    print("IMG LIST")
    print(img_list)
    time_img_list = sorted(img_list, key=lambda x: x[4])
    print("SORTED", time_img_list)

    for item in time_img_list:
        file1.writelines(item[3]+"\n")
    file1.close()

    return time_img_list

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

        outputfolder = r"/home/bernasconi/Documents/Bernasconi/res/openpose_std_res_output_hands"
        img = os.path.join(outputfolder, file)

        file1.writelines(img+"\n")
        images_path.append(file)

        img_ = imageio.imread(img)
        img_ = cv2.resize(img_, (128, 128))
        images.append(img_)

    file1.close()

    gif_path = 'mov_task/hands_from_sequence/'+current_time+'/KNN_movie_'+current_time+'.gif'
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

    rec_hands_img = ['mov_task/hands_from_sequence/'+current_time+'/hands_frame/'+hand_img_ for hand_img_ in frames_folder]

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

    cursor.close()
    conn.close()

    return zip(rec_hands_img, images, master_img, art_metadata)

"""
time_record():
    once the record button is pressed, the function GET the data from the webcam
    and handle it:
        - check if hand is detected on the camera
        - start recording
        - start a new thread for countdown execution
        - store keypoints at each frame
        @app.route("/mov_task/record", methods=['POST', 'GET'])
"""
def time_record():
    global switch, camera, recorded, rec, stop_thread, current_time, kps, folder_path, frames_folder, time_image_path
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    rec= True
    if(rec):
        now = datetime.now()

        print("now =", now)

        current_time = now.strftime("%Y-%m-%d_%H-%M")

        try:
            os.mkdir('./static/time_task/hand_picture/'+current_time)
        except OSError as error:
            pass

        folder_path = 'time_task/hand_picture/'+current_time
        try:
            os.mkdir('./static/'+folder_path+'/hands_frame')
        except OSError as error:
            pass

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

            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                if stop_thread:
                    stop_thread = False
                    countdown_thread = threading.Thread(target = countdown)
                    countdown_thread.start()

                for handLms in results.multi_hand_landmarks:
                    kp = []
                    kp_bbox = []
                    for id_, lm in enumerate(handLms.landmark):
                        kp.append([lm.x,lm.y])
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        kp_bbox.append([cx, cy])
            else:
                if not stop_thread:
                    stop_thread = True
                    countdown_thread.join()
                    kps = []

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        kps.append(kp)
        p1,p2 = getBBox(kp_bbox, 0.3)
        hand_img = img[p1[1]:p2[1], p1[0]:p2[0]]
        frame_name = 'frame_'+current_time+'_'+str(i)+'.jpg'
        time_image_path = folder_path+'/hands_frame/'+frame_name
        print('./static/'+time_image_path)
        cv2.imwrite('./static/'+time_image_path, hand_img)
        frames_folder.append(frame_name)


        print("Number of keypoints ", len(kps))
        camera.release()
        # After we release our webcam, we also release the output
        cv2.destroyAllWindows()

        return 1

    return 0

"""
mov_record():
    once the record button is pressed, the function GET the data from the webcam
    and handle it:
        - check if hand is detected on the camera
        - start recording
        - start a new thread for countdown execution
        - store keypoints at each frame
        @app.route("/mov_task/record", methods=['POST', 'GET'])
"""
def mov_record():
    global switch, camera, recorded, rec, stop_thread, current_time, kps, folder_path, frames_folder
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    rec= True
    if(rec):
        now = datetime.now()

        print("now =", now)

        current_time = now.strftime("%Y-%m-%d_%H-%M")

        try:
            os.mkdir('./static/mov_task/hands_from_sequence/'+current_time)
        except OSError as error:
            pass

        folder_path = 'static/mov_task/hands_from_sequence/'+current_time
        try:
            os.mkdir('./'+folder_path+'/hands_frame')
        except OSError as error:
            pass

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(folder_path+'/hand_output'+current_time+'.avi', fourcc, 20.0, (640, 480))

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
                    out = cv2.VideoWriter(folder_path+'/hand_output'+current_time+'.avi', fourcc, 20.0, (640, 480))
                    countdown_thread = threading.Thread(target = countdown)
                    countdown_thread.start()
                else:
                    out.write(img)

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
                    out.release()
                    kps = []

        print("Number of keypoints ", len(kps))
        camera.release()
        # After we release our webcam, we also release the output
        out.release()
        cv2.destroyAllWindows()

        return 1

    return 0

"""
video_feed():
    handle webcam display on .html page
"""
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

"""
get_file():
    handle route to get and display images stored in a different folder on local computer
"""
@app.route('/res/openpose_std_res_output_hands/<path:filename>') #for knn images
def get_file(filename):
    #return send_from_directory('../PhD_clusters/kmeans_clustering/', filename, as_attachment=True) #for kmeans images
    return send_from_directory('../../res/openpose_std_res_output_hands/', filename, as_attachment=True) #for knn images

"""
get_masterfile():
    handle route to get and display images stored in a different folder on local computer (here for original paintings)
"""
@app.route('/git-DVStudies/OAI_harvester/biblhertz_images/<path:filename>')
def get_masterfile(filename):
    return send_from_directory('../../git-DVStudies/OAI_harvester/biblhertz_images/', filename, as_attachment=True)

"""
time_task_manager():
    hand the movement task:
    - start the recording process
    - get results - call different functions to create the gif from the recorded sequence, get
    corresponding images and metadata
"""
@app.route("/time_task/<subtask>", methods=['POST', 'GET'])
def time_task_manager(subtask):
    if subtask == "record":
        res = time_record()
        if res == 1:
            return render_template('time_task.html', loading_state="True")
        else :
            return redirect(url_for('time_task_manager', subtask="results"))
    elif subtask == "results":
        img_display = get_hands_time_KNN(kps) #get_hands(kps)
        return render_template('time_task.html', time_path=time_image_path, img_display=img_display)

    return render_template('index.html', intro=1)

"""
mov_task_manager():
    hand the movement task:
    - start the recording process
    - get results - call different functions to create the gif from the recorded sequence, get
    corresponding images and metadata
"""
@app.route("/mov_task/<subtask>", methods=['POST', 'GET'])
def mov_task_manager(subtask):
    if subtask == "record":
        res = mov_record()
        if res == 1:
            return render_template('mov_task.html', loading_state="True")
        else :
            return redirect(url_for('mov_task_manger', subtask="results"))
    elif subtask == "results":
        gif_path, images = get_hands_KNN(kps) #get_hands(kps)
        img_display = getImgDisplay(images)
        return render_template('mov_task.html', gif_path='/'+gif_path, img_display=img_display)

    return render_template('index.html', intro=1)

"""
time_task():
    home page, check if webcam is on.
"""
@app.route("/time_task")
def time_task():
    global camera, mov_TASK, time_TASK
    mov_TASK = 0
    time_TASK = 1
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return render_template('time_task.html')

"""
mov_task():
    home page, check if webcam is on.
"""
@app.route("/mov_task")
def mov_task():
    global camera, mov_TASK, time_TASK
    mov_TASK = 1
    time_TASK = 0
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return render_template('mov_task.html')

"""
index():
    home page, select feature
"""
@app.route('/', defaults={'path': ''})
@app.route("/")
def index():
    return render_template('index.html', intro=1)


if __name__ == "__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)))
