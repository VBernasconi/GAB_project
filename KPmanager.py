import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import ntpath
import pandas as pd
import cv2
import imutils
from keras.preprocessing.image import load_img 

def saveImgBBox(kp, img_path, body_part, outputfolder, person_id, singleKP = False):
    img_name = getImgName(img_path)
    if not(singleKP):  
        p1,p2 = getBBox(kp, 0.3)
        img = cv2.imread(img_path)
        img = img[p1[1]:p2[1], p1[0]:p2[0]]
    else:
        img = cv2.imread(img_path)
        kp_left = [int(x)-10 for x in kp]
        kp_right = [int(x)+10 for x in kp]
        cv2.rectangle(img, kp_left, kp_right, (0,255,0), 3)
        imS = imutils.resize(img, width=500)
    if img.size > 0:
        img_name = img_name+"_"+str(person_id)+"_"+body_part+".jpg"
        #cv2.imshow(img_name, img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(outputfolder , img_name), img)
        
def createImgHands(kp, img_path, body_part, outputfolder, person_id, singleKP = False):
    img_name = getImgName(img_path)
    img = cv2.imread(img_path)
    if not(singleKP):  
        kp_left, kp_right = getBBox(kp, 0.3)        
    else:
        kp_left = [int(x)-10 for x in kp]
        kp_right = [int(x)+10 for x in kp]
    return kp_left, kp_right

def getBBox(kp, pix):
    xy_lst = getXY(kp)

    x_min = min(xy_lst, key=lambda x: x[0])
    y_min = min(xy_lst, key=lambda x: x[1])

    x_max = max(xy_lst, key=lambda x: x[0])
    y_max = max(xy_lst, key=lambda x: x[1])

    height = y_max[1] - y_min[1]
    pix = height*pix

    return (int(x_min[0]-pix), int(y_min[1]-pix)), (int(x_max[0]+pix), int(y_max[1]+pix))

def getKPwithoutConfidence(kp):
    kp_ = kp.copy()
    del kp_[3-1::3]
    return kp_

def getXY(kp):
    kp_ = getKPwithoutConfidence(kp)
    #return [kp_[i:i + 2] for i in range(0, len(kp_), 2)]
    #here -12 to remove the last kps corresponding to feet and background kp
    return [kp_[i:i + 2] for i in range(0, len(kp_), 2)]
# Returns a list of x and y coordinates for hand keypoints present in a given dataframe
# Returns corresponding file name
def getKeypoints(df, row):
    X_, Y_ =  [], []
    XY = []
    body_name = []

    for index, row in df.iterrows():

        body_ = row['keypoints']

        if any(body_) and row['num_keypoints']==17:
            xy = getXY(body_)
            x,y = zip(*xy)
            X_.append(x)
            Y_.append(y)
            XY.append(xy)
            img_name = row['path']+"_"+str(index)
            body_name.append(img_name)

    return X_, Y_, XY, body_name

def getImgName(path):
    head, tail = ntpath.split(path)
    split_string = tail.split("_", 1)
    split_string = split_string[0].split(".", 1)
    return split_string[0]

def getDFkeypoints(folderpath):
    filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]
    all_keypoints = []
    for path in filepaths:
        with open(path) as json_file:
            data = json.load(json_file)
            imgName = getImgName(path)
            for p in data['people']:
                p['img_id']=imgName
                all_keypoints.append(p)
                
    df = pd.DataFrame(all_keypoints)
    
    #set the correct person_id (-1 on json files)
    prev_img = 0
    person_id = 0
    for i, row in df.iterrows():
        if prev_img == row['img_id']:
            person_id +=1
        else:
            person_id = 0
        prev_img = row['img_id']
        df.at[i,'person_id'] = person_id
        
    return df

def isDuplicates(kp_list):
    for elem in kp_list:
        if kp_list.count(elem) > 2:
            return True
    return False

def getHandKeypoints(df):
    X_, Y_ =  [], []
    XY = []
    body_name = []

    for index, row in df.iterrows():
        for hand in ['lefthand', 'righthand']:
            if any(row[hand+"_kpts"]) and row[hand+"_valid"]==True:
                xy = getXY(row[hand+"_kpts"])
                emptyKP = sum([1 for list_ in xy if list_.count(0.0)>=2])
                if emptyKP <=1 and not isDuplicates(xy):
                    x,y = zip(*xy)
                    X_.append(x)
                    Y_.append(y)
                    XY.append(xy)
                    img_name = row['path']+"_"+str(index)+"_"+hand
                    body_name.append(img_name)

    return X_, Y_, XY, body_name

def plotBody(x, y, bone_lst, show = True):
    #X, Y = removeFeetKP(X, Y)
    #X_nz, Y_nz = removeZeroLocations(x, y)
    #plt.plot(X_nz,Y_nz, 'ro')
    plt.plot(x,y,'ro')

    for bone in bone_lst:
        #if (x[bone[0]]!= 0 and y[bone[0]]!=0) and (x[bone[1]]!= 0 and y[bone[1]]!=0):
        plt.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], 'r')

    plt.axis('equal')
    plt.gca().invert_yaxis() #flip the y axis to get the skeleton on the right side
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    if show:
        plt.gca().invert_yaxis()
        plt.show()

def angle_calculator(a, b, c):

    a= np.array(a)
    b= np.array(b)
    c= np.array(c)

    #first get the vectors
    BA = a-b
    BC = c-b

    #then we compute the angle
    unit_BA = BA / np.linalg.norm(BA)
    unit_BC = BC / np.linalg.norm(BC)

    dot_product = np.dot(unit_BA, unit_BC)
    #print(dot_product)  try:

    eps = 1e-6
    if 1.0 < dot_product < 1.0 + eps:
        dot_product = 1.0
    elif -1.0 - eps < dot_product < -1.0:
        dot_product = -1.0

    angle = np.arccos(dot_product)
    #print("Degree angle", np.degrees(angle))
    return angle

def computeNewCoordinate(a, b):
    if a[0] == b[0]:
        new_y = a[1]-((a[1]-b[1])/2)
        new_x = a[0]
    else:
        A_ = (b[1]-a[1])/(b[0]-a[0])
        B_ = a[1] - A_*a[0]
        new_x = a[0]-((a[0]-b[0])/2)
        new_y = A_* new_x + B_
    return [new_x, new_y]

def getJointsAngles(xy, joints, keepKP = False, unitVector = False, direction = False, bone_lst = []):
    xy_ = []
    index = 0
    for KP in xy:
        #plotBody(list(zip(*KP))[0],list(zip(*KP))[1])
        kp_jt_ = []
        prev_JT = [0,0,0]
        for JT in joints:
            a, b, c = KP[JT[0]], KP[JT[1]], KP[JT[2]]
            if b == c: #sometime keypoints are merged, here we define a slight shift for the first one
                print(index,"MERGED KEYPOINTS", a, b, c)
                KP[JT[1]] = computeNewCoordinate(a, b)
                b = KP[JT[1]]
                print("New coordinates for b:", b)
            if a == b :
                print(index, "MERGED KEYPOINTS", a, b, c)
                KP[JT[1]] = computeNewCoordinate(b, c)
                b = KP[JT[1]]
                print("New coordinates for b:", b)
            shoulderJT = angle_calculator(a, b, c)
            kp_jt_.append(shoulderJT)
        if unitVector:
            KP_unit = setUnitVector(KP, bone_lst)
            kp_jt_.extend(KP_unit.flatten())
        elif direction: #this one is to gen the general direction of the hand
            KP_unit = setUnitVector(KP, [[0, 9]])
            kp_jt_.extend(KP_unit.flatten())
        elif keepKP:
            kp_jt_.extend(KP.flatten())
        xy_.append(np.array(kp_jt_))
        index += 1
    return xy_

def getJointsAngles2Plan(xy, joints, keepKP = False, unitVector = False, direction = False, bone_lst = []):
    xy_ = []
    index = 0
    for KP in xy:
        kp_jt_ = []
        prev_JT = [0,0,0]
        for JT in joints:
            a, b, c = KP[JT[0]], KP[JT[1]], KP[JT[2]]
            if b == c: #sometime keypoints are merged, here we define a slight shift for the first one
                print(index,"MERGED KEYPOINTS", a, b, c)
                KP[JT[1]] = computeNewCoordinate(a, b)
                b = KP[JT[1]]
                print("New coordinates for b:", b)
            if a == b :
                print(index, "MERGED KEYPOINTS", a, b, c)
                KP[JT[1]] = computeNewCoordinate(b, c)
                b = KP[JT[1]]
                print("New coordinates for b:", b)
            JTangle = angle_calculator(a, b, c)
            JTangleX = np.sin(JTangle)
            JTangleY = np.cos(JTangle)
            kp_jt_.extend((JTangleX, JTangleY))
        if unitVector:
            KP_unit = setUnitVector(KP, bone_lst)
            kp_jt_.extend(KP_unit.flatten())
        elif direction: #this one is to gen the general direction of the hand
            KP_unit = setUnitVector(KP, [[0, 9]])
            kp_jt_.extend(KP_unit.flatten())
        elif keepKP:
            kp_jt_.extend(KP.flatten())
        xy_.append(np.array(kp_jt_))
        index += 1
    return xy_
    
def getUnitVector(a, b):
    a = np.array(a)
    b = np.array(b)

    AB = b-a
    unit_AB = AB /np.linalg.norm(AB)

    return unit_AB

def setUnitVector(xy, bone_lst):
    unit_xy = []

    for bone in bone_lst:
        unit_xy.append(getUnitVector(xy[bone[0]], xy[bone[1]]))

    return np.array(unit_xy)

def view_cluster_skeleton(cluster, labels, XY, bone_lst):
    plt.figure(figsize = (30,30));
    # gets the list of filenames for a cluster
    indices = [index for index, element in enumerate(labels) if element == cluster]

    bodies = [XY[i] for i in indices]

    if len(bodies) > 6:
        print(f"Getting 6 random samples from cluster of size {len(bodies)}")
        bodies = random.sample(bodies, 6)
    # plot each image in the cluster
    #fig, axes = plt.subplots(nrows=3, ncols=10)
    for i in range(len(bodies)):
        #plt.subplot(3,10,i+1)
        plt.subplot(3,10,i+1)
        plotBody(list(zip(*bodies[i]))[0],list(zip(*bodies[i]))[1], bone_lst, False)
        #plt.xlim([0, 1])
        #plt.ylim([0, 1])

    #plt.gca().invert_yaxis()
    plt.show()
    
#Creation of different plots

import matplotlib.pyplot as plt

def connectpoints(x,y,p1,p2, color):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'k-', color=color)

def plotHand(X, Y):
    plt.plot(X,Y, 'ro')
    wrist = 0
    prev_i = 0
    prev_main = 0
    for main_i, color in zip([5,9,13,17,21], ['red', 'yellow', 'green', 'blue', 'purple']):
        for index in range(prev_main, main_i):
            connectpoints(X,Y, prev_i, index, color)
            prev_i = index
        prev_i = 0
        prev_main = main_i

    plt.axis('equal')
    plt.gca().invert_yaxis()
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.show()
    
def plotHandLocation(X, Y):
    #First extract wrist coordinate for each hand
    X_ = [item[0] for item in X]
    Y_ = [item[0] for item in Y]
    plt.scatter(X_,Y_, c='red', alpha=0.5)
    #plt.rcParams['figure.figsize'] = [25, 20]
    plt.gca().invert_yaxis()
    plt.show()
    
def plotHandDirection(X, Y):
    #First extract wrist coordinate for each hand
    X_wrist = [item[0] for item in X]
    Y_wrist = [item[0] for item in Y]
    X_tip = [item[12] for item in X]
    Y_tip = [item[12] for item in Y]
    for i in range(0,len(X_wrist)):
        #if X_tip[i] > X_wrist[i]:
        plt.arrow(X_wrist[i], Y_wrist[i], X_tip[i]-X_wrist[i], Y_tip[i]-Y_wrist[i], head_width=1, color='green')
        #else:
            #plt.arrow(X_wrist[i], Y_wrist[i], X_wrist[i]-X_tip[i], Y_wrist[i]-Y_tip[i])
    # changing the rc parameters and plotting a line plot
    #plt.rcParams['figure.figsize'] = [25, 20]
    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    plt.gca().invert_yaxis()
    plt.show()
    
#Understanding clusters over time
def getClusterSampleOvertime(my_cluster, labels, hand_name, df_ = None):
    outputfolder = "/home/bernasconi/Documents/Bernasconi/res/openpose_std_res_output_hands"
    plt.figure(figsize = (25,25));
    indices = [index for index, element in enumerate(labels) if element == my_cluster]
    files = [hand_name[i] for i in indices]
    
    if df_ is None:
        df_ = df_handsyear_CLEANED
    
    df_my_cluster = df_.loc[df_['cluster'] == my_cluster]
    year_count = df_my_cluster.groupby('century')
    
    ordered_hands = []

    for index, file in enumerate(files):
        try:
            for row in df_[df_['id']==file]:
                ordered_hands.append((file, df_[df_['id']==file].iloc[0]["century"]))
        except:
            continue
    
    if len(ordered_hands) > 30:
        print(f"Getting 30 random samples from cluster number {my_cluster} of size {len(files)}")
        ordered_hands = random.sample(ordered_hands, 30)
    else:
        print(f"Getting samples from cluster number {my_cluster} of size {len(files)}")
        
    ordered_hands = sorted(ordered_hands, key=lambda x: x[1])
    # plot each image in the cluster
    for index, (file, year) in enumerate(ordered_hands):
        try:
            plt.subplot(10,10,index+1);
            img = load_img(os.path.join(outputfolder, file))
            img = np.array(img)
            plt.title(year)
            plt.imshow(img)
            plt.axis('off')   
        except OSError as e:
            print(file)

def showClustersRepartitionYear(df_= None, min_year = 1200, max_year=1751, labels=None):
    if labels is None:
        labels = [i for i in range(151) if i not in skip_cluster]
    
    if df_ is None:
        df_ = df_handsyear_CLEANED
        
    total_year_count = df_.groupby('century').size()
    my_df_cluster_year = pd.DataFrame()

    my_df_cluster_year['year'] = range(min_year, max_year, 50)#[1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700]

    for my_cluster in labels:
        x_ = []
        for year_, count_ in total_year_count.items():
            df_my_cluster = df_.loc[(df_['cluster'] == my_cluster) & 
                                                     (df_['century'] == year_)]
            #x_.append(len(df_my_cluster))
            perc = (len(df_my_cluster)*100)/count_
            x_.append(perc)
                    
        my_df_cluster_year[my_cluster] = x_
  
    # plot a Stacked Bar Chart using matplotlib
    ax = my_df_cluster_year.plot(
        x = 'year',
        kind = 'barh',
        stacked = True,
        title = 'Stacked Bar Graph',
        mark_right = True,
        figsize=(30,30))

    for c, l in zip(ax.containers, labels):
        ax.bar_label(c, labels=[str(l)]*len(c), label_type='center')