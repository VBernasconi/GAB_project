# Gestures for artworks browsing

Gestures for artworks browsing is project that allows you to browse through a collection of painted hands through your own gestures.

## How it works
Once the application starts, the user has the possibility to start recording a hand gesture of 5 seconds. The computer process the iamges and retrieve similar painted hands from the collection. It creates a .gif animation which reproduces the hand gesture through paintings. The user also has the possibility to brose through each frame to get more details about the original painting. For more details, check the [repository page](https://vbernasconi.github.io/GAB_project/#behind-the-scene).

## Technical specificites
- The collection of painted hands was created with [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (their pre-trained model was directly used on the paintings).
- The application uses the [Flask framework](https://flask.palletsprojects.com/en/2.0.x/).
- The hand detection from the webcamera uses [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)

## Repository
The present repository hosts the main code. The images used belong to the photographic collection of the [Bibliotheca Hertziana](https://www.biblhertz.it/en/photographic-collection) and are for the time being stored on a private server.
* GAB.py : the main Flask application file. Hold the route system for URL requests and all functions to record webcame images, store images, load classification model, apply model, retrieve corresponding image.
* __python_scripts/__ : 
  * KPmanager.py: Script with various functions to handle the keypoints
  * *.pkl* : pickles files with information on the collection of painted hands
* __template/__
  * index.html : an html page which content is dinamically adapted to the URL query with Flask in backend
* __static/__
  * hands_from_squence: A simple example of the output of the application. For each sequence recorded, a new folder is created with date and time information. For each sequence the following information is stored:
    * .avi : the video of the sequence in .avi format
    * .txt : a text file holding the route to all hand-paintings that are used for the creation of the .gif animation
    * .gif : the .gif animation created from the recorded sequence
    * hands_frame/ : a folder holding all the frames used from the recorded sequence that are analyzed and matched with their corresponding painted hands


## Installation
Install python and create a virtual environment. Then install Python dependencies using:
```
$ pip install -r requirements.txt
```
You can then simply launch the Flask application by running:
```
$ python 'GAB.py'
```
__NOTE:__ The collection of images belong to the photographic collection of the [Bibliotheca Hertziana](https://www.biblhertz.it/en/photographic-collection) and are stored on a private server. For the time being, they cannot be shared.
