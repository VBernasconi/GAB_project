## GAB - Gestures for artworks browsing
The present page presents Gestures for Artwork Browsing (GAB), a web application which proposes to use hand motions as a way to directly query pictorial hand gestures from the past. Based on materials from a digitized collection of Renaissance paintings, GAB enables users to record a sequence with the hand movement of their choice and outputs an animation reproducing that same sequence with painted hands. Fostering new research possibilities, the project is a novelty in terms of art database browsing and human-computer interaction, as it does not require traditional search tools such as text-based inputs based on metadata, and allows a direct communication with the content of the artworks.

### How it works
1. Record a gesture of 5 seconds with your hand
![Hand recording](/docs/assets/images/GAB_01.png)
3. Wait for the computer to process. It retrieves your hand from each frame of the recorded sequence. For each hand, the closest painted hand from the painting collection is found. More on the process in the [next section](#behind-the-scene)
4. Your result is then displayed. It consists of a .gif animation representing your movement and the possibility to browse through each frame and get further information on the original painting.
<p align="center"><img alt="Gif animation" src="/GAB_project/docs/assets/images/movie_knn_2021-10-14_12-34.gif"></p>
![Results detail](/docs/assets/images/GAB_02.png)

### Behind the scene
1. Before the creation of the application, a collection of painted hands was generated with the help of the [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) model. The latter detects body poses through a set of keypoints representing the shape of a skeleton. The hands were automatically croped from the keypoints information (a bounding box was created based on the extreme coordinates), resulting in a set of painted hands, which was later manually cleaned.
![Openpose detect](/docs/assets/images/GAGA_bibhertz.png)
3. A simple k-NN was trained on the keypoints information available for each hand.
4. When the user records a sequence, the [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) model is used to analyze the images in real-time. For each image, a hand is detected with its corresponding keypoints. The coordinates of these keypoints are stored to be later processed.
5. Once the sequence is fully recorded, the keypoints are then processed. The retrieval of similar hand poses is performed with the pre-trained k-NN model. It outputs a list of the five painted hands closest to the hand recorded. To avoid redundancy, the two previous images used in the sequence are looped through in order to check if a hand was already used in two previous frames.
```code

```

### Whole process summary
<video src="/GAB_project/docs/assets/images/GAB_Bernasconi_IUI_2022_small.mp4" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:1080px;"></video>
