# Face to Meme
#### Video Demo:  <URL HERE>
#### Description:

This program detects someones face given a video input and shows a meme that tries to match the expression on the persons face if any.
This project uses MediaPipe (https://github.com/google-ai-edge/mediapipe)
by Google, licensed under the Apache License 2.0.
**Do not erase or edit "face_landmarker.task" or "expression_model.joblib"** in order to improve the classification model you would need to train a new model from sratch using scikit learn. The tash can be found in mediapipe official documentation.

The main program is main.py. It starts the task, the model and captures input from the webcam, extracts the landmarks with mediapipe, the model predict an exression and using cv2 shows a meme along the face. 

The next important file is "expression_model.joblib" is a small model trained using schikit with a training dataset of 490 images of 35 different individual showing 7 different expressions. I took this data from Karolinska Directed Emotional Faces (KDEF) dataset.

Training the model was the hardest part and i used claude code to help me trim and clean the data as well to select the type of model.
The model was trained using Logistic Regression and scikit library. I decided to not include that part in the main repo because if you were to ask me i wouldnt know what does each part do exactly. But i will leave it in the dev branch.

Another hard part of the proccess was to find a good enough training dataset. The first workig demo was trained only using my face, it worked with me but it was bad at reading other peoples expressions. I then choose KDEF for the open source nature of it and because of the density and variety of it. Of course choosing a more robust training dataset(2000+ images) would improve the model. Adding more subtle recognition and even hand gestures recognition, this is one of the ideas i would like to apply to the project in the future.

# IMPORTAN - VIRTUAL ENVIROMENT.
In order to use this project successfully you need to setup a virtual enviroment with the dependencies listed in requirements.txt.

you can do so typing: 

pip install requirements.txt -r

or if you have uv installed
 
uv pip install requirements.txt -r

One of the main issues with my projects is version compatibility with python. The latest supported version is Python 3.14.7. This is due to mediapipe as long as i know. Of course it could be a local problem since i use Arch btw...sorry.


This was my final project for CS50 2025. I hope you like it :)