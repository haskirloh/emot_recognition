import cv2
import mediapipe as mp

import torch as torch
import torch.nn as nn
from ai import Perceptron
from sklearn.preprocessing import StandardScaler
cuda = torch.device('cuda')
mp_drawing_styles = mp.solutions.drawing_styles
categories = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
cap = cv2.VideoCapture(0)
success, image = cap.read()
face = mp.solutions.face_mesh.FaceMesh()
fd = mp.solutions.face_detection.FaceDetection()

draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def linear(lst):
    r = []
    for a in lst:
        if type(a) == list:
            r += linear(a)
        else:
            r += [a]
    return r


def getface(img):
    image = img
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face.process(imageRGB)
    landmarks = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
    return landmarks


PATH = 'C:/face_change/model.pt'
model = Perceptron()
model.load_state_dict(torch.load(PATH))
model.eval()

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
    success, image = cap.read()
    image = cv2.flip(image, 1)
    res2 = fd.process(image)
    res2 = res2.detections
    res = res2[0].location_data.relative_bounding_box
    height, width, _ = image.shape
    x, y = int(res.xmin * width), int(res.ymin * height)
    face_w = int(res.width * width)
    face_h = int(res.height * height)
    image = image[y:y + face_w, x:x + face_h]
    image = cv2.resize(image, (48,48))
    xd = getface(image)
    if xd:
        scaler = StandardScaler()
        # xd = scaler.fit_transform(xd)
        if type(xd[0]) == 'list':
            xd = linear([list(i) for i in linear(xd)])
        inp = torch.FloatTensor(xd)
        pred = model.forward(inp).type(torch.FloatTensor)
        print(pred)
        ind = int(pred.argmax())
        print(categories[ind])

    cv2.imshow("Face", image)
