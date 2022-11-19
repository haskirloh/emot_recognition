import cv2
import mediapipe as mp
import numpy as np
import torch as torch
from ai import Perceptron

mp_drawing_styles = mp.solutions.drawing_styles
categories = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
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



PATH = 'C:/face_change/model.pt'
model = Perceptron()
model.load_state_dict(torch.load(PATH))
model.eval()


def getface(img):
    face = mp.solutions.face_mesh.FaceMesh()
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

def bts_to_img(bts):
    buff = np.fromstring(bts, np.uint8)
    buff = buff.reshape(1, -1)
    img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    return img


def get_emotion(i):
    image = cv2.flip(i, 1)
    res2 = fd.process(image)
    res2 = res2.detections
    if res2:
        res = res2[0].location_data.relative_bounding_box
        height, width, _ = image.shape
        x, y = int(res.xmin * width), int(res.ymin * height)
        face_w = int(res.width * width)
        face_h = int(res.height * height)
        image = image[y:y + face_w, x:x + face_h]
        image = cv2.resize(image, (48, 48))
        cv2.imwrite('tmp.png', image)
        e1 = getface(image)
        if e1:
            # xd = scaler.fit_transform(xd)
            if type(e1[0]) == 'list':
                xd = linear([list(i) for i in linear(e1)])
            inp = torch.FloatTensor(e1)
            pred = model.forward(inp).type(torch.FloatTensor)
            ind = int(pred.argmax())
            return pred


