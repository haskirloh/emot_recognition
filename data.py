import cv2
import mediapipe as mp
import os
import threading
import sys
from PIL import Image

t1 = 0
t2 = 0
k = 0
face = mp.solutions.face_mesh.FaceMesh()
fd = mp.solutions.face_detection.FaceDetection()


def getface(img):
    image = img
    image = cv2.flip(image, 1)
    results = face.process(image)
    landmarks = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
    return landmarks


def get_all(root):
    global t1, t2
    k = 0
    for root, dirs, files in os.walk(root):
        for filename in files:
            if k % 100 == 0:
                print(root, k)
            img = Image.open(root + '/' + filename)
            img = img.resize((48, 48))
            img.save(root + '/' + filename)
            image = cv2.imread(root + '/' + filename)
            image = cv2.flip(image, 1)
            res2 = fd.process(image)
            res2 = res2.detections
            if res2:
                res = res2[0].location_data.relative_bounding_box
                height, width, _ = image.shape
                x, y = int(res.xmin * width), int(res.ymin * height)
                x = 0 if x < 0 else x
                y = 0 if y < 0 else y
                face_w = int(res.width * width)
                face_h = int(res.height * height)
                image = image[y:y + face_w, x:x + face_h]
            cv2.imwrite(root + '/' + filename, image)
            path = root.replace('images', 'pointed_images') + '/' + filename.replace('.png', '.txt').replace('.jpg',
                                                                                                             '.txt')
            xd = getface(image)
            if xd:
                t1 += 1
                with open(path, 'w') as file:
                    file.write(str(xd))
            else:
                t2 += 1
            k += 1


def getpoints(root):
    image = cv2.imread(root)
    # cv2.imshow('', image)
    image = cv2.flip(image, 1)
    path = root.replace('.png', '.txt').replace('.jpg', '.txt')
    xd = getface(image)
    if xd:
        with open(path, 'w') as file:
            file.write(str(xd))
            print(len(xd))


'''categories = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
t = []
for k2 in categories:
    t.append(threading.Thread(target=get_all, args=['images/' + k2]))
for k2 in t:
    k2.start()
for k2 in t:
    k2.join()'''
get_all('images')
'''for root, dirs, files in os.walk("images/"):
    for filename in files:
        if k % 100 == 0:
            print(root, k)
        img = Image.open(root + '/' + filename)
        img = img.resize((48, 48))
        img.save(root + '/' + filename)
        image = cv2.imread(root + '/' + filename)
        image = cv2.flip(image, 1)
        res2 = fd.process(image)
        res2 = res2.detections
        if res2:
            res = res2[0].location_data.relative_bounding_box
            height, width, _ = image.shape
            x, y = int(res.xmin * width), int(res.ymin * height)
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            face_w = int(res.width * width)
            face_h = int(res.height * height)
            image = image[y:y + face_w, x:x + face_h]
        path = root.replace('images', 'pointed_images') + '/' + filename.replace('.png', '.txt').replace('.jpg', '.txt')
        xd = getface(image)
        if xd:
            t1 += 1
            with open(path, 'w') as file:
                file.write(str(xd))
        else:
            t2 += 1
        k += 1'''
print(t1, t2)
