import os
import threading
from concurrent.futures import process
from re import S
import cv2
import mediapipe as mp
import sys
import random
import numpy
import torch as torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

cuda = torch.device('cuda')

scaler = StandardScaler()


def linear(lst):
    r = []
    for a in lst:
        if type(a) == list:
            r += linear(a)
        else:
            r += [a]
    return r


class Perceptron(torch.nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.neurons = nn.Sequential(
            nn.Linear(936, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        output = self.neurons(torch.flatten(x))
        return output


model = Perceptron()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = []


def train(data, t):
    global criterion
    try:
        inp = torch.FloatTensor(data)
        test = torch.FloatTensor(t)
        pred = model.forward(inp).type(torch.FloatTensor)
        err = criterion(pred, test)
        err.backward()
        optimizer.step()
        optimizer.zero_grad()
        if err_count % 100 == 0:
            print(test, pred, pred.argmax())
        return err
    except:
        PATH = 'C:/face_change/model.pt'
        torch.save(model.state_dict(), PATH)
        print('error')
        criterion = nn.MSELoss()
        return 1


def get_data(category):
    global data
    x = 1
    scaler = StandardScaler()
    for root, dirs, files in os.walk(f"pointed_images/{category}"):
        for filename in files:
            with open(root + '/' + filename, 'r') as file:
                p = None
                if 'anger' in root:
                    p = 0
                if 'disgust' in root:
                    p = 1
                if 'fear' in root:
                    p = 2
                if 'happiness' in root:
                    p = 3
                if 'neutrality' in root:
                    p = 4
                if 'sadness' in root:
                    p = 5
                if 'surprise' in root:
                    p = 6
                y = [0 for i in range(7)]
                y[p] = 1
                xd = eval(file.read())
                # xd = scaler.fit_transform(xd)
                h1, h2 = 0, 0
                for g in xd:
                    h1 += g[0]
                if type(xd[0]) == 'list':
                    xd = linear([list(i) for i in linear(xd)])
                # train(xd, y)
                data.append([linear(xd), y])
                if x % 100 == 0:
                    print(f'{x} photos have been processed in {category} \n')
                x += 1


if __name__ == '__main__':
    categories = ['anger' 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
    t = []
    for k in categories:
        t.append(threading.Thread(target=get_data, args=[k]))
    for k in t:
        k.start()
    for k in t:
        k.join()
    print('data initialized')
    print(len(data[0][0]))
    error = 0
    err_count = 0
    random.shuffle(data)
    '''with open('test.txt', 'w') as file:
        file.write(str(data))
    '''
    for j in range(5):
        random.shuffle(data)
        for i in data:
            e = train(i[0], i[1])
            error += e
            err_count += 1
            if err_count % 1000 == 0:
                print(e)
    '''with open('test.txt', 'r') as file:
        p = 0
        y = [0 for i in range(8)]
        y[p] = 1
        xd = eval(file.read())
        train(linear(xd), y)'''

    PATH = 'C:/face_change/model.pt'
    torch.save(model.state_dict(), PATH)
