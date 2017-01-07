from sinesp_client import SinespClient
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema

## Constants
kernel_size = (5,5)
threshold_low = 127
threshold_high = 255
sc = SinespClient()


## Check plate in a server
def check_plate(plate_number):
    info = sc.search(plate_number)
    return info
########################################################################################################################

## Segment Digits
def process_image(plate_img):
    if(len(plate_img.shape) == 3):
        plate_img = cv2.cvtColor(plate_img.copy(), cv2.COLOR_BGR2GRAY)
    plate_img_resized = cv2.resize(plate_img.copy(), (300, 100), interpolation=cv2.INTER_CUBIC)
    _, binary_img = cv2.threshold(plate_img_resized.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones(kernel_size, np.uint8)
    binary_img = cv2.erode(binary_img.copy(), kernel, iterations=1)
    binary_img = cv2.dilate(binary_img.copy(), kernel, iterations=1)
    return binary_img

def crop_segment(img):
    img_blurred = cv2.GaussianBlur(img.copy(), (1, 1), 0)
    img_edge = cv2.Canny(img_blurred.copy(), 0, 200)
    _, thresholded_edges = cv2.threshold(img_edge.copy(), 0, 255, cv2.THRESH_BINARY)
    countours, _ = cv2.findContours(thresholded_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areas = []
    boxes = []
    for cnt in countours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        areas.append(w * h)
        boxes.append((x, y, w, h))

    max_area_index = (areas.index(max(areas)))
    (xlim, ylim, wlim, hlim) = boxes[max_area_index]
    im_aux = img.copy()[ylim:ylim + hlim, xlim:xlim + wlim]
    return im_aux


def seg_img(img):
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    binary_img = process_image(img.copy())
    kernel = np.ones((5,5), np.uint8)
    binary_img = cv2.erode(binary_img.copy(), kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    binary_img = cv2.dilate(binary_img.copy(), kernel, iterations=1)
    img_edge = binary_img.copy()
    _, thresholded_edges = cv2.threshold(img_edge.copy(), 0, 255, cv2.THRESH_BINARY)
    countours, _ = cv2.findContours(thresholded_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, countours, -1, (0, 255, 0), -1)
    imgP = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2RGB)

    areas = []
    boxes = []
    for cnt in countours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        box = (x,y,x+w,y+h)
        areas.append(area)
        boxes.append(box)
        cv2.rectangle(imgP, (x, y), (x + w, y + h), (255, 0, 255), 2)

    max_areas_indexes = np.argsort(areas)[::-1]
    max_areas_indexes = max_areas_indexes[0:7]

    org = []
    for i in max_areas_indexes:
        org.append(boxes[i][0])

    organized_digits_index = np.argsort(org)

    maxA=[]
    for j in organized_digits_index:
        maxA.append(max_areas_indexes[j])


    digits = []
    for index in maxA:
        digits.append( binary_img[ boxes[index][1]:boxes[index][3], boxes[index][0] : boxes[index][2]   ]  )

    return digits
########################################################################################################################

## Identify digits


def identify(digits):

    blib =['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    clf_n = pickle.load(open("Training/Trained Neural Network Files/neural-numbers.pkl", 'rb'))
    clf_l = pickle.load(open("Training/Trained Neural Network Files/neural-letters.pkl", 'rb'))
    licencePlate = ""

    for i,img in enumerate(digits):
        img = cv2.resize(img.copy(), (28, 28), interpolation=cv2.INTER_CUBIC)

        x = (np.asfarray(img.copy()) / 255.0) * 0.99  # + 0.01
        x = np.array(x.copy()).flatten()
        x = x.reshape(1, -1)

        if(i < 3):
            result = clf_l.predict(x)
        else:
            result = clf_n.predict(x)

        licencePlate = licencePlate + str(result[0])

    return(licencePlate)
