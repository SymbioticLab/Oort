#!/usr/bin/env python3

import os
import re
import csv
import math
from PIL import Image
from os import listdir
from os.path import isfile, join

def cropImage(filename, coords, saveFileName):
    """Crop image specified by filename to coordinates specified."""
    # print(f"DEBUG: cropImage({filename},{coords})")

    # Open image and get height and width
    try:
        im = Image.open(filename)
        w, h = im.width, im.height

        # Work out crop coordinates, top, left, bottom, right
        l = int(math.floor(coords['left']  * w))
        r = int(math.floor(coords['right'] * w))
        t = int(math.floor(coords['top']   * h))
        b = int(math.floor(coords['bottom']* h))

        #print(coords)

        # Crop and save
        im = im.crop((l,b,r,t))
        im.save("crop/" + saveFileName, quality = 95)
    except:
        print('Error with coords {}'.format(coords))
    return

# Create output directory if not existing
if not os.path.exists('crop'):
    os.makedirs('crop')

# Process CSV file - expected format
# heading;heading
# 00000001.jpg?sr.dw=700;{'right': 0.9, 'bottom': 0.8, 'top': 0.1, 'left': 0.2}
# 00000002.jpg?sr.dw=700;{'right': 0.96, 'bottom': 0.86, 'top': 0.2, 'left': 0.25}

imgTaken = {}
imgFiles = set([f for f in listdir('./') if isfile(join('./', f))])

with open('../oidv6-train-annotations-bbox.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    count = 0

    for row in csv_reader:
        count += 1

        if count == 1:
            continue

        imgID = row[0]

        if str(imgID+'.jpg') not in imgFiles:
            continue

        labelName = row[2].replace('/', '_')
        boxFeatures = int(row[8]) + int(row[9]) + int(row[10]) + int(row[11]) + int(row[12])

        # only take the one with no noise
        if boxFeatures != 0:
            continue

        uniqueName = imgID + '__' + labelName

        if uniqueName in imgTaken:
            continue

        if count % 1000 == 0:
            print('Current ....{}'.format(count))

        imgTaken[uniqueName] = 0

        coords = {'left': float(row[4]), 'right': float(row[5]), 'bottom': float(row[6]), 'top': float(row[7])}

        try:
            cropImage(imgID+'.jpg', coords, uniqueName+'.jpg')
        except:
            pass

        #count += 1
