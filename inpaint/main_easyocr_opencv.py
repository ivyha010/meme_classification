from easyocr import Reader
import argparse
import cv2
import os
#import torch
#from PIL import Image
#import PIL
import numpy as np
import math
import time


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)


def inpaint_text(img_path):
    # read image
    img = cv2.imread(img_path)
    # OCR the input image using EasyOCR
    print("[INFO] OCR'ing input image...")
    reader = Reader(langs, gpu=args["gpu"] > 0)
    prediction_groups = reader.readtext(img)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups:
        x0, y0 = box[0][0] # tl
        x1, y1 = box[0][1] # tr
        x2, y2 = box[0][2] # br
        x3, y3 = box[0][3] # bl

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return (img)


def get_filenames(dir_path):
    res = []
    for file in os.listdir(dir_path):
        # check only png files
        if file.endswith('.png'):
            res.append(file)
    return res

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--langs", type=str, default="en", help="comma separated list of languages to OCR")
    ap.add_argument("-g", "--gpu", type=int, default=1,help="whether or not GPU should be used")
    args = vars(ap.parse_args())

    langs = args["langs"].split(",")
    print("[INFO] OCR'ing with the following languages: {}".format(langs))

    main_path = '/media/minhdanh/c6e3075c-22a5-4227-85ee-a1870fd3fe53/home/minhdanh/Downloads/hateful_memes/'
    img_folder = os.path.join(main_path, 'img')
    inpaint_folder = os.path.join(main_path, 'inpaint')
    if not os.path.exists(inpaint_folder):
        os.makedirs(inpaint_folder)

    fileList = get_filenames(img_folder)
    start_time = time.time()
    for img in fileList:
        image_path = os.path.join(img_folder, img)
        image = inpaint_text(image_path)
        cv2.imwrite(os.path.join(inpaint_folder, img), image)

    end_time = time.time()
    print("Running time: ", end_time - start_time)

