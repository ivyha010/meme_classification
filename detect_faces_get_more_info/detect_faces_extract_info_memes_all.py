from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from deepface import DeepFace
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import time


def open_json(js_file):
    dataset = []
    with open(js_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    all_face_array = []
    if len(results) > 0:
        for i in range(0, len(results)):
            # extract the bounding box from the first face
            x1, y1, width, height = results[i]['box']
            face_array = get_face(x1, y1, width, height, pixels, required_size)
            all_face_array.append(face_array)
    return all_face_array


def get_face(x1, y1, width, height, pixels, required_size):
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_arr = asarray(image)
    return face_arr


def analyze_face(facePixels, faceFolder):
    # save faces
    img = Image.fromarray(facePixels)
    face_img = os.path.join(faceFolder, "temp_face.png")
    img.save(face_img)
    face_analysis = DeepFace.analyze(img_path=face_img, enforce_detection=False)
    # obj = DeepFace.analyze(img_path=face_img, enforce_detection=False, actions=('age', 'gender', 'race', 'emotion'))
    obj_dominant = [face_analysis["dominant_emotion"], face_analysis["dominant_race"], face_analysis["gender"].lower()]
    del face_img
    return obj_dominant

def get_img_info(all_faces, face_Folder):
    join_all_result = ''
    if len(all_faces) > 0:
        all_result = []
        for i in range(0, len(all_faces)):
            result = analyze_face(all_faces[i], face_Folder)
            join_result = ' '.join(result)
            all_result.append(join_result)
        join_all_result = ', '.join(all_result)
    return join_all_result


if __name__=="__main__":
    inPath = "/home/minhdanh/Downloads/hateful_memes/"
    outPath = "/home/minhdanh/Downloads/hateful_memes/"
    img_folder = "/home/minhdanh/Downloads/hateful_memes/inpaint"   # path to inpainted images
    jsonlFile = "test_seen.jsonl" # "dev_seen.jsonl" "train.jsonl"

    start_time = time.time()
    temp = torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create a temporal folder to save faces
    face_folder = os.path.join(inPath, "temp_faces")
    if not (os.path.exists(face_folder)):
        os.makedirs(face_folder)

    # Read jsonl file:
    myData = open_json(os.path.join(inPath, jsonlFile))
    new_myData = []
    count = 0
    for i in range(0, len(myData)):
        img_file = myData[i]['img']
        find_slash = img_file.find('/')
        img = os.path.join(img_folder, img_file[(find_slash + 1):])
        # load a photo and extract faces
        all_face_pixels = extract_face(img)
        # get info from faces in a photo
        img_info = get_img_info(all_face_pixels, face_folder)
        # upfate data
        new_data = myData[i]
        new_data.update(text = img_info)
        new_myData.append(new_data)
        if count % 100 == 0:
            print("Processed: ", count, ", Running time: ", time.time()-start_time)
        count += 1

    # Write a json file:
    info_jsonl_file = os.path.join(outPath, "more_info_" + jsonlFile)
    with open(info_jsonl_file, 'w') as outfile:
        for i in range(0, len(new_myData)):
            json.dump(new_myData[i], outfile)
            outfile.write('\n')

    #check = open_json(info_jsonl_file)
    print("Overall running time: ", time.time() - start_time)




