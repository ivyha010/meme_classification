from PIL import Image   # pillow
import requests
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import os
import time
import json
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import math


def open_json(js_file):
    dataset = []
    with open(js_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

# Build a dataloader
def myDataloader(img_data, txt_data, batchsize):
    class my_dataset(Dataset):
        def __init__(self, imgData, txtData):
            self.imgData = imgData
            self.txtData = txtData

        def __getitem__(self, index):
            return self.imgData[index], self.txtData[index]

        def __len__(self):
            return len(self.imgData)
    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(img_data, txt_data), batch_size=batchsize, shuffle=False)
    return my_dataloader


def extract_features(tProcessor, tModel, tDataloader):
    all_img_featureVec = []
    all_text_featureVec = []
    count = 0
    for (img_batch, txt_batch) in tDataloader:
        for i in range(0, len(img_batch)):
            img = img_batch[i]
            image = Image.open(img)
            text = txt_batch[i]
            inputs = tProcessor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
            tModel = tModel.to(device)
            outputs = tModel(**inputs)
            img_featureVec = outputs.image_embeds
            #text_featureVec = outputs.text_embeds
            all_img_featureVec.append(img_featureVec)
            #all_text_featureVec.append(text_featureVec)

        if (count % 3) == 0:
            print("Processing: ", count * batch_size)
        count = count + 1
    return all_img_featureVec, all_text_featureVec


def process_split(tprocessor, tmodel, tdata_loader, jsonl_file, out_path, part=""):
    all_feature_vectors = extract_features(tprocessor, tmodel, tdata_loader)
    # Save the extracted features in a h5 file
    img_all_feature_vectors = torch.cat(all_feature_vectors[0], dim=0).to('cpu').detach().numpy()
    #text_all_feature_vectors = torch.cat(all_feature_vectors[1], dim=0).to('cpu').detach().numpy()
    img_save_file = "img_CLIP_features_" + os.path.splitext(jsonl_file)[0] + "_" + part +".h5"
    #text_save_file = "text_CLIP_features_" + os.path.splitext(jsonl_file)[0] + "_" + part +".h5"
    save_features(img_all_feature_vectors, os.path.join(out_path, img_save_file))
    #save_features(text_all_feature_vectors, os.path.join(out_path, text_save_file))


def save_features(tData, savePath):
    h5file = h5py.File(savePath, mode='w')
    h5file.create_dataset("default", data=tData, dtype=np.float32)
    h5file.close()


def read_h5file(h5_filePath):
    h5file = h5py.File(h5_filePath, 'r')
    getData = h5file.get("default")
    dataArray = np.array(getData)
    features = torch.from_numpy(dataArray)  # .to(device)  # Convert numpy arrays to tensors on gpu
    h5file.close()
    return features


if __name__ == '__main__':
    inPath = "/home/minhdanh/Downloads/hateful_memes"
    outPath = "/home/minhdanh/Downloads/hateful_memes/extracted_features"
    img_folder = "/home/minhdanh/Downloads/hateful_memes/inpaint"  # path of inpainted images
    jsonlFile = "test_seen.jsonl" # "train.jsonl" # "dev_seen.jsonl"

    start_time = time.time()
    temp = torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 20

    # Load the pretrained model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Read jsonl file:
    myData = open_json(os.path.join(inPath, jsonlFile))

    # Build dataloader
    all_img = []
    all_text = []
    for i in range(0, len(myData)):
        img_file = myData[i]['img']
        find_slash = img_file.find('/')
        img = os.path.join(img_folder, img_file[(find_slash + 1):])
        all_img.append(img)
        #all_text.append(myData[i]['text'].replace('"',''))
        all_text.append("")

    # Create a temporal folder
    temp_outPath = os.path.join(outPath, "temp_split")
    if not (os.path.exists(temp_outPath)):
        os.makedirs(temp_outPath)

    # split the dataset into parts
    split_size = 1000
    num_split = int(math.ceil(len(myData)/split_size))
    for j in range(0, num_split):
        start = j*split_size
        end = (j+1)*split_size
        if j<(num_split-1):
            data_loader = myDataloader(all_img[start:end], all_text[start:end], batch_size)
        else:
            data_loader = myDataloader(all_img[start:], all_text[start:], batch_size)
        process_split(processor, model, data_loader, jsonlFile, temp_outPath, "part_"+ str(j))
        print("Processed part: ", j)


    # Concatenate all splits into one file
    all_feat = []
    for j in range(0, num_split):
        img_save_file_j = "img_CLIP_features_"+ os.path.splitext(jsonlFile)[0] + "_part_" + str(j) + ".h5"
        img_features_j = read_h5file(os.path.join(temp_outPath, img_save_file_j))
        all_feat.append(img_features_j)
    all_feat = torch.cat(all_feat, dim=0).to('cpu').detach().numpy()
    save_features(all_feat, os.path.join(outPath, "img_CLIP_features_" + os.path.splitext(jsonlFile)[0] + ".h5"))
    #check = read_h5file(os.path.join(outPath,  "img_CLIP_features_" + os.path.splitext(jsonlFile)[0] + ".h5"))

    print('Running time: ', time.time() - start_time)



