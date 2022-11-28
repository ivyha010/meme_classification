from transformers import RobertaTokenizer, RobertaModel
import os
import time
import json
import h5py
import numpy as np
import torch, torch.utils.data
import math

def open_json(js_file):
    dataset = []
    with open(js_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def extract_features(tTokenizer, tModel, tDataLoader):
    all_featureVec = []
    count = 0
    for text_batch in tDataLoader:
        for text in text_batch:
            encoded_input = tTokenizer(text, return_tensors='pt').to(device)
            tModel = tModel.to(device)
            output = tModel(**encoded_input)
            featureVec = output.pooler_output
            all_featureVec.append(featureVec)
        if (count % 3) == 0:
            print("Processing: ", count * batch_size)
        count = count + 1
    return all_featureVec

def save_features(all_feature_vectors, jsonl_file, out_path, part = ""):
    # Save the extracted features in a h5 file
    save_file_name = "roberta_features_" + os.path.splitext(jsonl_file)[0] + part + ".h5"
    h5file = h5py.File(os.path.join(out_path, save_file_name), mode='w')
    h5file.create_dataset("default", data=all_feature_vectors, dtype=np.float32)
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
    jsonlFile = "dev_seen.jsonl"  #  "test_seen.jsonl"  # "train.jsonl"

    start_time = time.time()
    temp = torch.cuda.is_available()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 100

    # Load the pretrained model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    # Read jsonl file:
    myData = open_json(os.path.join(inPath, jsonlFile))

    # Get text
    all_text = []
    for i in range(0, len(myData)):
        all_text.append(myData[i]['text'])

    # Create a temporal folder
    temp_outPath = os.path.join(outPath, "temp_split")
    if not (os.path.exists(temp_outPath)):
        os.makedirs(temp_outPath)

    # split the dataset into parts
    split_size = 1000
    num_split = int(math.ceil(len(myData) / split_size))
    for j in range(0, num_split):
        start = j * split_size
        end = (j + 1) * split_size
        if j < (num_split - 1):
            data_loader = torch.utils.data.DataLoader(dataset=all_text[start:end], batch_size=batch_size, shuffle=False)
        else:
            data_loader = torch.utils.data.DataLoader(dataset=all_text[start:], batch_size=batch_size, shuffle=False)
        feature_split  = extract_features(tokenizer, model, data_loader)
        feature_split = torch.cat(feature_split, dim=0).to('cpu').detach().numpy()
        save_features(feature_split, jsonlFile, outPath, "_part_" + str(j))
        del data_loader
        print("Processed part: ", j)

    # Concatenate all splits into one file
    all_feat = []
    for j in range(0, num_split):
        txt_save_file_j =  "roberta_features_" + os.path.splitext(jsonlFile)[0] + "_part_" + str(j) + ".h5"
        txt_features_j = read_h5file(os.path.join(outPath, txt_save_file_j))
        all_feat.append(txt_features_j)
    all_feat = torch.cat(all_feat, dim=0).to('cpu').detach().numpy()
    save_features(all_feat, jsonlFile, outPath)
    #check = read_h5file(os.path.join(outPath, "roberta_features_" + os.path.splitext(jsonlFile)[0] + ".h5"))

    print('Running time: ', time.time()-start_time)
