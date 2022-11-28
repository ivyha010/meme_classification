import subprocess
import numpy as np
import argparse
import os
import time
import gc
from collections import Mapping, Container
from sys import getsizeof
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorchtools import EarlyStopping
from Memes_model_2 import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import random
import json

def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, np.unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

# Memory check
def memoryCheck():
    ps = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    print(ps.communicate(), '\n')
    os.system("free -m")

# Free memory
def freeCacheMemory():
    torch.cuda.empty_cache()
    gc.collect()

# Build dataloaders
def myDataloader(imgFeatures, txtFeatures_1, txtFeatures_2, ids, labels, args, shuffleBool=False):
    class my_dataset(Dataset):
        def __init__(self, imgData, txtData_1, txtData_2, id, lbl):
            self.imgData = imgData
            self.txtData_1 = txtData_1
            self.txtData_2 = txtData_2
            self.id = id
            self.lbl = lbl

        def __getitem__(self, index):
            return self.imgData[index], self.txtData_1[index], self.txtData_2[index], self.id[index], self.lbl[index]

        def __len__(self):
            return len(self.imgData)

    # Build dataloaders
    my_dataloader = DataLoader(dataset=my_dataset(imgFeatures, txtFeatures_1, txtFeatures_2, ids, labels), batch_size=args.batch_size, shuffle=shuffleBool)
    return my_dataloader


# TRAIN
def train_func(train_loader, validate_loader, the_model, optimizer, criter, device, n_epochs, patience):
    start_time = time.time()
    # to track the training loss as the model trains
    train_losses = []
    val_losses = []
    # to track the validation loss as the model trains
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):
        # Adjust learning rate
        # adjust_learning_rate(optimizer, epoch)
        # Train the model
        the_model.train()  # prep model for training
        count_batches = 0
        for (img_feature, txt_feature_1, txt_feature_2, _, labels) in train_loader:
            img_feature, txt_feature_1, txt_feature_2, labels = img_feature.to(device), txt_feature_1.to(device), txt_feature_2.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            out = the_model.forward(img_feature, txt_feature_1, txt_feature_2)
            # Loss
            loss =  criter(out, labels.float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            if (count_batches % 100) == 0:
                print('Batch: ', count_batches)
            count_batches += 1

            # Free catch memory
            del img_feature, txt_feature_1, txt_feature_2, labels
            freeCacheMemory()

        # validate the model (for early stopping)
        # prep model for evaluation
        the_model.eval()
        for (vimg_feature, vtxt_feature_1, vtxt_feature_2, _, vlabel) in validate_loader:
            vimg_feature, vtxt_feature_1, vtxt_feature_2, vlabel = vimg_feature.to(device), vtxt_feature_1.to(device), vtxt_feature_2.to(device), vlabel.to(device)
            vout = the_model( vimg_feature, vtxt_feature_1, vtxt_feature_2)
            # validation loss:
            batch_val_losses = criter.forward(vout, vlabel.float())
            val_losses.append(batch_val_losses.item())
            del vout, vimg_feature, vtxt_feature_1, vtxt_feature_2, vlabel
            freeCacheMemory()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        valid_loss = np.average(val_losses)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                     f' train_loss: {train_loss:.8f} ' +
                     f' valid_loss: {valid_loss:.8f} ')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        val_losses = []
        # early_stopping needs the valid_loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss.item(), the_model)
        print('Epoch[{}/{}]: Training time: {} seconds '.format(epoch, n_epochs, time.time() - start_time))
        start_time = time.time()
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    the_model.load_state_dict(torch.load('checkpoint.pt'))
    return the_model, avg_train_losses, avg_val_losses


# VALIDATE
def validate_func(validate_loader, the_model, device):
    the_model.eval()
    all_vout_labels = []
    all_vlabels = []
    all_y_prob = []
    for (vimg_feature, vtxt_feature_1, vtxt_feature_2, _, v_labels) in validate_loader:
        vimg_feature, vtxt_feature_1, vtxt_feature_2, v_labels = vimg_feature.to(device), vtxt_feature_1.to(device), vtxt_feature_2.to(device), v_labels.to(device)
        vout = the_model(vimg_feature, vtxt_feature_1, vtxt_feature_2)
        vout_labels = vout.detach().numpy().round()
        vout_labels = vout_labels.astype(int)
        all_vout_labels.append(vout_labels)
        all_vlabels.append(v_labels.detach().cpu().detach().numpy())
        all_y_prob.append(vout.detach().numpy())

        del vimg_feature, vtxt_feature_1, vtxt_feature_2
        freeCacheMemory()

    all_vout_labels = np.concatenate(all_vout_labels, axis=0)
    all_vlabels = np.concatenate(all_vlabels, axis=0)
    all_y_prob = np.concatenate(all_y_prob, axis=0)

    # Classification accuracy
    val_acc = (all_vout_labels == all_vlabels).mean()
    print('Accuracy: ', val_acc * 100)

    return all_vout_labels, all_vlabels, all_y_prob


# Load extracted feature files and json files
def loadingfiles(feature_file, json_file):
    h5file = h5py.File(feature_file, mode='r')
    getKey = list(h5file.keys())[0]
    getData = h5file.get(getKey)
    features = np.asarray(getData)
    features = torch.from_numpy(features)
    h5file.close()
    # Load IDs and labels
    all_id, all_labels = get_info_json(json_file)
    return features, all_labels, all_id

def get_info_json(js_file):
    # Read json file
    dataset = []
    with open(js_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    # Get IDs and labels
    all_id = []
    all_labels = []
    for i in range(0, len(dataset)):
        all_id.append(dataset[i]['id'])
        all_labels.append(dataset[i]['label'])
    return all_id, all_labels

# Main
def main(args):
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Manual seed
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.random.initial_seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)

    # Data
    # for model training
    img_train_features, train_labels, train_id = loadingfiles(img_feature_file_train, label_file_train)
    txt_train_features_1,_,_ = loadingfiles(txt_1_feature_file_train, label_file_train)
    txt_train_features_2,_,_ = loadingfiles(txt_2_feature_file_train, label_file_train)
    memoryCheck()

    # for model validation
    img_val_features, val_labels, val_id = loadingfiles(img_feature_file_val, label_file_val)
    txt_val_features_1, _, _ = loadingfiles(txt_1_feature_file_val, label_file_val)
    txt_val_features_2, _, _ = loadingfiles(txt_2_feature_file_val, label_file_val)
    memoryCheck()

    # for tesing model
    img_test_features, test_labels, test_id = loadingfiles(img_feature_file_test, label_file_test)
    txt_test_features_1, _, _ = loadingfiles(txt_1_feature_file_test, label_file_test)
    txt_test_features_2, _, _ = loadingfiles(txt_2_feature_file_test, label_file_test)

    # standardize:
    scaler_1 = StandardScaler()
    img_train_features = torch.from_numpy(scaler_1.fit_transform(img_train_features)).float()
    img_val_features = torch.from_numpy(scaler_1.transform(img_val_features)).float()
    img_test_features = torch.from_numpy(scaler_1.transform(img_test_features)).float()

    scaler_2 = StandardScaler()
    txt_train_features_1 = torch.from_numpy(scaler_2.fit_transform(txt_train_features_1)).float()
    txt_val_features_1 = torch.from_numpy(scaler_2.transform(txt_val_features_1)).float()
    txt_test_features_1 = torch.from_numpy(scaler_2.transform(txt_test_features_1)).float()

    scaler_3 = StandardScaler()
    txt_train_features_2 = torch.from_numpy(scaler_3.fit_transform(txt_train_features_2)).float()
    txt_val_features_2 = torch.from_numpy(scaler_3.transform(txt_val_features_2)).float()
    txt_test_features_2 = torch.from_numpy(scaler_3.transform(txt_test_features_2)).float()

    train_dataset = myDataloader(img_train_features, txt_train_features_1, txt_train_features_2, train_id, train_labels, args, True)
    validate_dataset = myDataloader(img_val_features, txt_val_features_1, txt_val_features_2, val_id, val_labels, args, False)
    test_dataset = myDataloader(img_test_features, txt_test_features_1, txt_test_features_2, test_id, test_labels, args, False)

    memoryCheck()

    # input_size for the model
    img_dim = img_train_features.shape[1]
    txt_dim_1 = txt_train_features_1.shape[1]
    txt_dim_2 = txt_train_features_2.shape[1]

    m_start_time = time.time()
    # Build the model
    model = network(img_dim, txt_dim_1, txt_dim_2).to(device)
    model = model.to(device)
    memoryCheck()

    # Loss and optimizer
    # Loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr) #, weight_decay=args.wd, momentum=args.mm)

    model, train_losses, valid_losses = train_func(train_dataset, validate_dataset, model, optimizer, criterion, device, args.num_epochs, args.patience)
    print('Training time: ', time.time() - m_start_time)

    # Validation
    print("Validation set: ")
    val_voutput, val_grdtruth, val_prob = validate_func(validate_dataset, model, device)
    val_auc_score = roc_auc_score(torch.from_numpy(val_grdtruth), val_prob)
    print("auc_roc: ", val_auc_score)

    # Test
    print("Testing set: ")
    test_voutput, test_grdtruth, test_prob  = validate_func(test_dataset, model, device)
    test_auc_score = roc_auc_score(torch.from_numpy(test_grdtruth), test_prob)
    print("auc_roc: ", test_auc_score)

    # Save joint space embedding model
    rndnum = time.time()
    model_name = str(int(rndnum)) + "_model.pth"
    torch.save(model.state_dict(), os.path.join(args.model_path, model_name))

    # Save predicted values:
    afilename = str(int(rndnum)) + "_predicted_output.h5"
    h5file = h5py.File(os.path.join(pred_path, afilename), mode='w')
    h5file.create_dataset('default', data=np.array(test_voutput, dtype=np.int))
    h5file.close()

    os.remove('./checkpoint.pt')


if __name__ == "__main__":
    dir_path = "/home/minhdanh/Downloads/hateful_memes"
    feature_path = os.path.join(dir_path, "extracted_features")  # path to extracted features
    model_path = os.path.join(dir_path, 'models')  # path to save models
    pred_path = os.path.join(dir_path, 'outputs')  # path to outputs

    if not (os.path.exists(os.path.join(dir_path, model_path))):
        os.makedirs(os.path.join(dir_path, model_path))

    if not (os.path.exists(os.path.join(dir_path, pred_path))):
        os.makedirs(os.path.join(dir_path, pred_path))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=model_path, help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience; how long to wait after last time validation loss improved')
    parser.add_argument('--batch_size', type=int, default=128, help='number of feature vectors loaded per batch') # 128
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='initial learning rate') # 0.0005
    #parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    #parser.add_argument('--mm', type=float, default=0.9, help='momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 123)')
    args = parser.parse_args()
    print(args)

    # for training
    label_file_train = os.path.join(dir_path, "train.jsonl")
    img_feature_file_train = os.path.join(feature_path, "img_CLIP_features_train.h5")
    txt_1_feature_file_train = os.path.join(feature_path, "roberta_features_train.h5")
    txt_2_feature_file_train = os.path.join(feature_path, "roberta_features_more_info_train.h5")

    # for validation
    label_file_val = os.path.join(dir_path, "dev_seen.jsonl")
    img_feature_file_val = os.path.join(feature_path, "img_CLIP_features_dev_seen.h5")
    txt_1_feature_file_val = os.path.join(feature_path, "roberta_features_dev_seen.h5")
    txt_2_feature_file_val = os.path.join(feature_path, "roberta_features_more_info_dev_seen.h5")

    # for testing
    label_file_test = os.path.join(dir_path, "test_seen.jsonl")
    img_feature_file_test = os.path.join(feature_path, "img_CLIP_features_test_seen.h5")
    txt_1_feature_file_test = os.path.join(feature_path, "roberta_features_test_seen.h5")
    txt_2_feature_file_test = os.path.join(feature_path, "roberta_features_more_info_test_seen.h5")

    main_start_time = time.time()
    lbl_range = ["Hateful", "Non-Hateful"]  # "Hateful": 1, "Non-Hateful": 0
    main(args)
    print('Total running time: {:.5f} seconds'.format(time.time() - main_start_time))

