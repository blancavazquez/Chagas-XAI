"""
Blanca Vázquez <blanca.vazquez@iimas.unam.mx>
IIMAS, UNAM
2026

-------------------------------------------------------------------------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------------

Code for training MTS models using Random undersampling of the majority class.
"""

import os
import torch
import numpy as np
import pandas as pd
from pypots.optim import Adam
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from utils_training import (tefn, brits, grud, saits, patchtst, timesnet,autoformer, ts2vec)

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly

"""------------------ Hyper-parameters -------------------"""
device = ('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')
epochs = 200 
batch_size = 64 #[32,64,128]
patience = 20 
optimizer = Adam(lr=1e-3)
data_augmentation = "no" 
feature_selection = "si" 
augmentation_rate = 0 
lead_names_ptb = ["DI", "DII", "DIII", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
lead_names = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
lead_names_FE = ['AVR', 'DI', 'DII', 'V1', 'AVF']

""" -----------------------------------------------------"""

def plot_ecg(sig_tensor, lead_names, database,title='ECG Sample'):
    """
    Input: las señales deben ser de tamaño (12, 4096)
    """
    if database == "ptbxl":
        SAMPLING_RATE = 500 
        title = "Original (500 Hz)"
    
    if database == "samitrop":
        SAMPLING_RATE = 400 
        title = "Original (400 Hz)"

    """sig_tensor: (12, T)"""
    fig, axes = plt.subplots(12, 1, figsize=(14, 10), sharex=True)
    t = np.arange(sig_tensor.shape[1]) / SAMPLING_RATE
    for i, ax in enumerate(axes):
        ax.plot(t, sig_tensor[i], lw=0.7, color='royalblue')
        ax.set_ylabel(lead_names[i], fontsize=7, rotation=0, labelpad=20)
        ax.set_yticks([])
        ax.spines[['top','right','left']].set_visible(False)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title + " " + database, fontsize=11)
    plt.tight_layout()
    plt.show()

def labeling(database, num_signals):
    if database == "samitrop":
        y_label = np.full(shape=num_signals, fill_value=1) #Positive label
    else: #ptbxl
        y_label = np.full(shape=num_signals, fill_value=0) #Negative label
    return y_label

def loading_signals(path, folder, time_steps, database):
    path = path+folder
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))] #Get the filename of all files
    print("Total of files: ", len(files), folder)

    DF_per_patient = np.zeros((1,12,time_steps))
    i=1
    for file in files: #Extract signals for each file
        print("file: ", file, len(files), i)
        path_matlab= path+"/"+file
        data_ecg = loadmat(path_matlab)
        signals = data_ecg['muestra'] 
        signals_transpose = signals.T #(4096,12)

        if database == "ptbxl":
            df_ecg = pd.DataFrame(signals_transpose, columns = lead_names_ptb) #(4096, 12)
            df_ecg = df_ecg[lead_names] 
        else:
            signals_transpose = signals_transpose[48:4048] 
            df_ecg = pd.DataFrame(signals_transpose, columns = lead_names) #(4000, 12)

        df_ecg = df_ecg.to_numpy().T #[12,500]
        df_ecg = df_ecg[np.newaxis, :, :] #[1,12,5000]
        DF_per_patient = np.concatenate((df_ecg, DF_per_patient), axis=0)
        i=i+1

    x = DF_per_patient[:-1]
    y = labeling(database, x.shape[0])
    print("x: ", x.shape, "y:", y.shape)
    return x, y


def bandpass_filter(signal, fs, low=0.5, high=40):
    """Filtro pasa-banda """
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def resample_signal(signal, fs_orig, fs_target):
    """ Remuestreo """
    if fs_orig == fs_target:
        return signal
    up = 4
    down = 5
    return resample_poly(signal, up, down)

def normalize(signal):
    """ Normalización (Z-score) """
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def preprocess(signals, fs_orig, fs_target, database):
    print("Before processing (original): ", signals.shape) 
    pd_total = pd.DataFrame()

    if feature_selection == "no":
        DF_all_patients = np.zeros((1,len(lead_names),4000))
    else:
        DF_all_patients = np.zeros((1,len(lead_names_FE),4000))

    for id_pat in range(signals.shape[0]):
        selected_patient = signals[id_pat, :, :] #(12, 5000)

        signals_transpose = selected_patient.T #(5000,12)
        df_ecg = pd.DataFrame(signals_transpose, columns = lead_names)#(5000,12)

        DF_per_patient = pd.DataFrame()
        for lead in lead_names:
            signal = df_ecg[lead].to_numpy() #(5000)
            signal = bandpass_filter(signal, fs_orig)#(5000)
            signal = resample_signal(signal, fs_orig, fs_target) #(4000)
            signal = normalize(signal) #(4000)  
            DF_signal = pd.DataFrame(signal,columns=[lead]) #(4000, 1)        
            DF_per_patient = pd.concat([DF_signal, DF_per_patient], axis = 1) 
        
        if feature_selection == "si":
            DF_per_patient = DF_per_patient[lead_names_FE] 
        DF_per_patient_numpy = DF_per_patient.to_numpy().T #(4000,12)
        DF_per_patient_numpy = DF_per_patient_numpy[np.newaxis, :, :] #[1,12,4000]
        DF_all_patients = np.concatenate((DF_per_patient_numpy, DF_all_patients), axis=0) #(3, 12, 5000)
    x = DF_all_patients[:-1]
    return x

def concat_datasets(xptb, xsamitrop, yptb, ysamitrop,):
    xtrain = np.concatenate((xptb, xsamitrop), axis=0) #(3, 12, 5000)
    ytrain = np.concatenate((yptb, ysamitrop), axis=0)
    print("xtrain: ", xtrain.shape, "ytrain: ", ytrain.shape)
    return xtrain, ytrain 


class Dataset(Dataset):
  #Loading data
  def __init__(self, xtrue, ylabel):
    self.xtrue = xtrue
    self.label = ylabel

  def __len__(self):
    return len(self.xtrue)
  
  def __getitem__(self, i):
    target_in_true = torch.tensor(self.xtrue[i], dtype=torch.float) #torch.Size([6, 8])
    target_out = torch.tensor(self.label[i], dtype=torch.int) #torch.Size([6, 8])
    return target_in_true,target_out

print("\n\n*************************************")
print("*1) Loading and processing(PTBXL)* ")
database = "ptbxl"
path_ptb = "../data/PTB_Aceptadas_500/Split_a1631/"
original_frequency = 500
target_frecuency = 400
time_steps = 5000

xtrain_ptb, ytrain_ptb = loading_signals(path_ptb, "train", time_steps, database)
xval_ptb, yval_ptb = loading_signals(path_ptb, "val", time_steps, database)
xtest_ptb, ytest_ptb = loading_signals(path_ptb, "test", time_steps, database)

xtrain_ptb = preprocess(xtrain_ptb, original_frequency, target_frecuency, database)
xval_ptb = preprocess(xval_ptb, original_frequency, target_frecuency, database)
xtest_ptb = preprocess(xtest_ptb, original_frequency, target_frecuency, database)
print("*************************************")

print("\n\n*************************************")
print("*2) Loading and processing(SAMITROP)* ")
database = "samitrop"
path_samitrop = "../data/SamiTropAceptadas/Split_a1631/"
original_frequency = 400
time_steps = 4000

xtrain_samitrop, ytrain_samitrop = loading_signals(path_samitrop, "train", time_steps, database) #x(batchsize, leads, 4000), y(batchsize)
xval_samitrop, yval_samitrop = loading_signals(path_samitrop, "val", time_steps, database) #x(batchsize, leads, 4000), y(batchsize)
xtest_samitrop, ytest_samitrop = loading_signals(path_samitrop, "test", time_steps, database) #x(batchsize, leads, 4000), y(batchsize)

xtrain_samitrop = preprocess(xtrain_samitrop, original_frequency, target_frecuency, database)#x(batchsize, leads, 4000),
xval_samitrop = preprocess(xval_samitrop, original_frequency, target_frecuency, database)#x(batchsize, leads, 4000),
xtest_samitrop = preprocess(xtest_samitrop, original_frequency, target_frecuency, database)#x(batchsize, leads, 4000),
print("\n\n*************************************")

print("\n\n*************************************")
print("*3) Concat (PTB + SAMITROP)* ")
xtrain, ytrain = concat_datasets(xtrain_ptb, xtrain_samitrop, ytrain_ptb, ytrain_samitrop) #(batchsize, leads, 4000) (batchsize,)
xval, yval = concat_datasets(xval_ptb, xval_samitrop, yval_ptb, yval_samitrop) #(batchsize, leads, 4000) (batchsize,)
xtest, ytest = concat_datasets(xtest_ptb, xtest_samitrop, ytest_ptb, ytest_samitrop) #(batchsize, leads, 4000) (batchsize,)

print("\n\n*************************************")
print("*4) Convert to diccionary* ")
# assemble the final processed data into a dictionary

if feature_selection == "no":
    num_leads_dict = len(lead_names)
else:
    num_leads_dict = len(lead_names_FE)

data = {
        "n_classes": 2,
        "n_steps": num_leads_dict,
        "n_features": 4000,
        "train_X": xtrain,
        "train_y": ytrain,
        "val_X": xval,
        "val_y": yval,
        "test_X": xtest,
        "test_y": ytest,
    }

dataset_for_training = {
    "X": data['train_X'],
    "y": data['train_y'],
}

dataset_for_validating = {
    "X": data['val_X'],
    "y": data['val_y'],
}

dataset_for_testing = {
    "X": data['test_X'],
    "y": data['test_y'],
}

print("dataset_for_training: ", dataset_for_training["X"].shape,dataset_for_training["y"].shape) #(batchsize, leads, 4000) (batchsize,)
print("dataset_for_validating: ", dataset_for_validating["X"].shape,dataset_for_validating["y"].shape) #(batchsize, leads, 4000) (batchsize,)
print("dataset_for_testing: ", dataset_for_testing["X"].shape,dataset_for_testing["y"].shape) #(batchsize, leads, 4000) (batchsize,)

assert dataset_for_training["X"].shape[1] == num_leads_dict

print("\n\n*************************************")
print("*Forecasting* ")

timesnet(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)

ts2vec(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)

patchtst(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)

grud(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)
######################################
tefn(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)

brits(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)

autoformer(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)

saits(data, dataset_for_training,dataset_for_validating,dataset_for_testing, batch_size,epochs,patience,optimizer,data_augmentation, feature_selection, augmentation_rate, device)
