#!/usr/bin/env python
# coding: utf-8

# In[41]:


import sys
import argparse
from datetime import datetime
# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import random
import time

#replace with pytorch lightning
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

from indx import *
from query import *

parser = argparse.ArgumentParser(description='Take inputs')
parser.add_argument('epochs', metavar='epochs', type=int,
                    help='enter epochs')
parser.add_argument('batch_size', metavar='batch_size', type=int,
                    help='enter batch size')
parser.add_argument('lr', metavar='lr', type=float,
                    help='enter lr')
parser.add_argument('nof', metavar='nof', type=int,
                    help='enter no. of features')
parser.add_argument('topk', metavar='topk', type=int,
                    help='enter topk')
parser.add_argument('max_steps', metavar='max_steps', type=int,
                    help='enter max_steps')
parser.add_argument('nprobe', metavar='nprobe', type=int,
                    help='enter nprobe')
parser.add_argument('nqueries', metavar='nqueries', type=int,
                    help='enter nqueries')
parser.add_argument('project_name', metavar='project_name', type=str,
                    help='enter project_name')
parser.add_argument('--random_size', metavar='random_size', nargs="+",type=int,
                    help='enter random_size')


args = parser.parse_args()
# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device(device)
print(device)


# In[4]:

print(args.nof)
df=pd.read_csv("HIGGS.csv",header=None)
df.head()


# In[5]:


#subsample

df = df[:args.nof]
# df = df[:100000]


# In[6]:


print(df.shape)


# In[7]:


sns.countplot(x = 0, data=df)


# In[8]:


X = df.iloc[:, 1:]
y = df.iloc[:, 0]


# In[17]:


EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 2
TOPK=args.topk
MAX_STEPS=args.max_steps
FLAG=True
NPROBE=args.nprobe
NQUERIES=args.nqueries
METRIC=" NO FAISS EXACT NEIGHBORS"
RANDOM_SIZE=args.random_size
PROJECT_NAME=args.project_name
# EPOCHS = 2
# BATCH_SIZE = 60
# LEARNING_RATE = 0.0007
# NUM_FEATURES = len(X.columns)
# NUM_CLASSES = 2
# TOPK=10
# MAX_STEPS=20
# FLAG=True
# NPROBE=10
# NQUERIES=BATCH_SIZE
# METRIC=" NO FAISS EXACT NEIGHBORS"
# RANDOM_SIZE=[30]
# PROJECT_NAME="TRIAL_SCRIPT"


# In[18]:


#CREATING THE SWEEP CONFIG
sweep_config = {
    'method': 'grid'
    }

parameters_dict = {
    "LEARNING_RATE": {
        'value': LEARNING_RATE
        },
    "NUM_FEATURES": {
        'value': NUM_FEATURES
        },
    "NUM_CLASSES": {
        'value': NUM_CLASSES
        },
    "EPOCHS": {
        'value': EPOCHS
        },
    "BATCH_SIZE": {
        'value': BATCH_SIZE
        },
    "TopK": {
        'value': TOPK
        },
    "MAX_STEPS": {
        'value': MAX_STEPS
        },
    "Data_Points": {
        'value': args.nof
        },
    "LSH_BATCH": {
        'value': FLAG
        },
    "NPROBE": {
        'value': NPROBE
        },
    "NQUERIES": {
        'value': BATCH_SIZE
        },
    "METRIC": {
        'value': METRIC
        },
    "RANDOM_SIZE":{
        'values':RANDOM_SIZE
        }
    }
# 
sweep_config['parameters'] = parameters_dict


# In[147]:


sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)


# In[128]:





# In[20]:


# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)


# In[21]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[22]:


#split training data into two subarrays based on class
X_train_class=[]
for i in range(NUM_CLASSES):
    X_train_class.append(np.ascontiguousarray(X_train[y_train == i]))
print(X_train_class[0].shape)
# #Create Faiss index for each class
# index(X_train_class[0],"IVF1024,PQ28","CLASS0")
# index(X_train_class[1],"IVF1024,PQ28","CLASS1")


# In[91]:


HIST0=[]
HIST1=[]


# In[24]:


def get_class_distribution(obj):
    count_dict = {
        "class_0": 0,
        "class_1": 0,
    }
    
    for i in obj:
        if i == 0: 
            count_dict['class_0'] += 1
        elif i == 1: 
            count_dict['class_1'] += 1           
        else:
            print("Check classes.")
            
    return count_dict


# In[25]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
# Train
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
# Validation
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Val Set')
# Test
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')


# In[26]:


class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())


# In[27]:


target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)


# In[28]:


class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(class_weights)


# In[29]:


class_weights_all = class_weights[target_list]


# In[30]:


weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)


# In[31]:


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=8192)
val_loader = DataLoader(dataset=val_dataset, batch_size=8192)
test_loader = DataLoader(dataset=test_dataset, batch_size=8192)


# In[32]:


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
#         x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
#         x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x


# In[33]:


import torch
from torch import autograd
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax()

    def __init__(self):
        super().__init__()
        
    def forward(self, x, target):
        log_softmax =  x - x.exp().sum(-1).log().unsqueeze(-1)
        loss = -log_softmax[range(target.shape[0]), target]
#         print(loss)
        return loss,loss.mean(), loss.argsort()


# In[34]:


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc


# In[98]:


def faissQuery(X_train_class, Queries,nprobe,nqueries):
    
    tempbatch=set()
    for i in range(NUM_CLASSES):
        for j in range(len(Queries[i])):
            LOSSES=np.sum(np.square(X_train_class[i]-Queries[i][j].cpu().numpy()),1)
            idx=LOSSES.argsort()[1:nqueries] #starting from 1 because 0th index will be the same element
            for k in range(len(idx)):
                tempbatch.add((idx[k],i))
    return tempbatch


# In[131]:


#Should shuffle after creation?
#random_size is controlling the randomness in batch creation, random_size=0 means 100% random samples
#random_size=BATCH_SIZE/2 means 50% random samples and 50% LSH samples.
#so random_size should be set as per the BATCH_SIZE such that BATCH_SIZE-random_size)/2 is Integer
def createBatch(tempbatch,BATCH_SIZE,HIST0,HIST1,NN=0):
    tempbatch=list(tempbatch)
#     loss=sort(loss)
    LSHbatch=random.choices(range(len(tempbatch)),k=NN)
    batchX = [X_train_class[tempbatch[x][1]][tempbatch[x][0]] for x in LSHbatch]
    batchY = [tempbatch[x][1] for x in LSHbatch]
    for x in LSHbatch:
        if tempbatch[x][0]==0:
            HIST0.append(tempbatch[x][0])
        else:
            HIST1.append(tempbatch[x][0])

    randomBatch0 = random.choices(X_train_class[0], k=int((BATCH_SIZE-NN)/2))
    randomBatch1 = random.choices(X_train_class[1], k=int((BATCH_SIZE-NN)/2))
    if NN==BATCH_SIZE:
        ("")
    elif NN!=0:
        batchX = np.concatenate((batchX, randomBatch0, randomBatch1))
        batchY= np.concatenate((batchY, [0]*int((BATCH_SIZE-NN)/2), [1]*int((BATCH_SIZE-NN)/2)))
    else:
        batchX = np.concatenate((randomBatch0, randomBatch1))
        batchY= np.concatenate(([0]*int(BATCH_SIZE/2), [1]*int(BATCH_SIZE/2)))
    
    batchX, batchY = shuffle(batchX, batchY)
    
    return np.array(batchX), np.array(batchY).astype(np.int64)


# In[44]:

random.seed(2020)
temp_indices=list(range(0,len(X_train)))
temp_indices=random.choices(temp_indices,k=BATCH_SIZE)
random.seed(datetime.now())

# In[122]:


#RUN THIS ONLY ONCE
# model=model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
# torch.save(model.state_dict(), 'model_weights_same_inits-2.pth')


# In[40]:


LSHBatching=True


# In[146]:


def training_custom(config=None):
    # INITIALIZE NEW WANDB RUN
    with wandb.init(config=config) as run:
        #USING THE CONFIG TO SET THE HYPERPARAMETERS FOR EACH RUN
        config = wandb.config
        wandb.define_metric("custom_step")
        wandb.define_metric("Train Loss", step_metric='custom_step')
        wandb.define_metric("Val Loss", step_metric='custom_step')
        wandb.define_metric("Train Accuracy", step_metric='custom_step')
        wandb.define_metric("Val Accuracy", step_metric='custom_step')
        wandb.define_metric("validation/loss", step_metric='custom_step')
        run.name = "NN-"+str(config.RANDOM_SIZE)
        print("Begin training.")
        #INITIALIZE NEW MODEL FOR EVERY RUN
        model = MulticlassClassification(num_feature = config.NUM_FEATURES, num_class=config.NUM_CLASSES)
        model.load_state_dict(torch.load('model_weights_same_inits-2.pth'))
        model.to(device)

        criterion = CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        oldBatch = set()
        #INITIALIZE STATS FOR EVERY RUN
        accuracy_stats = {
            'train': [],
            "val": []
        }
        loss_stats = {
            'train': [],
            "val": []
        }
        #FIRST BATCH FOR EVERY RUN SHOULD BE THE SAME, TEMP_INDICES HAS BEEN RUN ONCE AND WILL BE USED EVERYTIME
        X_train_LSH=X_train[temp_indices]
        y_train_LSH=y_train[temp_indices].astype(np.int64)
        for e in tqdm(range(1, config.EPOCHS)):
            
            # TRAINING
            model.train()
            step=0
            for steps in tqdm(range(1, config.MAX_STEPS+1)):
                X_train_batch= torch.from_numpy(X_train_LSH).to(device)
                y_train_batch=torch.from_numpy(y_train_LSH).to(device)

                optimizer.zero_grad()
                
                y_train_pred = model(X_train_batch.float())
                
                loss,train_loss, sorted_index = criterion(y_train_pred, y_train_batch)
                
                gradients_batch=[]
                for b in range(len(loss)):
                    optimizer.zero_grad()
                    loss[b].backward(retain_graph=True)
                    sum_grad=0
                    for x in model.parameters():
                        sum_grad+=torch.sum(torch.square(x.grad))
                    gradients_batch.append(sum_grad.cpu())
                    
                optimizer.zero_grad()
                if LSHBatching:
                    #PROCEDURE FOR NEW BATCHES
                    with torch.no_grad():
                        sorted_index=np.argsort(gradients_batch)
                        X_train_batch_Topk = X_train_batch[sorted_index[-config.TopK:]]
                        y_train_batch_Topk = y_train_batch[sorted_index[-config.TopK:]]

                        #Query for nearest neighbors
                        
                        nextBatch = faissQuery(X_train_class,[X_train_batch_Topk[y_train_batch_Topk == 0], X_train_batch_Topk[y_train_batch_Topk == 1]],config.NPROBE,config.NQUERIES)
                        
                        oldBatch = nextBatch

                        X_train_LSH,y_train_LSH=createBatch(nextBatch,config.BATCH_SIZE,HIST0,HIST1,config.RANDOM_SIZE)

                else:
                    with torch.no_grad():
                        X_train_LSH,y_train_LSH=createBatch(nextBatch,config.BATCH_SIZE,HIST0,HIST1,config.RANDOM_SIZE)
                ##############################

                
                train_loss.backward()
                
                with torch.no_grad():
                    delta=torch.sum(torch.square(model.layer_1.weight.grad))+torch.sum(torch.square(model.layer_2.weight.grad))+torch.sum(torch.square(model.layer_3.weight.grad))+torch.sum(torch.square(model.layer_out.weight.grad))
                    wandb.log({f"Gradients {e}":delta,f"Train Loss {e}":train_loss,'custom_step2':steps})
                    
                    
                optimizer.step()


            with torch.no_grad():
                val_epoch_loss = 0
                val_epoch_acc = 0
                train_epoch_loss=0
                train_epoch_acc=0
                model.eval()
                # VALIDATION LOSS AND ACCURACY
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                    y_val_pred = model(X_val_batch)

                    val_l,val_loss, _ = criterion(y_val_pred, y_val_batch)
                    val_acc = multi_acc(y_val_pred, y_val_batch)

                    val_epoch_loss += val_loss.item()
                    val_epoch_acc += val_acc.item()
                    break;
                # TRAIN LOSS AND ACCURACY

                for X_train_batch, y_train_batch in train_loader:

                    X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

                    y_train_pred = model(X_train_batch)

                    train_l,train_loss, _ = criterion(y_train_pred, y_train_batch)
                    train_acc = multi_acc(y_train_pred, y_train_batch)

                    train_epoch_loss += train_loss.item()
                    train_epoch_acc += train_acc.item()
                    break;

            loss_stats['train'].append(train_epoch_loss)
            loss_stats['val'].append(val_epoch_loss)
            accuracy_stats['train'].append(train_epoch_acc)
            accuracy_stats['val'].append(val_epoch_acc)


            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f} | Train Acc: {train_epoch_acc:.3f}| Val Acc: {val_epoch_acc:.3f}')
            #LOGGING LOSS AND ACCURACY TO WANDB
            wandb.log({"Train Loss":train_epoch_loss, "Val Loss":val_epoch_loss,"Train Accuracy":train_epoch_acc,"Val Accuracy":val_epoch_acc, 'custom_step':e})
        plt.hist(HIST0,label='class0')
        plt.hist(HIST1,label='class1')
        plt.legend(loc='upper right')
        plt.show()
        


# In[148]:


#SWEEPING OVER THE HYPERPARAMETERS
wandb.agent(sweep_id, training_custom)

#NO NEED TO RUN ANYTHING BEYOND THIS CELL


# In[ ]:




