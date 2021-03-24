from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('xgb_train_aug_v2.csv')

m = pd.read_csv('xgb_train_INCLUDE.csv').shape[0]

print((train==0.).sum().sum())
train.replace(0, np.nan, inplace=True)
print((train==0.).sum().sum())


augList = []

for i in tqdm(range(m)):
    arr = list( train.iloc[i].values[2:-1].astype('float') )
    temp_arr = train.iloc[i].values
    j=0
    for l in range(10,151,10):
        k = l+j
        arr[k*134:k*134] = (np.array(arr[(k-1)*134 : (k-1+1)*134]) + 
                            np.array(arr[(k+1)*134 : (k+1+1)*134])/2)
        j +=1
    
    arr.insert(0,temp_arr[0])
    arr.insert(1,temp_arr[1])
    arr.append(temp_arr[-1])
    augList.append(arr)



headers = list(train.columns)[2:-1]
#print(headers)
keyHeader = headers[:134]
keyHeader = [ keyHeader[i][:-1] for i in range(len(keyHeader)) ]
appendHeader = []
for i in range(155,170):
    appendHeader +=  [keyHeader[k] + str(i) for k in range(len(keyHeader))] 

#print(train.columns.tolist()[:-1] + appendHeader + [train.columns.tolist()[-1]])

#train  = train.reindex(columns=train.columns.tolist()[:-1] + appendHeader + [train.columns.tolist()[-1]])

target = train[train.columns.tolist()[-1]]
train.drop(columns=[train.columns.tolist()[-1]] , inplace=True)
for h in appendHeader:
    train[h] = 0
train['target'] = target

train = train.reindex(train.index.to_list() + list(range(m*6 , m*7)))
print(train.shape)

train.at[m*6:m*7 , :] = augList

train.to_csv('xgb_train_aug_v3.csv')
