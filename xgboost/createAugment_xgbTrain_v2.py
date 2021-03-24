from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import sklearn
import seaborn as sn


train = pd.read_csv('xgb_train_INCLUDE.csv')
m = train.shape[0]

# # Reorder train columns

print('Reorder train columns')

new_train = pd.DataFrame()
new_train['filePath'] = train['filePath']
for f in tqdm(range(1,155)):
    for p in range(25):
        new_train['pose_x' + str(p) + '_f' + str(f)] = train['pose_x' + str(p) + '_f' + str(f)]
        new_train['pose_y' + str(p) + '_f' + str(f)] = train['pose_y' + str(p) + '_f' + str(f)]
    for h1 in range(21):
        new_train['h1_x' + str(h1) + '_f' + str(f)] = train['h1_x' + str(h1) + '_f' + str(f)]
        new_train['h1_y' + str(h1) + '_f' + str(f)] = train['h1_y' + str(h1) + '_f' + str(f)]
    
    for h2 in range(21):
        new_train['h2_x' + str(h2) + '_f' + str(f)] = train['h2_x' + str(h2) + '_f' + str(f)]
        new_train['h2_y' + str(h2) + '_f' + str(f)] = train['h2_y' + str(h2) + '_f' + str(f)]

new_train['target'] = train['target']


# # Theta = +7 degrees Augmentation

print('+7 degrees augmentation')

theta = 7 * (np.pi / 180)
c , s = np.cos(theta) , np.sin(theta)
augmentMatrix = np.array([[c , -s] , [s , c]])
augList = []
    
new_train = new_train.reindex(new_train.index.to_list() + list(range(m , m*2)))

for i in tqdm(range(train.shape[0])):
    arr = new_train.iloc[i].values[1:-1]
    temp_arr = new_train.iloc[i].values
    arr = arr.reshape((-1 , 2))
    arr = np.matmul(arr , augmentMatrix)
    arr = arr.flatten()
    temp_arr[1:-1] = arr
    augList.append(temp_arr)
    


# In[8]:


new_train.at[m: , :] = augList


# # Theta = -7 degrees Augmentation



print('-7 degrees augmentation')
theta = -7 * (np.pi / 180)
c , s = np.cos(theta) , np.sin(theta)
augmentMatrix = np.array([[c , -s] , [s , c]])
augList = []

new_train = new_train.reindex(new_train.index.to_list() + list(range(m*2 , m*3)))

for i in tqdm(range(train.shape[0])):
    arr = new_train.iloc[i].values[1:-1]
    temp_arr = new_train.iloc[i].values
    arr = arr.reshape((-1 , 2))
    arr = np.matmul(arr , augmentMatrix)
    arr = arr.flatten()
    temp_arr[1:-1] = arr
    augList.append(temp_arr)


new_train.at[m*2: , :] = augList

# # Random gaussian sampling
print('Gaussian sampling')
new_train = new_train.reindex(new_train.index.to_list() + list(range(m*3 , m*4)))


# In[38]:


#print(new_train.columns.values[1:135])
dv = 0.05 * 10**-2
sv = 0.08 * 10**-2
lv = 0.08 * 10**-1
sigma = [sv , dv,dv,dv,dv,dv,dv, sv,sv,sv,sv, lv,lv,lv,lv,
         sv,sv,sv,sv,sv,sv,sv,sv, lv,lv]

sigma = sigma * 2 + [dv] * 84
sigma = sigma * 154


# In[41]:



augList = []

for i in tqdm(range(train.shape[0])):
    arr = new_train.iloc[i].values[1:-1].astype('float')
    #print(len(arr))
    arr = arr.reshape((-1 , 2))
    arr[:,0] /= 1920
    arr[:,1] /= 1080
    arr = arr.flatten()
    arr = np.random.normal(arr , sigma)
    arr = arr.reshape((-1 , 2))
    arr[:,0] *= 1920
    arr[:,1] *= 1080
    arr = arr.flatten()
    temp_arr = new_train.iloc[i].values
    temp_arr[1:-1] = arr
    augList.append(temp_arr)


# In[14]:


new_train.at[m*3: , :] = augList


# # cutout , random nans

# In[42]:

print('cutout: randomly 1500/20636 points made nan')
new_train = new_train.reindex(new_train.index.to_list() + list(range(m*4 , m*5)))


# In[73]:


nanCount = 750
augList = []

for i in tqdm(range(train.shape[0])):
    arr = new_train.iloc[i].values[1:-1].astype('float')
    temp_arr = new_train.iloc[i].values
    nanIdx = np.random.choice(int(len(arr)/2), nanCount, replace=False)
    arr[nanIdx] = np.nan
    arr[nanIdx+1] = np.nan
    temp_arr[1:-1] = arr
    augList.append(temp_arr)


new_train.at[m*4: , :] = augList


# # Downsample : shift rows left

print('randomly 10th frame removed, total 15 frames removed:')
new_train = new_train.reindex(new_train.index.to_list() + list(range(m*5 , m*6)))

# choose an frame number 'idx', remove 
augList = []

for i in tqdm(range(train.shape[0])):
    arr = new_train.iloc[i].values[1:-1].astype('float')
    temp_arr = new_train.iloc[i].values
    drop_idx = np.arange(1,16) * 10 #154 frames , drop every 15th frame
    for idx in drop_idx:
        arr = np.delete(arr ,np.arange(134 * (idx-1) , 134 * idx))
        arr = np.array(list(arr) + [np.nan]*134)
    temp_arr[1:-1] = arr
    augList.append(temp_arr)

new_train.at[m*5: , :] = augList


new_train.to_csv('xgb_train_aug_v2.csv')

