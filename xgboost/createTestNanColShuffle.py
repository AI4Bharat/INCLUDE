from tqdm import tqdm
import pandas as pd
import numpy as np


# In[2]:


test = pd.read_csv('xgb_test_INCLUDE.csv')



# # Reorder test columns

new_test = pd.DataFrame()
new_test['filePath'] = test['filePath']
for f in tqdm(range(1,155)):
    for p in range(25):
        new_test['pose_x' + str(p) + '_f' + str(f)] = test['pose_x' + str(p) + '_f' + str(f)]
        new_test['pose_y' + str(p) + '_f' + str(f)] = test['pose_y' + str(p) + '_f' + str(f)]
    for h1 in range(21):
        new_test['h1_x' + str(h1) + '_f' + str(f)] = test['h1_x' + str(h1) + '_f' + str(f)]
        new_test['h1_y' + str(h1) + '_f' + str(f)] = test['h1_y' + str(h1) + '_f' + str(f)]
    
    for h2 in range(21):
        new_test['h2_x' + str(h2) + '_f' + str(f)] = test['h2_x' + str(h2) + '_f' + str(f)]
        new_test['h2_y' + str(h2) + '_f' + str(f)] = test['h2_y' + str(h2) + '_f' + str(f)]

new_test['target'] = test['target']

header = list(test.columns)[1:-1]
keyHeader = header[:134]
keyHeader = [ keyHeader[i][:-1] for i in range(len(keyHeader))  ]
appendHeader = []
for i in range(155,170):
    appendHeader += [ keyHeader[k] + str(i) for k in range(len(keyHeader))]

new_test = new_test.reindex(columns = new_test.columns.tolist()[:-1] + appendHeader + [new_test.columns.tolist()[-1]])


new_test.to_csv('xgb_test_colShuffleAug.csv')

