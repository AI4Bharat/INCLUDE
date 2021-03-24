import pandas as pd
import numpy as np
from tqdm import tqdm

trainFiles = pd.read_csv('Train_Test_Split/Include_train_test/train.csv')
testFiles = pd.read_csv('Train_Test_Split/Include_train_test/test.csv')

s = 'INCLUDE_csv/'

filePath = []
for i in range(trainFiles.shape[0]):
    path = trainFiles.iloc[i].loc['FilePath']
    path = path[:-3]
    path = path + 'csv'
    filePath.append(path)

trainFiles['FilePath'] = filePath

filePath = []
for i in range(testFiles.shape[0]):
    path = testFiles.iloc[i].loc['FilePath']
    path = path[:-3]
    path = path + 'csv'
    filePath.append(path)

testFiles['FilePath'] = filePath


max_rows = 154 #coded and verified with INCLUDE paper
cols = 134
features = 134*154 # = 20636

headers = pd.read_csv(s + trainFiles['FilePath'].values[3]).columns.values
print(headers)

headers = headers[1:-1]

longHeader = []
for i in range(154):
    h = [w + '_f' + str(i + 1) for w in headers]
    longHeader += h


trainDf = pd.DataFrame(index= trainFiles['FilePath'].values , columns= longHeader)
trainDf.index.name = 'filePath'
testDf = pd.DataFrame(index= testFiles['FilePath'].values , columns= longHeader)
testDf.index.name = 'filePath'

print(trainDf.shape , testDf.shape)


for i in tqdm(range(trainDf.shape[0])):
    path = s + trainFiles.iloc[i].loc['FilePath']
    df = list(pd.read_csv(path).values[:,1:-1].astype(float).flatten())
    df += [0] * (20636 - len(df))
    trainDf.iloc[i] =df


for i in tqdm(range(testDf.shape[0])):
    path = s + testFiles.iloc[i].loc['FilePath']
    df = list(pd.read_csv(path).values[:,1:-1].astype(float).flatten())
    df += [0] * (20636 - len(df))
    testDf.iloc[i] =df

trainDf['target'] = trainFiles['Word'].values
testDf['target'] = testFiles['Word'].values
print(trainDf.head())
print(testDf.head())


trainDf.to_csv('xgb_train_INCLUDE.csv')
testDf.to_csv('xgb_test_INCLUDE.csv')
