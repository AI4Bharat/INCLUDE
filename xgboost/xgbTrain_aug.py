import pandas as pd
import numpy as np
import xgboost
import time


startTime = time.time()

train = pd.read_csv('xgb_train_aug_v3.csv')
test = pd.read_csv('xgb_test_colShuffleAug.csv')

trainTarget = train.loc[:,-1]
testTarget = test.values[:,-1]

clf = xgboost.XGBClassifier(booster='gbtree' , silent = 0 , max_depth= 2 ,
                            subsample=0.9923301318585108 , colsample_bytree= 0.7747027267489391 , reg_lambda = 3,
                            objective='multi:softprob',tree_method = 'gpu_hist') # test50 accuracy = 

clf.fit(train.loc[:,'pose_x0_f1':'h2_y20_f169'] , train.values[:,-1])

testPred = clf.predict(test.values[: , 'pose_x0_f1':'h2_y20_f169'])
trainPred = clf.predict(train.values[: , 'pose_x0_f1':'h2_y20_f169'])

temp = 0
for i in range(len(testPred)):
    if testTarget[i] == testPred[i]:
        temp += 1

print('test accuracy = ' , temp / len(testPred))

temp = 0
for i in range(len(trainPred)):
    if trainTarget[i] == trainPred[i]:
        temp += 1

print('train accuracy = ' , temp / len(trainPred))
