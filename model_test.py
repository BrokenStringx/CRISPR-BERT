import numpy as np
import os
import pandas as pd
from keras.callbacks import *
from keras.layers import *
from keras.utils import np_utils
from model_train import Shuffle, Test_DataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from load_data import loadData
from model_train import Shuffle, Test_DataGenerator
=======

from load_data import loadData_pickle
>>>>>>> 2d74038 (figure update)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Encoder-change import BERT_encode,C_RNN_encode
from model import build_bert
BATCH_SIZE=100
# Negative, Positive, label =loadData('change_CIRCLE_seq_10gRNA_wholeDataset.txt')
Negative, Positive, label =loadData_pickle('II1.pkl')
Negative=Shuffle(Negative)
Positive=Shuffle(Positive)
Train_Validation_Negative, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
Train_Validation_Positive, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)

Xtest = np.vstack((Test_Negative, Test_Positive))
Xtest = Shuffle(Xtest)
Test_Data_token,Test_Data_segment=BERT_encode(Xtest)
Test_DATA_encode=np.array(C_RNN_encode(pd.DataFrame(Xtest)))
test_D  = Test_DataGenerator(Test_Data_token,Test_Data_segment,Test_DATA_encode,batch_size=BATCH_SIZE)

weighs_path = "weight/I1.h5"
model=build_bert()
model.load_weights(weighs_path)

y_pred=model.predict_generator(test_D.__iter__(),steps=len(test_D))
y_test=[]
y_test = [1 if float(i) > 0.0 else 0 for i in Xtest[:, 1]]
y_test = np_utils.to_categorical(y_test)
y_prob = y_pred[:, 1]
y_prob = np.array(y_prob)
y_pred = [int(i[1] > i[0]) for i in y_pred]
y_test = [int(i[1] > i[0]) for i in y_test]
fpr, tpr, au_thres = roc_curve(y_test, y_prob)
auroc = auc(fpr, tpr)
precision, recall, pr_thres = precision_recall_curve(y_test, y_prob)
prauc = auc(recall, precision)
f1score = f1_score(y_test, y_pred)
precision_scores = precision_score(y_test, y_pred)
recall_scores = recall_score(y_test, y_pred)
mcc=matthews_corrcoef(y_test, y_pred)
print("AUROC=%.3f, PRAUC=%.3f, F1score=%.3f, Precision=%.3f, Recall=%.3f,Mcc=%.3f" % (auroc, prauc, f1score, precision_scores, recall_scores,mcc))
