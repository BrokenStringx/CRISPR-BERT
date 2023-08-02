import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from keras.callbacks import *
from keras.callbacks import ReduceLROnPlateau
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from load_data import loadData
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Encoder import Encoder,token_dict,BERT_encode,C_RNN_encode
from model import build_bert

def Train_DataGenerator(Negative_token,Negative_segment,Positive_token,Positive_segment,Negative_encode,Positive_encode,batchsize,positive_num,negative_num):

    Num_Positive, Num_Negative = len(Positive_token), len(Negative_token)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
    Index_Negative = [i for i in range(Num_Negative)]
    np.random.seed(2020)
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    X1_input =[]
    X2_input=[]
    X_input=[]
    Y_input = [] # 
 
    while True:
        for i in range(Total_num_batch):
                Negative_num=int(negative_num*(batchsize/50))
                for j in range(Negative_num):
                    x1, x2=Negative_token[Index_Negative[j + i *Negative_num]],Negative_segment[Index_Negative[j + i * Negative_num]]
                    X_input.append(Negative_encode[Index_Negative[j + i * Negative_num]])
                    X1_input.append(x1)
                    X2_input.append(x2)
                    Y_input.append(0)
                for k in range(int((positive_num*batchsize/50))):
                    x1, x2=Positive_token[Index_Positive[k]],Positive_segment[Index_Positive[k]]
                    X_input.append(Positive_encode[Index_Positive[k]])
                    X1_input.append(x1)
                    X2_input.append(x2)
                    Y_input.append(1)

                Y_input = np_utils.to_categorical(Y_input)
                yield [np.array(X_input),np.array(X1_input),np.array(X2_input)],np.array(Y_input)
                X1_input = []
                X2_input=[]
                X_input=[]
                Y_input = []
                Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                


def DataGenerator(Negative_token,Negative_segment,Positive_token,Positive_segment,Negative_encode,Positive_encode,batchsize,positive_num,negative_num):
    
    Num_Positive, Num_Negative = len(Positive_token), len(Negative_token)
    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
    Index_Negative = [i for i in range(Num_Negative)]
    np.random.seed(2020)
    random.shuffle(Index_Negative)
    Total_num_batch = int(Num_Negative / batchsize)
    X1_input =[]
    X2_input=[]
    X_input=[] 
    Y_input = []

    while True:
        for i in range(Total_num_batch):
    
                Negative_num=int(negative_num*(batchsize/50))
                for j in range(Negative_num):
                    x1, x2=Negative_token[Index_Negative[j + i *Negative_num]],Negative_segment[Index_Negative[j + i * Negative_num]]
                    X_input.append(Negative_encode[Index_Negative[j + i * Negative_num]])
                    X1_input.append(x1)
                    X2_input.append(x2)
                    Y_input.append(0)
        
                for k in range(int(positive_num*(batchsize/50))):
                    x1, x2=Positive_token[Index_Positive[k]],Positive_segment[Index_Positive[k]]
                    X_input.append(Positive_encode[Index_Positive[k]])
                    X1_input.append(x1)
                    X2_input.append(x2)
                    Y_input.append(1)
                Y_input = np_utils.to_categorical(Y_input)
                yield [np.array(X_input),np.array(X1_input),np.array(X2_input)],np.array(Y_input)
                X_input=[]
                X1_input = []
                X2_input=[]
                Y_input = []
                Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')

class Test_DataGenerator:
    def __init__(self, Test_Data_token,Test_Data_segment, Test_Data_encode,batch_size):
        self.Test_Data_token = Test_Data_token
        self.Test_Data_segment=Test_Data_segment
        self.Test_Data_encode=Test_Data_encode
        self.batch_size = batch_size
        self.steps = len(self.Test_Data_token) // self.batch_size
        if len(self.Test_Data_token) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.Test_Data_token)))
            X,X1, X2= [], [],[]
            for i in idxs:
                x=self.Test_Data_encode[i]
                x1, x2 = self.Test_Data_token[i],self.Test_Data_segment[i]
                X.append(x)
                X1.append(x1)
                X2.append(x2)
                
                if len(X1) == self.batch_size or i == idxs[-1]:
                    yield [np.array(X),np.array(X1), np.array(X2)]
                    [X, X1, X2] = [], [], []
                    
def Shuffle(x):
    x=np.array(x)
    Seq = [i for i in range(len(x))]  # 列表解析
    random.shuffle(Seq)  # 将序列中所有元素随机排序
    x = x[Seq]  # 将样本顺序打乱
    x = np.array(x)
    return x

model = build_bert()


AUROC = []
PRAUC = []
F1_SCORE = []
PRECISION = []
RECALL = []
MCC = []
Negative, Positive, label =loadData('data/change_CIRCLE_seq_10gRNA_wholeDataset.txt')

IR=len(Positive)/len(Negative)
positive_num=0
negative_num=0
positive_weight=0
if IR>250:
    if IR>2000:
        positive_weight=50
        positive_num=25
        negative_num=25
        
    else:
        positive_weight=30
        positive_num=20
        negative_num=30
else:
    positive_weight=5   
    positive_num=15
    negative_num=35
class_weights = {0: 0.5, 1: positive_weight}


Negative=Shuffle(Negative)
Positive=Shuffle(Positive)

Train_Validation_Negative, Test_Negative = train_test_split(Negative, test_size=0.15, random_state=42)
Train_Validation_Positive, Test_Positive = train_test_split(Positive, test_size=0.15, random_state=42)
Train_Negative, Validation_Negative = train_test_split(Train_Validation_Negative, test_size=0.2, random_state=42)
Train_Positive, Validation_Positive = train_test_split(Train_Validation_Positive, test_size=0.2, random_state=42)
Train_Negative_token,Train_Negative_segment=BERT_encode(Train_Negative)
Validation_Negative_token,Validation_Negative_segment=BERT_encode(Validation_Negative)
Train_Positive_token,Train_Positive_segment=BERT_encode(Train_Positive)
Validation_Positive_token,Validation_Positive_segment=BERT_encode(Validation_Positive)
Train_Positive = pd.DataFrame(Train_Positive)
Train_Negative = pd.DataFrame(Train_Negative)
Validation_Positive = pd.DataFrame(Validation_Positive)
Validation_Negative = pd.DataFrame(Validation_Negative)
Train_Negative_encode=np.array(C_RNN_encode(Train_Negative))
Validation_Negative_encode=np.array(C_RNN_encode(Validation_Negative))
Train_Positive_encode=np.array(C_RNN_encode(Train_Positive))
Validation_Positive_encode=np.array(C_RNN_encode(Validation_Positive))




Xtest = np.vstack((Test_Negative, Test_Positive)) 
Xtest = Shuffle(Xtest)
Test_Data_token,Test_Data_segment=BERT_encode(Xtest)
Test_DATA_encode=np.array(C_RNN_encode(pd.DataFrame(Xtest)))
BATCH_SIZE=256
Train_NUM_BATCH = int(len(Train_Negative) / BATCH_SIZE)
Valid_NUM_BATCH= int(len(Validation_Negative) / BATCH_SIZE)


test_D  = Test_DataGenerator(Test_Data_token,Test_Data_segment,Test_DATA_encode,batch_size=BATCH_SIZE)
earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=25, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',min_delta=0.01,factor=0.1, patience=10,min_lr=0.000001, verbose=1)


EPOCH=30
num = 5
if __name__ == '__main__':
    for i in range(num):
        print("processing fold #", i + 1)
        print("begin model training...")
        model.fit_generator(
                            Train_DataGenerator(Train_Negative_token,Train_Negative_segment,Train_Positive_token,Train_Positive_segment,Train_Negative_encode,Train_Positive_encode,batchsize=BATCH_SIZE,positive_num=positive_num,negative_num=negative_num),
                            validation_data=DataGenerator(Validation_Negative_token,Validation_Negative_segment,Validation_Positive_token,Validation_Positive_segment,Validation_Negative_encode,Validation_Positive_encode,batchsize=BATCH_SIZE,positive_num=positive_num,negative_num=negative_num),
                            callbacks=[reduce_lr,earlyStop],
                            epochs=EPOCH,
                            class_weight=class_weights,
                            steps_per_epoch=Train_NUM_BATCH,
                            validation_steps=1,)

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

