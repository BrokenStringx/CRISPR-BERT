import numpy as np
from keras_bert import Tokenizer
class Encoder:
    def __init__(self,data):
        self.data = data
        self.encoded_dict_indel = {'aa': [1, 0, 0, 0, 0, 0, 0],'at': [1, 1, 0, 0, 0, 1, 0],'ag': [1, 0, 1, 0, 0, 1, 0],'ac': [1, 0, 0, 1, 0, 1, 0],
                                   'ta': [1, 1, 0, 0, 0, 0, 1],'tt': [0, 1, 0, 0, 0, 0, 0],'tg': [0, 1, 1, 0, 0, 1, 0],'tc': [0, 1, 0, 1, 0, 1, 0],
                                   'ga': [1, 0, 1, 0, 0, 0, 1],'gt': [0, 1, 1, 0, 0, 0, 1],'gg': [0, 0, 1, 0, 0, 0, 0],'gc': [0, 0, 1, 1, 0, 1, 0],  
                                   'ca': [1, 0, 0, 1, 0, 0, 1],'ct': [0, 1, 0, 1, 0, 0, 1],'cg': [0, 0, 1, 1, 0, 0, 1],'cc': [0, 0, 0, 1, 0, 0, 0],
                                   'ax': [1, 0, 0, 0, 1, 1, 0],'tx': [0, 1, 0, 0, 1, 1, 0],'gx': [0, 0, 1, 0, 1, 1, 0],'cx': [0, 0, 0, 1, 1, 1, 0],
                                   'xa': [1, 0, 0, 0, 1, 0, 1],'xt': [0, 1, 0, 0, 1, 0, 1],'xg': [0, 0, 1, 0, 1, 0, 1],'xc': [0, 0, 0, 1, 1, 0, 1], 
                                   'xx': [0, 0, 0, 0, 0, 0, 0]}
        self.encode()
    def encode(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        data_bases = list(self.data)
        j=0
        code_list.append(encoded_dict['xx'])#26进位补两个xx
        for i in range(24):
            code_list.append(encoded_dict[data_bases[j]+data_bases[j+1]])
            j=j+3
        code_list.append(encoded_dict['xx'])
        self.on_off_code = np.array(code_list)
        
token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,'aa': 2, 'ac': 3, 'ag': 4, 'at': 5,
                'ca': 6, 'cc': 7, 'cg': 8, 'ct': 9,
                'ga': 10, 'gc': 11, 'gg': 12, 'gt': 13,
                'ta': 14, 'tc': 15, 'tg': 16, 'tt': 17,
                'ax':18,'xa':19,'cx':20,'xc':21,'gx':22,
                'xg':23,'tx':24,'xt':25,'xx':26
}
tokenizer = Tokenizer(token_dict)
def BERT_encode(data):
    idxs=list(range(len(data)))
    X1, X2= [], []
    for i in idxs:
         text,y= data[i]
         x1, x2 = tokenizer.encode(text)
         X1.append(x1)
         X2.append(x2)
         i=i+1
    return X1,X2
def C_RNN_encode(data):
    encode=[]
    for idx, row in data.iterrows():
        en = Encoder(row[0])
        encode.append(en.on_off_code)
    return encode