import numpy as np
from keras_bert import Tokenizer
class Encoder:
    def __init__(self,data):
        self.data = data
        self.encoded_dict_indel = {'AA': [1, 0, 0, 0, 0, 0, 0],'AT': [1, 1, 0, 0, 0, 1, 0],'AG': [1, 0, 1, 0, 0, 1, 0],'AC': [1, 0, 0, 1, 0, 1, 0],
                                   'TA': [1, 1, 0, 0, 0, 0, 1],'TT': [0, 1, 0, 0, 0, 0, 0],'TG': [0, 1, 1, 0, 0, 1, 0],'TC': [0, 1, 0, 1, 0, 1, 0],
                                   'GA': [1, 0, 1, 0, 0, 0, 1],'GT': [0, 1, 1, 0, 0, 0, 1],'GG': [0, 0, 1, 0, 0, 0, 0],'GC': [0, 0, 1, 1, 0, 1, 0],
                                   'CA': [1, 0, 0, 1, 0, 0, 1],'CT': [0, 1, 0, 1, 0, 0, 1],'CG': [0, 0, 1, 1, 0, 0, 1],'CC': [0, 0, 0, 1, 0, 0, 0],
                                   'A_': [1, 0, 0, 0, 1, 1, 0],'T_': [0, 1, 0, 0, 1, 1, 0],'GX': [0, 0, 1, 0, 1, 1, 0],'C_': [0, 0, 0, 1, 1, 1, 0],
                                   '_A': [1, 0, 0, 0, 1, 0, 1],'_T': [0, 1, 0, 0, 1, 0, 1],'XG': [0, 0, 1, 0, 1, 0, 1],'_C': [0, 0, 0, 1, 1, 0, 1],
                                   '--': [0, 0, 0, 0, 0, 0, 0]}
        self.encode()
    def encode(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        data_bases = list(self.data)
        j=0
        code_list.append(encoded_dict['--'])#
        for i in range(24):
            code_list.append(encoded_dict[data_bases[j]+data_bases[j+1]])
            j=j+3
        code_list.append(encoded_dict['--'])
        self.on_off_code = np.array(code_list)
        
token_dict = {
    '[CLS]': 0,
    '[SEP]': 1,'AA': 2, 'AC': 3, 'AG': 4, 'AT': 5,
                'CA': 6, 'CC': 7, 'CG': 8, 'CT': 9,
                'GA': 10, 'GC': 11, 'GG': 12, 'GT': 13,
                'TA': 14, 'TC': 15, 'TG': 16, 'TT': 17,
                'A_':18,'_A':19,'C_':20,'_C':21,'G_':22,
                '_G':23,'T_':24,'_T':25,'--':26
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