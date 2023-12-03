import numpy as np
import pandas as pd

def loadData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [_.strip() for _ in f.readlines()]
    for i, line in enumerate(lines):
        if i:
            items = line.split(',')
           
            data.append([items[0],items[1],items[2]])
            
    return data


class Encoder:
    def __init__(self, on_seq, off_seq, with_category = False, label = None, with_reg_val = False, value = None):
 
        self.on_seq = on_seq
        self.off_seq =off_seq
        self.encode_sgRNA_DNA()

    def encode_sgRNA_DNA(self):
        code_list = []
        s=''
        sgRNA_bases = self.on_seq
        DNA_bases=self.off_seq
        for i in range(len(sgRNA_bases)):
            if i+1==len(sgRNA_bases):
                if  sgRNA_bases[i]=='_':
                    temp=str('x')+DNA_bases[i]
                elif DNA_bases[i]=='_':
                     temp=sgRNA_bases[i]+str('x')

                elif sgRNA_bases[i]==DNA_bases[i]=='-':
                     temp=str('xx')
               
                else :temp=sgRNA_bases[i]+DNA_bases[i]
                # temp=sgRNA_bases[i]+DNA_bases[i]
                s=s+temp
            else:
                if  sgRNA_bases[i]=='_':
                    temp=str('x')+DNA_bases[i]+' '
                elif DNA_bases[i]=='_':
                     temp=sgRNA_bases[i]+str('x')+' '
                elif sgRNA_bases[i]==DNA_bases[i]=='-':
                     temp=str('xx')+' '
                else :temp=sgRNA_bases[i]+DNA_bases[i]+' '
                # temp=sgRNA_bases[i]+DNA_bases[i]+' '
                s=s+temp
        code_list.append(s)
            # print(code_list)
        self.on_off_code = code_list
        
def CHANGE(DATA):
    data=[]
    for idx, row in DATA.iterrows():
        on_seq = row[0].lower()
        off_seq = row[1].lower()
        label=row[2].lower()
        en = Encoder(on_seq=on_seq, off_seq=off_seq, with_category=True, label=label)
        # print(en.on_off_code)
        
        data.append([en.on_off_code,label])

    return data

data= loadData('datasets/data.txt')
data= pd.DataFrame(data)
data=CHANGE(data)
data= pd.DataFrame(data)
data[0] = data[0].apply(lambda x: ''.join(x[0]))
data[1] = data[1].apply(lambda x: x[0])  

# data.to_csv('data/change_data.csv', index=False, sep=',',header=False)
pickle_out = open("datasets/data.pkl", "wb")
pkl.dump(data, pickle_out)
pickle_out.close()
