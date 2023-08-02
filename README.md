# CRISPR-BERT
CRISPR-BERT: Enhancing CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT-based convolutional and recurrent neural networks
## Environment
CUDA Version:11.8<br>
Kera-bert 0.89<br>
Keras library 2.4.3<br>
tensorflow-gru 2.5.0
## File Description
Encoder.py Used for encoding sequences<br>
load_data.py Used for loading the data from datasets<br>
CIRCLE_seq_10gRNA_wholeDataset_2.h5 the weights for the CRISPR-BERT model on dataset CIRCLE_seq_10gRNA_wholeDataset<br>
## Training and testing CRISPR-bert
python model_train.py<br>
python model_test.py<br>
