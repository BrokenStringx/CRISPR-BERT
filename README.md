# CRISPR-BERT
CRISPR-BERT: Enhancing CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT-based convolutional and recurrent neural networks
## Environment
[CUDA](https://developer.nvidia.com/cuda-toolkit) is necessary for training model in GPU：
CUDA Version:11.8<br>
<br>
The Python packages should be installed :<br>
* [Kera-bert](https://github.com/CyberZHG/keras-bert) 0.89
* [Keras](https://keras.io/) 2.4.3
* [tensorflow-gpu](https://www.tensorflow.org/install/pip) 2.5.0
* [scikit-learn](https://scikit-learn.org/stable/) 0.24.2
## File description
* Encoder.py: You can used this file to encoding sgRNA-DNA sequences and get two coding format of C_RNN needed and BERT needed.<br>
* load_data.py: Used for loading the data from datasets.
* I1.h5: The weight for the CRISPR-BERT model on this dataset.
* model.py: The CRISPR-BERT model with C_RNN and BERT.
* weight directory: Include BERT weight, we suggest that you can try different scale of [BERT](https://github.com/google-research/bert) models, you can also load other weight of BERT model in model.py.
## Testing CRISPR-BERT
python model_test.py: Run this file to evaluate the CRISPR-BERT model. (Include loading model weight and datasets, demonstrate model performance of six metrics)<br>
## Datasets 
Include 7 public datasets:
* I1->CIRCLE_seq_10gRNA_whole
* I2->elevation_6gRNA_whole
* hek->HEK293t
* k562->K562
* II4
* II5
* II6
