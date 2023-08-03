import os
import tensorflow as tf
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras_bert import load_trained_model_from_checkpoint
config_path     = 'uncased_L-2_H-256_A-4/bert_config.json' 
checkpoint_path = 'uncased_L-2_H-256_A-4/bert_model.ckpt'
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=False)
         
def build_bert():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None) 
    for l in bert_model.layers:
        l.trainable = True
    X_in = Input(shape=(26, 7))
    x1_in = Input(shape=26)
    x2_in = Input(shape=26)
    x_in = Reshape((1, 26, 7))(X_in)
    x_bert = bert_model([x1_in, x2_in])
    conv_1 = Conv2D(5, 1, padding='same', activation='relu')(x_in)
    conv_2 = Conv2D(15, 2, padding='same', activation='relu')(x_in)
    conv_3 = Conv2D(25, 3, padding='same', activation='relu')(x_in)
    conv_4 = Conv2D(35, 5, padding='same', activation='relu')(x_in)
    conv_output = tf.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])
    conv_output = Reshape((26, 80))(conv_output)
    conv_output = Bidirectional(GRU(40, return_sequences=True))(conv_output)  
    x_bert = Bidirectional(GRU(40, return_sequences=True))(x_bert) 
    feature_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
    weight_1 = Lambda(lambda x: x * 0.2)
    weight_2 = Lambda(lambda x: x * 0.8)
    x = feature_concat([weight_1(conv_output), weight_2(x_bert)])
    # x = Lambda(lambda x: x[:, 0])(x)  
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    x = Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.35)(x)
    p = Dense(2, activation='softmax')(x)
    model = Model(inputs=[X_in, x1_in, x2_in], outputs=p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0001),  
                  metrics=['accuracy'])
    print(model.summary())
    return model
