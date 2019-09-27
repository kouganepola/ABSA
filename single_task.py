__author__ = 'Koumudi'
import os
from utils import read_dataset
from custom_metrics import f1
from tensorflow.python.keras.callbacks import TensorBoard, LambdaCallback
from tensorflow.python.keras import initializers, regularizers, optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Embedding, Concatenate
import datetime
import numpy as np


class MTLABSA:
    def __init__(self, embedding_dim=100, batch_size=64, n_hidden=100, learning_rate=0.005, n_class=3, max_sentence_len=40, l2_reg_val=0.003):
        ############################
        self.DATASET = ['laptop']

        self.MODEL_TO_LOAD = './models/mtl_absa_saved_model.h5'
        ###########################
        self.EMBEDDING_DIM = embedding_dim
        self.BATCH_SIZE = batch_size
        self.N_HIDDEN = n_hidden
        self.LEARNING_RATE = learning_rate
        self.N_CLASS = n_class
        self.MAX_SENTENCE_LENGTH = max_sentence_len
        self.EPOCHS = 4
        self.L2_REG_VAL = l2_reg_val
        self.MAX_ASPECT_LENGTH = 5
        self.INITIALIZER = initializers.RandomUniform(minval=-0.002, maxval=0.002)
        self.REGULARIZER = regularizers.l2(self.L2_REG_VAL)


        self.LSTM_PARAMS = {
            'units': self.N_HIDDEN,
            'activation': 'tanh',
            'recurrent_activation': 'hard_sigmoid',
            'dropout': 0,
            'recurrent_dropout': 0
            }

        self.DENSE_PARAMS = {
            'kernel_initializer': self.INITIALIZER,
            'bias_initializer': self.INITIALIZER,
            'kernel_regularizer': self.REGULARIZER,
            'bias_regularizer': self.REGULARIZER,
            'dtype':'float32'

            }

        self.texts_raw_indices, self.texts_raw_without_aspects_indices, self.texts_left_indices, self.texts_left_with_aspects_indices, \
        self.aspects_indices, self.texts_right_indices, self.texts_right_with_aspects_indices, self.dataset_index,\
        self.polarities_matrix,self.polarities,\
        self.embedding_matrix, \
        self.tokenizer = \
            read_dataset(types=self.DATASET,
                         mode='train',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)



        print('Build model...')
        inputs_l = Input(shape=(self.MAX_SENTENCE_LENGTH,),dtype='int64')
        inputs_r = Input(shape=(self.MAX_SENTENCE_LENGTH,),dtype='int64')
        #input_dataset = Input(shape=(1,),dtype='float32')


        Embedding_Layer = Embedding(input_dim=len(self.embedding_matrix) ,
                                    output_dim=self.EMBEDDING_DIM,
                                    input_length=self.MAX_SENTENCE_LENGTH,
                                    mask_zero=True,
                                    weights=[self.embedding_matrix],
                                    trainable=False)
        x_l = Embedding_Layer(inputs_l)
        x_r = Embedding_Layer(inputs_r)

        x_l = LSTM(name='sentence_left',**self.LSTM_PARAMS)(x_l)
        x_r = LSTM(go_backwards=True, name='sentence_right',**self.LSTM_PARAMS)(x_r)

        x= Concatenate(name='last_shared')([x_l,x_r])

        #x= Dense(self.N_HIDDEN,name='t1_dense_10',**self.DENSE_PARAMS)(x)
        output = Dense(self.N_CLASS,name='t1_dense_3',**self.DENSE_PARAMS)(x)

        model = Model(inputs=[inputs_l, inputs_r], outputs=output)
        model.summary()


        weightsToLoad_3 = np.load('./weights/mtl_absa_saved_model_3_2019_09_27_17_00.pkl.npy',allow_pickle=True)
        weightsToLoad_4 = np.load('./weights/mtl_absa_saved_model_4_2019_09_27_17_00.pkl.npy',allow_pickle=True)
        # weightsToLoad_6 = np.load('inits.pkl.npy', allow_pickle=True)
        model.layers[3].set_weights(weightsToLoad_3)
        model.layers[4].set_weights(weightsToLoad_4)
        # model.layers[6].set_weights(weightsToLoad_6)

        if os.path.exists(self.MODEL_TO_LOAD):
            print('loading saved model...')
            model.load_weights(self.MODEL_TO_LOAD)

        self.model = model

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizers.Adam(lr=self.LEARNING_RATE), metrics=['acc', f1])

    def train(self,):
        tbCallBack = TensorBoard(log_dir='./mtltd_lstm_logs', histogram_freq=4, write_graph=True, write_images=True)

        def modelSave(epoch, logs):

            if (epoch + 1) % 4 == 0:
                currentDT = datetime.datetime.now()
                model_name = './models/single_task_model_'+currentDT.strftime("%Y_%m_%d_%H_%m")+'.h5'
                self.model.save(model_name)
                print("Model is saved")

        msCallBack = LambdaCallback(on_epoch_end=modelSave)

        # texts_raw_indices, texts_raw_without_aspects_indices, texts_left_indices, texts_left_with_aspects_indices, \
        # aspects_indices, texts_right_indices, texts_right_with_aspects_indices, dataset_index, \
        # polarities_matrix = \
        #     read_dataset(types=self.DATASET,
        #                  mode='validate',
        #                  embedding_dim=self.EMBEDDING_DIM,
        #                  max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)



        self.model.fit([self.texts_left_indices, self.texts_right_indices],
                       self.polarities,
                        validation_split=0.1,
                       shuffle=True,
                       epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                       verbose=2,callbacks=[tbCallBack,msCallBack])


    def test(self):
        tw_texts_raw_indices, tw_texts_raw_without_aspects_indices, tw_texts_left_indices, tw_texts_left_with_aspects_indices, \
        tw_aspects_indices, tw_texts_right_indices, tw_texts_right_with_aspects_indices, tw_dataset_index, \
        tw_polarities_matrix,tw_polarities= \
            read_dataset(types=self.DATASET,
                         mode='test',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

        self.model.evaluate([tw_texts_left_indices,tw_texts_right_indices],\
                            tw_polarities,steps=1)





if __name__ == '__main__':


    model = MTLABSA()
    model.train()
    model.test()