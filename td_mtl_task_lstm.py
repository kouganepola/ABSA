__author__ = 'Koumudi'
import os
from utils import read_dataset
from custom_metrics import f1,multitask_loss,multitask_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, LambdaCallback
from tensorflow.python.keras import initializers, regularizers, optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Activation, LSTM, Embedding, Concatenate, Lambda, Bidirectional
import datetime
import tensorflow as tf
from attention_layer import Attention
from tensorflow.python.keras import backend as K
import numpy as np

class MTLABSA:
    def __init__(self, embedding_dim=100, batch_size=128, n_hidden=100, learning_rate=0.01, n_class=3, max_sentence_len=40, l2_reg_val=0.0006):
        ############################
        self.DATASET = ['restaurant','laptop']
        self.TASK_INDICES=[1002,1003,1005]  ##1001-twitter, 1002-restaurant, 1003-laptop, 1004-others, 1005-general
        self.LOSS_WEIGHTS = {1002:0.7,1003:0.4,1005:0.5}
        self.MODEL_TO_LOAD = './models/mtl_absa_saved_model'
        ###########################
        self.SCORE_FUNCTION = 'mlp'
        self.EMBEDDING_DIM = embedding_dim
        self.BATCH_SIZE = batch_size
        self.N_HIDDEN = n_hidden
        self.LEARNING_RATE = learning_rate
        self.N_CLASS = n_class
        self.MAX_SENTENCE_LENGTH = max_sentence_len
        self.EPOCHS = 4
        self.L2_REG_VAL = l2_reg_val
        self.MAX_ASPECT_LENGTH = 5
        self.INITIALIZER = initializers.RandomUniform(minval=-0.003, maxval=0.003)
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
        input_dataset = Input(shape=(1,),dtype='float32')
        inputs_aspect = Input(shape=(self.MAX_ASPECT_LENGTH,), dtype='int64', name='inputs_aspect')
        nonzero_count = Lambda(lambda xin: tf.count_nonzero(xin, dtype='float32'))(inputs_aspect)


        Embedding_Layer = Embedding(input_dim=len(self.embedding_matrix) ,
                                    output_dim=self.EMBEDDING_DIM,
                                    input_length=self.MAX_SENTENCE_LENGTH,
                                    mask_zero=True,
                                    weights=[self.embedding_matrix],
                                    trainable=False)

        aspect = Embedding(input_dim=len(self.embedding_matrix),
                           output_dim=self.EMBEDDING_DIM,
                           input_length=self.MAX_ASPECT_LENGTH,
                           mask_zero=True,
                           weights=[self.embedding_matrix],
                           trainable=False, name='aspect_embedding')(inputs_aspect)

        x_l = Embedding_Layer(inputs_l)
        x_r = Embedding_Layer(inputs_r)

        x_aspect = Bidirectional(LSTM(name='aspect', return_sequences=True, **self.LSTM_PARAMS), merge_mode='sum')(
            aspect)
        x_aspect = Lambda(lambda xin: K.sum(xin[0], axis=1) / xin[1], name='aspect_mean')([x_aspect, nonzero_count])

        x_l = LSTM(name='sentence_left', return_sequences=True,**self.LSTM_PARAMS)(x_l)
        x_r = LSTM(go_backwards=True, return_sequences=True,name='sentence_right',**self.LSTM_PARAMS)(x_r)

        x= Concatenate(name='last_shared',axis=1)([x_l,x_r])

        #twitter task layers
        tw_attention = Attention(score_function=self.SCORE_FUNCTION,
                                     initializer=self.INITIALIZER, regularizer=self.REGULARIZER,
                                     name='t1_attention')((x, x_aspect))
        tw_x = Lambda(lambda x: K.squeeze(x, 1))(tw_attention)
        tw_x= Dense(self.N_HIDDEN,name='t1_dense_10',**self.DENSE_PARAMS)(tw_x)
        twitter_x = Dense(self.N_CLASS,name='t1_dense_3',**self.DENSE_PARAMS)(tw_x)
        twitter_x = Concatenate(name= "twitter_output")([twitter_x,input_dataset])

        #rest task layers
        rest_attention = Attention(score_function=self.SCORE_FUNCTION,
                                 initializer=self.INITIALIZER, regularizer=self.REGULARIZER,
                                 name='t2_attention')((x, x_aspect))
        rest_x = Lambda(lambda x: K.squeeze(x, 1))(rest_attention)
        rest_x= Dense(self.N_HIDDEN,name='t2_dense_10',**self.DENSE_PARAMS)(rest_x)
        rest_x = Dense(self.N_CLASS,name='t2_dense_3',**self.DENSE_PARAMS)(rest_x)
        rest_x = Concatenate(name="rest_output")([rest_x,input_dataset])

        #general task layers
        general_attention = Attention(score_function=self.SCORE_FUNCTION,
                                 initializer=self.INITIALIZER, regularizer=self.REGULARIZER,
                                 name='t3_attention')((x, x_aspect))
        general_x = Lambda(lambda x: K.squeeze(x, 1))(general_attention)
        general_x= Dense(self.N_HIDDEN,name='t3_dense_10',**self.DENSE_PARAMS)(general_x)
        general_x = Dense(self.N_CLASS,name='t3_dense_3',**self.DENSE_PARAMS)(general_x)
        general_x = Concatenate(name="general_output")([general_x,input_dataset])

        model = Model(inputs=[inputs_l, inputs_r,input_dataset,inputs_aspect], outputs=[twitter_x, rest_x, general_x])
        model.summary()

        dictionary = {v.name: i for i, v in enumerate(model.layers)}
        print(dictionary)

        if os.path.exists(self.MODEL_TO_LOAD):
            print('loading saved model...')
            model.load_weights(self.MODEL_TO_LOAD)

        self.model = model

        self.model.compile(loss={'twitter_output': multitask_loss(self.LOSS_WEIGHTS, self.TASK_INDICES[0]),
                                 'rest_output': multitask_loss(self.LOSS_WEIGHTS, self.TASK_INDICES[1]),
                                 'general_output': multitask_loss(self.LOSS_WEIGHTS, self.TASK_INDICES[2])},
                           optimizer=optimizers.Adam(lr=self.LEARNING_RATE), metrics=[multitask_accuracy, f1])

        # currentDT = datetime.datetime.now()
        # weightsAndBiases_left = self.model.layers[5].get_weights()
        # weightsAndBiases_right = self.model.layers[6].get_weights()
        # weightsAndBiases_bidirectional = self.model.layers[7].get_weights()
        # weightsAndBiases_attention = self.model.layers[12].get_weights()
        #
        # weights_d10 = self.model.layers[18].get_weights()[0]
        # biases_d10 = self.model.layers[18].get_weights()[1]
        # weightsAndBiases_out = self.model.layers[22].get_weights()
        #
        # layer5 = './weights/mtl_absa_saved_model_5_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        # layer6 = './weights/mtl_absa_saved_model_6_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        # layer7 = './weights/mtl_absa_saved_model_7_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        # layer12 = './weights/mtl_absa_saved_model_13_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        # layer18_1 = './weights/mtl_absa_saved_model_biases_19_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        # layer18 = './weights/mtl_absa_saved_model_weights_19_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        # layer22= './weights/mtl_absa_saved_model_23_' + currentDT.strftime("%Y_%m_%d_%H_%M") + '.pkl'
        #
        # np.save(layer5, arr=weightsAndBiases_left)
        # np.save(layer6, arr=weightsAndBiases_right)
        # np.save(layer7, arr=weightsAndBiases_bidirectional)
        # np.save(layer12, arr=weightsAndBiases_attention)
        # np.save(layer18_1, arr=biases_d10)
        # np.save(layer18, arr=weights_d10)
        # np.save(layer22, arr=weightsAndBiases_out)

    def train(self,X,y):
        tbCallBack = TensorBoard(log_dir='./mtltd_lstm_logs', histogram_freq=4, write_graph=True, write_images=True)

        def modelSave(epoch, logs):

            if (epoch + 1) % 4 == 0:
                currentDT = datetime.datetime.now()
                model_name = './models/mtl_absa_saved_model_'+currentDT.strftime("%Y_%m_%d_%H_%M")+'.h5'
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



        self.model.fit(X,
                       {'twitter_output':y,'rest_output':y,'general_output':y},
                        validation_split=0.1,
                       shuffle=True,
                       epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                       callbacks=[msCallBack],verbose=0)

    def test_unseen(self):
        test_datasets = ['twitter', 'hotel']
        laptop_texts_raw_indices, laptop_texts_raw_without_aspects_indices, laptop_texts_left_indices, laptop_texts_left_with_aspects_indices, \
        laptop_aspects_indices, laptop_texts_right_indices, laptop_texts_right_with_aspects_indices, laptop_dataset_index, \
        laptop_polarities_matrix, laptop_polarities = \
            read_dataset(types=[test_datasets[0]],
                         mode='test',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

        hotel_texts_raw_indices, hotel_texts_raw_without_aspects_indices, hotel_texts_left_indices, hotel_texts_left_with_aspects_indices, \
        hotel_aspects_indices, hotel_texts_right_indices, hotel_texts_right_with_aspects_indices, hotel_dataset_index, \
        hotel_polarities_matrix, hotel_polarities = \
            read_dataset(types=[test_datasets[1]],
                         mode='test',
                         embedding_dim=self.EMBEDDING_DIM,
                         max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)

        polarities = [laptop_polarities, hotel_polarities]

        for j in range(len(polarities)):
            print("Dataset id : {}".format(test_datasets[j]))
            test_unique_elements, test_counts_elements = np.unique(polarities[j], return_counts=True)

            for i in range(len(test_unique_elements)):
                print("Sentiment : {0} , Count : {1}".format(test_unique_elements[i], test_counts_elements[i]))

        self.model.evaluate([laptop_texts_left_indices, laptop_texts_right_indices, laptop_dataset_index,laptop_aspects_indices], \
                            [laptop_polarities_matrix, laptop_polarities_matrix, laptop_polarities_matrix], steps=1)
        self.model.evaluate([hotel_texts_left_indices, hotel_texts_right_indices, hotel_dataset_index,hotel_aspects_indices], \
                            [hotel_polarities_matrix, hotel_polarities_matrix, hotel_polarities_matrix], steps=1)

    def test(self, X, y):
        # tw_texts_raw_indices, tw_texts_raw_without_aspects_indices, tw_texts_left_indices, tw_texts_left_with_aspects_indices, \
        # tw_aspects_indices, tw_texts_right_indices, tw_texts_right_with_aspects_indices, tw_dataset_index, \
        # tw_polarities_matrix,tw_polarities= \
        #     read_dataset(types=['twitter'],
        #                  mode='test',
        #                  embedding_dim=self.EMBEDDING_DIM,
        #                  max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)
        #
        # self.model.evaluate([tw_texts_left_indices,tw_texts_right_indices,tw_dataset_index],\
        #                     [tw_polarities_matrix,tw_polarities_matrix,tw_polarities_matrix],steps=1)
        #
        # rest_texts_raw_indices, rest_texts_raw_without_aspects_indices, rest_texts_left_indices, rest_texts_left_with_aspects_indices, \
        # rest_aspects_indices, rest_texts_right_indices, rest_texts_right_with_aspects_indices, rest_dataset_index, \
        # rest_polarities_matrix,rest_polarities= \
        #     read_dataset(types=['restaurant'],
        #                  mode='test',
        #                  embedding_dim=self.EMBEDDING_DIM,
        #                  max_seq_len=self.MAX_SENTENCE_LENGTH, max_aspect_len=self.MAX_ASPECT_LENGTH)
        #
        # self.model.evaluate([rest_texts_left_indices,rest_texts_right_indices,rest_dataset_index],\
        #                     [rest_polarities_matrix,rest_polarities_matrix,rest_polarities_matrix],steps=1)
        #

        self.model.evaluate(X, [y, y, y], steps=1)

if __name__ == '__main__':

    model = MTLABSA()
    model.train()
    model.test()