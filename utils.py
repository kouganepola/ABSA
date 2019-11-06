import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import string
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def load_word_vec( embedding_dim=100):

    fname = './glove.6B.'+str(embedding_dim)+'d.txt' \
        if embedding_dim != 300 else './glove.6B.300d.txt'

    fin = open(fname, encoding='utf8')

    w2v = []
    word_dict = ""
    w2v.append([0.] * embedding_dim)
    cnt = 0
    # stop = stopwords.words('english')
    for line in fin:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:

            print ('a bad word embedding: {}'.format(line[0]))
            continue
        #if line[0] not in stop:
        w2v.append([float(v) for v in line[1:]])
        word_dict+=line[0]+" "

    w2v = np.asarray(w2v, dtype=np.float64)
    return word_dict,w2v


def read_dataset(types=['twitter','restaurant'], mode='train', embedding_dim=100, max_seq_len=40, max_aspect_len=3, polarities_dim=3):
    #print("preparing data...")
    fname = {
        'twitter': {
            'train': './data/twitter/all.raw',
            'test': './data/twitter/test.raw',
            'validate':'./data/twitter/test.raw'

        },
        'restaurant': {
            'train': './data/restaurant_1/all.raw',
            'test': './data/restaurant_1/rest2014test.raw',
            # 'train': './data/restaurant/train.raw',
            # 'test': './data/restaurant/test.raw',
            'validate': './data/restaurant/test.raw'
        },
        'laptop': {
            'train': './data/laptop_1/all.raw',
            'test': './data/laptop_1/test.raw',
            # 'train': './data/laptop/train.raw',
            # 'test': './data/laptop/test_short.raw',
            'validate': './data/laptop/test_short.raw'

        },
        'hotel': {
            'train': './data/laptop_1/train.raw',
            'test': './data/hotel/hotel.raw',
            # 'train': './data/laptop/train.raw',
            # 'test': './data/laptop/test_short.raw',
            'validate': './data/laptop/test_short.raw'

        }
    }

    texts_raw = []
    texts_raw_without_aspects = []
    texts_left = []
    texts_left_with_aspects = []
    texts_right = []
    texts_right_with_aspects = []
    aspects = []
    polarities = []
    dataset_index = []


    word_dict,word_vec = load_word_vec(embedding_dim)

    for j in range(len(types)):
        fin = open(fname[types[j]][mode], 'r', encoding='utf8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        print("number of {0} {1} data: {2}".format(types[j] , mode,len(lines)/3))



        for i in range(0, len(lines), 3):

            text_left, _, text_right = [s.lower().strip().split() for s in lines[i].lower().partition("$t$")]
            if (len(text_left) < max_seq_len and len(text_right) < max_seq_len):
                text_left = textCleaner(text_left)
                text_right = textCleaner(text_right)
                text_left =" ".join(text_left)
                text_right = " ".join(text_right)

                aspect = lines[i+1].lower().strip()
                aspect = " ".join(textCleaner(aspect))

                polarity = lines[i+2].strip()

                text_left = ". "+text_left
                text_right = text_right+" ."
                text_raw = text_left + " " + aspect + " " + text_right

                texts_raw.append(text_raw)
                texts_raw_without_aspects.append(text_left + " " + text_right)
                texts_left.append(text_left)
                texts_left_with_aspects.append(text_left + " " + aspect)
                texts_right.append(text_right)
                texts_right_with_aspects.append(aspect + " " + text_right)
                aspects.append(aspect)
                polarities.append(int(polarity))
                if (types[j]=='twitter'):
                    dataset_index.append([1001])
                elif (types[j]=='restaurant'):
                    dataset_index.append([1002])
                elif (types[j]=='laptop'):
                    dataset_index.append([1003])
                else:
                    dataset_index.append([1004])


    polarities_k = np.array(polarities)
    polarities_k[polarities_k==-1]=2
    polarities_matrix = K.eval(tf.one_hot(indices=polarities_k, depth=polarities_dim+1))
    polarities = K.eval(tf.one_hot(indices=polarities_k, depth=polarities_dim))

    text_words = word_dict.strip().split()
    #print('tokenizing...')
    tokenizer = Tokenizer(filters="\t\n")
    tokenizer.fit_on_texts(text_words)

    texts_raw_indices = tokenizer.texts_to_sequences(texts_raw)
    texts_raw_indices = pad_sequences(texts_raw_indices, maxlen=max_seq_len ,  padding='post')
    texts_raw_without_aspects_indices = tokenizer.texts_to_sequences(texts_raw_without_aspects)
    texts_raw_without_aspects_indices = pad_sequences(texts_raw_without_aspects_indices, maxlen = max_seq_len,  padding='post')
    texts_left_indices = tokenizer.texts_to_sequences(texts_left)
    texts_left_indices = pad_sequences(texts_left_indices, maxlen = max_seq_len,  padding='post')
    texts_left_with_aspects_indices = tokenizer.texts_to_sequences(texts_left_with_aspects)
    texts_left_with_aspects_indices = pad_sequences(texts_left_with_aspects_indices, maxlen = max_seq_len,  padding='post')
    texts_right_indices = tokenizer.texts_to_sequences(texts_right)
    texts_right_indices = pad_sequences(texts_right_indices,maxlen=max_seq_len,truncating='post')
    texts_right_with_aspects_indices = tokenizer.texts_to_sequences(texts_right_with_aspects)
    texts_right_with_aspects_indices = pad_sequences(texts_right_with_aspects_indices,maxlen=max_seq_len,truncating='post')
    aspects_indices = tokenizer.texts_to_sequences(aspects)
    aspects_indices = pad_sequences(aspects_indices,maxlen=max_aspect_len,  padding='post')

    for i in range(len(polarities_matrix)):
        polarities_matrix[i][3] = dataset_index[i][0]

    dataset_index=np.array(dataset_index)
    dataset_index=dataset_index.astype('float64')


    if mode == 'validate' or mode=='test':
        return texts_raw_indices, texts_raw_without_aspects_indices, texts_left_indices, texts_left_with_aspects_indices, \
               aspects_indices, texts_right_indices, texts_right_with_aspects_indices,dataset_index, polarities_matrix,polarities_k

    #print('loading word vectors...')

    embedding_matrix = word_vec

    return texts_raw_indices, texts_raw_without_aspects_indices, texts_left_indices, texts_left_with_aspects_indices, \
           aspects_indices, texts_right_indices, texts_right_with_aspects_indices, \
           dataset_index,polarities_matrix,polarities_k, \
           embedding_matrix, \
           tokenizer

def load_data_kfold(k,dataset,emb_dim = 100,seq_len=40,aspect_len=5):
    texts_raw_indices, texts_raw_without_aspects_indices, texts_left_indices, texts_left_with_aspects_indices, \
    aspects_indices, texts_right_indices, texts_right_with_aspects_indices, dataset_index, \
    polarities_matrix, polarities_k, \
    embedding_matrix, \
    tokenizer = \
        read_dataset(types=dataset,
                     mode='train',
                     embedding_dim=emb_dim,
                     max_seq_len=seq_len, max_aspect_len=aspect_len)

    X_train = [texts_left_indices,texts_right_indices,dataset_index,aspects_indices]
    X = texts_right_indices

    y = polarities_k
    y_train=polarities_matrix

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X, y))

    return folds, X_train, y_train, y

def textCleaner(words):
    return_string = []
    for word in words:
        for punctuation in string.punctuation:
            new_word = ''
            if punctuation in word:
                text = word.split(punctuation)
                for word_part in text:
                    new_word = new_word + word_part + ' ' + punctuation + ' '
                word = new_word[0:-3]
                # double space
                double_space = "  "
                count = word.count(double_space)
                for i in range(count):
                    word = word.replace(double_space," ")
        # it's' or he's (perfect)
        apostrophe_s = "' s"
        if apostrophe_s in word:
            word = word.replace(apostrophe_s,"'s")
        # I'm'
        apostrophe_m = "' m"
        if apostrophe_m in word:
            word = word.replace(apostrophe_m,"'m")
        # we've'
        apostrophe_ve = "' ve"
        if apostrophe_ve in word:
            word = word.replace(apostrophe_ve, "'ve")
        # we're
        apostrophe_re = "' re"
        if apostrophe_re in word:
            word = word.replace(apostrophe_re, "'re")
        # exclamation mark
        exclamation = '! ! ! ! !'
        for i in range(5, 1, -1):
            if exclamation in word:
                # multiple !s: !*5+
                if(i==5):
                    word = word.replace(exclamation, exclamation.replace(' ', '')+' ')
                else:
                    word = word.replace(exclamation, exclamation.replace(' ', ''))
            exclamation = exclamation[0:-2]
        # ... three dots
        three_dots = '. . .'
        if three_dots in word:
            word = word.replace(three_dots, three_dots.replace(' ', ''))
        # aspect sign $t$
        aspect_sign = '$ t $'
        if aspect_sign in word:
            word = word.replace(aspect_sign, aspect_sign.replace(' ', ''))
        for w in word.split():
             return_string.append(w)
    return return_string

if __name__ == '__main__':
    read_dataset()