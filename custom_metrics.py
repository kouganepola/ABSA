from tensorflow.python.keras import backend as K


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r))

def multitask_loss(loss_weights,task):

        def loss(y_true, y_pred):

            loss_val =  -1*K.sum(K.log(K.softmax(y_pred[:,:-1]))*y_true[:,:-1],axis=-1)

            return K.mean(K.switch(K.equal(task,1005),loss_weights[task]*loss_val,K.switch(K.equal(y_true[:,-1],task), loss_val,loss_weights[task]*loss_val)))


        return loss

def multitask_accuracy(y_true,y_pred):
    """Multi-Task accuracy metric.

    Only computes a batch-wise average of accuracy.

    Computes the accuracy, a metric for multi-label classification of
    how many items are selected correctly.
    """
    return K.mean(K.equal(K.argmax(y_true[:,:-1], axis=-1),K.argmax(y_pred[:,:-1], axis=-1)))