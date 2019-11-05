import numpy as np
from utils import load_data_kfold,read_dataset
from td_mtl_model import MTLABSA

dataset = ['twitter','restaurant']
k = 7
folds, X_train, y_train = load_data_kfold(k,dataset)

for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j)

    X_train_cv = [i[train_idx] for i in X_train[:3]]
    y_train_cv = y_train[train_idx]
    X_valid_cv = [i[val_idx] for i in X_train[:3]]
    y_valid_cv = y_train[val_idx]

    model= None
    model = MTLABSA()
    model.train(X_train_cv,y_train_cv)
    model.test(X_valid_cv, y_valid_cv)

    # name_weights = "final_model_fold" + str(j) + "_weights.h5"
    # callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)
    # generator = gen.flow(X_train_cv, y_train_cv, batch_size=batch_size)
    # model = get_model()
    # model.fit_generator(
    #     generator,
    #     steps_per_epoch=len(X_train_cv) / batch_size,
    #     epochs=15,
    #     shuffle=True,
    #     verbose=1,
    #     validation_data=(X_valid_cv, y_valid_cv),
    #     callbacks=callbacks)

    #print(model.evaluate(X_valid_cv, y_valid_cv))