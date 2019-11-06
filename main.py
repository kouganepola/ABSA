import numpy as np
from utils import load_data_kfold,read_dataset
from td_mtl_model import MTLABSA

dataset = ['restaurant','laptop']
k = 7
folds, X_train, y_train, polarities= load_data_kfold(k,dataset)

for j, (train_idx, test_idx) in enumerate(folds):
    print('\nFold ', j)

    X_train_cv = [i[train_idx] for i in X_train[:3]]
    y_train_cv = y_train[train_idx]
    polarities_train_cv = polarities[train_idx]

    X_valid_cv = [i[test_idx] for i in X_train[:3]]
    y_valid_cv = y_train[test_idx]
    polarities_valid_cv = polarities[test_idx]

    dataset_indices = np.unique(X_train[2])
    test_indices = []

    for idx in dataset_indices:
        unique_elements, counts_elements = np.unique(polarities_train_cv[np.where(X_train_cv[2] == idx)[0]], return_counts=True)

        test_data = np.where(X_valid_cv[2]==idx)[0]

        test_unique_elements, test_counts_elements = np.unique(polarities_valid_cv[test_data], return_counts=True)

        print("Dataset id : {}".format(idx))

        for i in range (len(unique_elements)):
            print ("Train : Sentiment : {0} , Count : {1}".format(unique_elements[i],counts_elements[i]))
            print("Test : Sentiment : {0} , Count : {1}".format(test_unique_elements[i], test_counts_elements[i]))

        test_indices.append(test_data)

    test_X_1 = [i[test_indices[0]] for i in X_valid_cv]
    test_y_1 = y_valid_cv[test_indices[0]]

    test_X_2 = [i[test_indices[1]] for i in X_valid_cv]
    test_y_2 = y_valid_cv[test_indices[1]]

    model= None
    model = MTLABSA()
    model.train(X_train_cv,y_train_cv)
    print(dataset_indices[0])
    model.test(test_X_1, test_y_1)
    print(dataset_indices[1])
    model.test(test_X_2, test_y_2)
    model.test_unseen()


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