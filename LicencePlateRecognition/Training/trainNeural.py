import numpy as np
import matplotlib.pyplot
import cv2
import pickle
import glob
import progressbar
import random
from sklearn.neural_network import MLPClassifier
import time

def train_neural(train_code, hiddenNodes = 200):

    if(train_code == 0):
        strg = "Generated Symbols/numbers/*"
        save_train_directory = "Trained Neural Network Files/neural-numbers.pkl"
        nSymbols = 10
        pass
    else:
        strg = "Generated Symbols/letters/*"
        save_train_directory = "Trained Neural Network Files/neural-letters.pkl"
        nSymbols = 26
        pass

    outputNodes = nSymbols

    X = [];
    Y = []

    print("Reading Started.")
    for j, im in enumerate(glob.glob(strg)):
        symbol = im[im.index('-') - 1]
        x = cv2.imread(im, 0)
        x = cv2.resize(x, (28, 28), interpolation=cv2.INTER_CUBIC)
        x = (np.asfarray(x) / 255.0) * 0.99  # + 0.01
        x = np.array(x).flatten()
        X.append(x)
        Y.append(symbol)
    print("Reading Finished.")


    ## NN
    print("Training Started.")
    inputNodes = 28 * 28
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hiddenNodes, hiddenNodes), random_state=1)
    clf.fit(X, Y)

    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                  beta_1=0.9, beta_2=0.999, early_stopping=False,
                  epsilon=1e-08, hidden_layer_sizes=(hiddenNodes, hiddenNodes), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=200, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)

    with open(save_train_directory, 'wb') as f:
        pickle.dump(clf, f)

    print("Training Finished.")
    pass


########################################################################################################################
if __name__ == "__main__":
    train_code = 1  # 0 = numbers, 1 = letters
    train_neural(train_code)