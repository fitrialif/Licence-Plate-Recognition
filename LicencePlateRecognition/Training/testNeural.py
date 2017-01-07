import numpy as np
import cv2
import pickle
import glob
import random


def testNeural(test_mode = 0, nTests = 10): # 0: numbers, 1: letters

    if(test_mode == 0):
        clf = pickle.load(open("Trained Neural Network Files/neural-numbers.pkl", 'rb'))
        strg = "Generated Symbols/numbersTest/*"
        nSymbols = 10
        pass
    else:
        clf = pickle.load(open("Trained Neural Network Files/neural-letters.pkl", 'rb'))
        strg = "Generated Symbols/lettersTest/*"
        nSymbols = 26
        pass


    pass


    outputNodes =  nSymbols
    contRight = 0

    if(nTests == -1):
        nTests = len(glob.glob(strg))

    amostraDeImagens = random.sample(glob.glob(strg), nTests)
    for j, im in enumerate(amostraDeImagens):
        # print(im)
        symbol = im[ im.index("-") - 1 ]
        # print(symbol)

        x = cv2.imread(im,0)
        x = cv2.resize(x, (28, 28), interpolation = cv2.INTER_CUBIC)
        # cv2.imshow("img",x)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        x = (np.asfarray(x)/255.0)*0.99 #+ 0.01
        x = np.array(x).flatten()
        x = x.reshape(1, -1)

        # print (symbol)
        # print (clf.predict(x)[0] )
        # print("/")
        #break

        if( str(symbol) == str(clf.predict(x)[0])):
            contRight = contRight + 1


    print(contRight)
    print(nTests)
    perc = float(contRight)/float(nTests)
    print(perc)


########################################################################################################################
if __name__ == "__main__":
    test_mode = 1
    nTests = -1
    testNeural(test_mode,nTests)
    pass
