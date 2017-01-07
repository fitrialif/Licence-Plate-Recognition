import numpy as np
import cv2
import imutils
import random
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


rotation = 2
blib = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def show_image(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_symbols(train_code = 0, show = False, nSamplesPerSymbol = 100):
    if(train_code == 0):    # numbers
        dataTrainingPath = "Generated Symbols/numbers/"
        nSymbols = 10
        contSymbols = 0
        pass
    elif(train_code == 1):  # letters
        dataTrainingPath = "Generated Symbols/letters/"
        nSymbols = 36
        contSymbols = 10
        pass
    elif(train_code == 2):  # numbers_test
        dataTrainingPath = "Generated Symbols/numbersTest/"
        nSymbols = 10
        contSymbols = 0
        pass
    elif(train_code == 3):  # letters_test
        dataTrainingPath = "Generated Symbols/lettersTest/"
        nSymbols = 36
        contSymbols = 10
        pass
    elif(train_code == 4):  # test
        dataTrainingPath = "Generated Symbols/test/"
        nSymbols = 10
        contSymbols = 0
        pass

    def cropSegment(img):
        imgGrey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGrey.copy(), (5, 5), 0)
        imgEdge = cv2.Canny(imgBlurred.copy(), 0, 200)
        _, thresholdedEdges = cv2.threshold(imgEdge.copy(), 0, 255, cv2.THRESH_BINARY)

        countours, _ = cv2.findContours(thresholdedEdges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        boxes = []

        for cnt in countours:
            x, y, w, h = cv2.boundingRect(cnt)

            areas.append(w * h)
            boxes.append((x, y, w, h))

        maxAreaIndex = (areas.index(max(areas)))
        (xlim, ylim, wlim, hlim) = boxes[maxAreaIndex]
        imAux = img.copy()[ylim:ylim + hlim, xlim:xlim + wlim, :]
        imm = np.zeros((imAux.shape[0]+3,imAux.shape[1]+3,4))
        imm[0:hlim,0:wlim,:] = imAux
        # print(imm.shape)
        return imAux
        pass

    while (contSymbols < nSymbols):
        name = blib[contSymbols] + "-"
        contSamplesPerSymbol = 0

        while (contSamplesPerSymbol < nSamplesPerSymbol):
            nam = name + str(contSamplesPerSymbol) + ".jpg"
            imagePIL = Image.new("RGBA", (40, 100), (0, 0, 0))
            draw = ImageDraw.Draw(imagePIL)
            font = ImageFont.truetype("MANDATOR.ttf", size=random.randrange(62, 65))

            if (contSymbols == 1):
                textPos = (12, 12)
            else:
                textPos = (-3, 12)

            draw.text(textPos, blib[contSymbols], (255, 255, 255), font=font)
            img = np.array(imagePIL.copy())
            # img = cropSegment(img.copy())
            # if (show): show_image(img)
            img = imutils.rotate(img.copy(), random.uniform(-rotation, rotation))
            # if (show): show_image(img)
            img = cv2.resize(img.copy(), (28, 28), interpolation=cv2.INTER_CUBIC)
            # if (show): show_image(img)
            img = np.uint8(img.copy())
            img = cropSegment(img.copy())
            img = cv2.resize(img.copy(), (28, 28), interpolation=cv2.INTER_CUBIC)
            # img = skimage.util.random_noise(img.copy(), mode='s&p', seed=None, clip=True, amount=0.05)
            ##
            ###################################
            if(show): show_image(img)


            cv2.imwrite((dataTrainingPath + nam), img)
            contSamplesPerSymbol = contSamplesPerSymbol + 1

        contSymbols = contSymbols + 1

########################################################################################################################
if __name__ == "__main__":
    train_code = 1  #  0 = numbers,
                    #  1 = letters,
                    #  2 = numbers_test,
                    #  3 = letters_test,
                    #  4 = tests
    generate_symbols(train_code, show=False, nSamplesPerSymbol=1000)
    print("Finished.")