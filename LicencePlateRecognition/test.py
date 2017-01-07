from PlateRecognition import PlateRecognition
import cv2

img = cv2.imread("Tests/images/plates/t9.jpg")
PR = PlateRecognition()

plate_number = PR.recognize_digits(img)
print(plate_number)

