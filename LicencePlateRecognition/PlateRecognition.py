import functions

class PlateRecognition(object):

    def __init__(self):
        pass


    def find_plate(self,img):
        """
         This method looks for a licence plate in a image.
         It returns the cropped licence plate if it finds it.
         """
        pass


    def recognize_digits(self,img_plate):
        """
        This method segments the digits of the licence plate
        image given by the user and identifies each digit
        using a neural network.
        It returns a string containing the digits of the
        licence plate.
        """
        digits = self.segment_digits(img_plate)
        licence_plate = functions.identify(digits)

        return licence_plate


    def segment_digits(self, img_plate):
        """
        """
        digits = functions.seg_img(img_plate)
        return digits


    def monitor_plates(self,img):
        """
        This method receives a image, looks for a licence
        plate in it, recognize the plate digits (if existent)
        and checks in a server for found licence plate
        information
        """
        plate = self.__sad_plate(img)
        if(plate=="-1"):
            info = "-1"
        else:
            info = self.__check_server(plate)

        return info


    def __sad_plate(self, img):
        """
        This method looks for a plate in the given image.
        If it finds it, it will recognize the digits
        of the found plate.
        It returns the string containing the digits of the
        licence plate.
        """

        plate = self.find_plate(img)

        if( plate.size == 0  ):
            licence_plate_number = "-1"
        else:
            licence_plate_number = self.recognize_digits(plate)

        return licence_plate_number
        pass


    def __check_server(self, plate_number):
        """
        This method checks into a government website and returns
        the information of the licence plate number given.
        """
        return functions.check_plate(plate_number)




