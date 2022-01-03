from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2
import os

#Path to Tesseract executable. Usually located at C:\Program Files\Tesseract-OCR\tesseract.exe for Windows.
#This is only required if Tesseract executable is not defined in PATH, otherwise keep the line below commented.
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\afeef\\Documents\\Development\\TesseractOCR\\tesseract.exe' #CHANGE THIS BEFORE COMMIT

class SobelANPR:
    def __init__(self, algo, input_dir, morph, minAR=2.5, maxAR=5, debug=False, save=False):  
        self.minAR = minAR  #These two lines are to define the aspect ratio (AR) of the region of interest (ROI). Try experimenting with these values!
        self.maxAR = maxAR  #If the region of interest is within the boundary (minAR and maxAR) set here, it will be considered as a license plate
        self.debug = debug
        self.save = save
        self.algo = algo
        self.input_dir = input_dir
        self.morph = morph

    def debug_imshow(self, title, image, waitKey=False):    #If debug argument (-d) is set to 1, the script will show the whole image processing pipeline
        if self.debug:                                      #and wait for user input before continuing to the next step.
            cv2.imshow(title, image)
            if waitKey:
                cv2.waitKey(0)

    def save_result(self, name, image):                     #This function is used to save the final image which contains the ROI
        try:                                                #in a folder and also the image of the extracted ROI.
            result_dir = ['result_sobel', 'result_canny', 'result_edgeless']    
            image_dir = r'.' + '\{}'.format(result_dir[self.algo-1] + '_{}'.format(self.input_dir))
            print(image_dir)
            os.chdir(image_dir)
        except:
            print("No folder directory found. Creating...")
            os.mkdir(image_dir)
            os.chdir(image_dir)
            
        if self.save and self.debug != 1:
            cv2.imwrite(name, image)
            
        os.chdir('..')

    def morphology_operation(self, gray, rectKern):
        if self.morph=='bh':
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
            self.debug_imshow("Blackhat", blackhat, waitKey=True)

            squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
            self.debug_imshow("Closing operation", light, waitKey=True)
            light = cv2.threshold(light, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            self.debug_imshow("Light Regions", light, waitKey=True)

            return [blackhat, light]

        elif self.morph=='th':
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKern)
            self.debug_imshow("Tophat", tophat, waitKey=True)

            squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dark = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
            self.debug_imshow("Closing operation", dark, waitKey=True)
            dark = cv2.threshold(dark, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            self.debug_imshow("Dark Regions", dark, waitKey=True)

            return [tophat, dark]

    def locate_license_plate_candidates(self, gray, image, keep=5):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morphology = self.morphology_operation(gray, rectKern)
        morph = morphology[0]
        luminance = morphology[1]

        gradX = cv2.Sobel(morph, ddepth=cv2.CV_32F,
            dx=1, dy=0, ksize=3)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX, waitKey=True)

        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        self.debug_imshow("Gaussian", gradX, waitKey=True)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh, waitKey=True)

        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)
        self.debug_imshow("Grad Erode/Dilate", thresh, waitKey=True)

        thresh = cv2.bitwise_and(thresh, thresh, mask=luminance)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        oriCopy = image.copy()
        for c in cnts:
            cv2.drawContours(oriCopy, [c], -1, 255, 2)
            self.debug_imshow("Contours", oriCopy)

        self.debug_imshow("Masked", thresh, waitKey=True)

        return cnts

    def locate_license_plate(self, iteration, gray, candidates, clearBorder=False):
        lpCnt = None
        roi = None

        candidates = sorted(candidates, key=cv2.contourArea)

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if ar >= self.minAR and ar <= self.maxAR:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                if clearBorder:
                    roi = clear_border(roi)

                self.debug_imshow("License Plate", licensePlate, waitKey=True)
                self.debug_imshow("ROI", roi, waitKey=True)

                self.save_result('roi{}.png'.format(iteration), licensePlate)
                break

        return (roi, lpCnt)

    def build_tesseract_options(self, psm=7):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " --psm {}".format(psm)
        return options

    def find_and_ocr(self, iteration, image, psm=7, clearBorder=False):
        lpText = None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_imshow("Grayscale", gray, waitKey=True)
        candidates = self.locate_license_plate_candidates(gray, image, 5)
        (lp, lpCnt) = self.locate_license_plate(iteration, gray, candidates, clearBorder=clearBorder)

        if lp is not None:
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp, waitKey=True)

        return (lpText, lpCnt)

class CannyANPR(SobelANPR):
    def locate_license_plate_candidates(self, gray, image, keep = 5):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morphology = self.morphology_operation(gray, rectKern)
        morph = morphology[0]
        luminance = morphology[1]

        canny = cv2.Canny(morph, 200, 230) # Originally 400,450; try to experiment with these values.
        self.debug_imshow("Canny", canny, waitKey=True)

        gaussian = cv2.GaussianBlur(canny, (5,5), 0)
        gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gaussian, 0, 255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh, waitKey=True)
        
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)
        self.debug_imshow("Eroded & Dilated", thresh, waitKey=True)

        thresh = cv2.bitwise_and(thresh, thresh, mask=luminance)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        oriCopy = image.copy()
        for c in cnts:
            cv2.drawContours(oriCopy, [c], -1, 255, 2)
            self.debug_imshow("Contours", oriCopy)

        self.debug_imshow("Masked", thresh, waitKey=True)

        return cnts

class EdgelessANPR(SobelANPR):
    def locate_license_plate_candidates(self, gray, image, keep = 5):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morphology = self.morphology_operation(gray, rectKern)
        morph = morphology[0]
        luminance = morphology[1]

        gaussian = cv2.GaussianBlur(morph, (5,5), 0)
        thresh = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(thresh, 0, 255, 
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Thresholded", thresh, waitKey=True)

        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=3)
        self.debug_imshow("Eroded & Dilated", thresh, waitKey=True)

        thresh = cv2.bitwise_and(thresh, thresh, mask=luminance)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        oriCopy = image.copy()
        for c in cnts:
            cv2.drawContours(oriCopy, [c], -1, 255, 2)
            self.debug_imshow("Contours", oriCopy)

        self.debug_imshow("Masked", thresh, waitKey=True)

        return cnts