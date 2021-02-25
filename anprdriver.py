from numpy.lib.function_base import average
from anprclass import CannyANPR, EdgelessANPR, SobelANPR
from imutils import paths
import argparse
from statistics import median
import imutils
import time
import sys
import cv2

def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
	help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not to show additional visualizations")
ap.add_argument("-a", "--algorithm", type=int, default=1,
    help="choose an edge detection method (1 - Sobel edge detection, 2 - Canny edge detection, 3 - Edge-less approach")
ap.add_argument("-s", "--save", type=int, default = -1,
    help="whether to save or not the results in a folder")
ap.add_argument("-m", "--morphology", type=int, default = 1,
    help="whether to use black hat or top hat")
args = vars(ap.parse_args())

anpr = None
iteration = 0
positive = 0
list_time = []
list_positive_time = []
avg_time = 0.0
avg_positive_time = 0.0

algo = args["algorithm"]
if algo == 1:
    anpr = SobelANPR(algo, morph=args["morphology"], debug=args["debug"] > 0, save=args["save"] > 0)
elif algo == 2:
    anpr = CannyANPR(algo, morph=args["morphology"], debug=args["debug"] > 0, save=args["save"] > 0)
elif algo == 3:
    anpr = EdgelessANPR(algo, morph=args["morphology"], debug=args["debug"] > 0, save=args["save"] > 0)
else:
    print('Invalid algorithm choice')
    sys.exit()

imagePaths = sorted(list(paths.list_images(args["input"])))
for imagePath in imagePaths:
    iteration += 1
    start_time = time.time()
    originimage = cv2.imread(imagePath)
    # cv2.imshow("Original", originimage)
    image = imutils.resize(originimage, width=400, height=400)

    image = cv2.bilateralFilter(image, 3, 105, 105)
    cv2.imshow("Bilateral Filter", image)

    (lpText, lpCnt) = anpr.find_and_ocr(iteration, image, psm=args["psm"], clearBorder=args["clear_border"] > 0)

    if lpText is not None and lpCnt is not None:
        positive += 1
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        (x, y, w, h) = cv2.boundingRect(lpCnt)
        cv2.putText(image, cleanup_text(lpText), (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # print("[INFO] {}".format(lpText))
        anpr.debug_imshow("Output ANPR", image, waitKey=True)

    end_time = time.time()
    process_time = end_time - start_time
    process_time = round(process_time, 3)

    list_time.append(process_time)
    print("Processing time = {0:.3f}".format(process_time))

    anpr.save_result("Final{}.jpg".format(iteration), image)
    anpr.save_time(iteration, process_time)

med = median(list_time)
print("med = {}".format(med))
for times in range(len(list_time)):
    if list_time[times] >= med:
        list_positive_time.append(list_time[times])

avg_time = round(average(list_time), 3)
anpr.save_time(0, avg_time, 1)

avg_positive_time = round(average(list_positive_time), 3)
anpr.save_time(0, avg_positive_time, 2)