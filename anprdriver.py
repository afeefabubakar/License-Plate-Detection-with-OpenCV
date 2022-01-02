from anprclass import CannyANPR, EdgelessANPR, SobelANPR
from imutils import paths
import argparse
import imutils
import sys
import cv2

def cleanup_text(text):
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="Path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="Whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
	help="Default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="Whether or not to show additional visualizations")
ap.add_argument("-a", "--algorithm", type=int, default=1,
    help="Choose an edge detection method (1 - Sobel edge detection, 2 - Canny edge detection, 3 - Edge-less approach")
ap.add_argument("-s", "--save", type=int, default = -1,
    help="Whether to save or not the results in a folder")
ap.add_argument("-m", "--morphology", type=str, default = 'bh',
    help="Whether to use black hat (-m bh) or top hat (-m th). Black hat is better at detecting black-on-white license plate while top hat is the opposite.")
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
    originimage = cv2.imread(imagePath)
    image = imutils.resize(originimage, width=400, height=400)

    image = cv2.bilateralFilter(image, 3, 105, 105)
    anpr.debug_imshow("Bilateral Filter", image, waitKey=True)

    (lpText, lpCnt) = anpr.find_and_ocr(iteration, image, psm=args["psm"], clearBorder=args["clear_border"] > 0)

    if lpText is not None and lpCnt is not None:
        positive += 1
        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        (x, y, w, h) = cv2.boundingRect(lpCnt)
        cv2.putText(image, cleanup_text(lpText), (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print("[INFO] Registration number: {}".format(lpText))
        anpr.debug_imshow("Output ANPR", image, waitKey=True)

    anpr.save_result("Final{}.jpg".format(iteration), image)
    cv2.destroyAllWindows()