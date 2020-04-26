import re

import cv2
import numpy as np
import pandas as pd
import pytesseract
import matplotlib.pyplot as plt


TESSERACT_CONFIDENCE = 90

config = r'--oem 1 --psm 3'
pattern1 = re.compile(r'[^\w ]')  # delete non alphanumeric symbols
pattern2 = re.compile(r'\s+')  # delete double spaces


def tesseract_predict(img, confidence_threshold=TESSERACT_CONFIDENCE):
    """
    Recognize text from image by pytesseract

    :param img: numpy array - image to check
    :param confidence_threshold: int threshold for tesseract
    :return txt: string, text on image
    """
    res = pytesseract.image_to_data(img, lang='eng+rus', config=config, output_type='dict')
    words = [word for conf, word in zip(res['conf'], res['text']) if int(conf) >= confidence_threshold]
    # res = pytesseract.image_to_data(img, lang='eng+rus', config=conf, output_type='data.frame')
    # words = res[res['conf']>= confidence_threshold]['text'].values.tolist()
    txt = ' '.join(words)
    txt = re.sub(pattern1, ' ', txt.lower())
    txt = re.sub(pattern2, ' ', txt).strip()

    return txt


def auto_canny(img, sigma=0.33):
    """
    Util for canny edge detection
    """
    # compute the median of the single channel pixel intensities
    med = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edged = cv2.Canny(img, lower, upper)

    return edged


def image_preprocess(img_path, inverse_img=False):
    """
    Read image and make preprocessing

    :param img_path: string, path to image
    :param inverse_img: bool, bitwise inverse
    :return img: numpy array, image
    """
    # zero means load in grayscale
    img = cv2.imread(img_path, 0)

    if inverse_img:
        img = cv2.bitwise_not(img)
        # img = cv2.medianBlur(img, 5)
        # img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.bilateralFilter(img, 50, 50, 50)
        # kernel = np.ones((3,3),np.uint8)
        # img = cv2.dilate(img, kernel, iterations=1)
        # img = cv2.erode(img, kernel, iterations=1)
        # img = auto_canny(img)

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 21)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return img


def show_image_raw(img):
    """
    Show loaded image (numpy array)
    """
    fig, ax = plt.subplots(figsize=(15, 5), ncols=1, nrows=1)
    ax.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.show()


def get_text_from_image(img_path, print_image=False):
    """
    Preprocess image and use tesseract
    :param img_path: str, path to image
    :param print_image: bool, show image or not
    """
    # image read and preprocess
    img = image_preprocess(img_path, False)

    # use tesseract to recognize text
    txt = tesseract_predict(img)

    if len(txt.split(' ')) < 3:
        img = image_preprocess(img_path, inverse_img=True)
        txt = tesseract_predict(img)

    if print_image:
        show_image_raw(img)

    return img_path, txt


def check_length(str_true, str_pred):
    if str_pred == '':
        return 0
    return len(str_pred.split(' ')) / len(str_true.split(' '))
