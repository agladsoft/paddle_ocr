import itertools
import re
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import psycopg2
import sys
from itertools import combinations
from collections import Counter
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    import Image
import pytesseract


def find_score_and_text(data, labels=None):
    boxes = len(data["level"])
    list_i = []
    dict_text = {}
    list_score = []
    text_ocr = ""
    for i in range(boxes):
        (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        # Draw box
        if data["text"][i] and not list_i[-1]:
            text_ocr = str(data["text"][i]) if labels else " ".join(data["text"])
            x_min = x
            y_min = y
            list_score.clear()
            list_score.append(data["conf"][i])
        if data["text"][i] and list_i[-1] and labels:
            text_ocr += " " + str(data["text"][i])
            list_score.append(data["conf"][i])
            x_max = x + w
            y_max = y + h
        elif not data["text"][i]:
            try:
                dict_text[text_ocr] = (x_min, y_min, x_max, y_max, np.mean(list_score), np.std(list_score))
            except:
                pass
        list_i.append(data["text"][i])

    return text_ocr, list_score


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),
                                      key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


def find_row_col(row, x, y):
    for row_new, rows in enumerate(row, 1):
        rows.sort(key=lambda x: x.x1)
        for let in range(len(rows)):
            if x == rows[let].y1:
                for col_new, col in enumerate(rows, 1):
                    if y == col.x1:
                        return row_new, col_new


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class Cell:
    def __init__(self, img, x1, y1, width, height):
        self.img = img
        self.parent = None
        self.childs = []
        self.x1 = x1
        self.y1 = y1
        self.x2 = x1 + width
        self.y2 = y1 + height
        self.text = None
        self.score = None
        self.std = None
        self.row = None
        self.col = None
        self.child_max_row = None
        self.child_max_col = None

    def __gt__(self, other):
        return self.x1 <= other.x1, self.y1 <= other.y1, self.x2 >= other.x2, self.y2 >= other.y2

    def __str__(self):
        coords = self.x1, self.y1, self.x2, self.y2
        return str(coords)

    def __repr__(self):
        return self.__str__()

    def recognize(self):
        if self.childs:
            bitnot2 = bitnot.copy()
            for coordinates in self.childs:
                print(coordinates.x1, coordinates.y1, coordinates.x2, coordinates.y2)
                cv2.rectangle(bitnot2, (coordinates.x1, coordinates.y1), (coordinates.x2,
                                                                          coordinates.y2), (255, 255, 255), -1)
        else:
            bitnot2 = bitnot
        finalimg = bitnot2[self.y1 + 2:self.y2 - 2, self.x1 + 1:self.x2 - 1]
        # plotting = plt.imshow(finalimg, cmap="gray")
        # plt.show()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        resizing = cv2.resize(finalimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dilation = cv2.dilate(resizing, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        erosion = cv2.fastNlMeansDenoising(erosion, None, 20, 7, 21)
        text = pytesseract.image_to_data(erosion, output_type="dict", lang="rus+eng")
        out, list_score = find_score_and_text(text, labels=False)
        inner = out.strip()
        inner = inner.translate({ord(c): " " for c in "!@#$%^&*()[]{};<>?\|`~-=_+"})
        self.text = inner
        self.score = np.mean(list_score) if len(list_score) != 0 else None
        self.std = np.std(list_score) if len(list_score) != 0 else None

    def get_structure_of_row(self):
        if not self.childs:
            return
        list_h = [elem.y2 - elem.y1 for elem in self.childs]
        min_h_of_cell = min(list_h)
        self.childs.sort(key=lambda c: c.y1)
        current_row = 0
        prev_child = self.childs[0]
        current_row_y1 = prev_child.y1
        prev_child.row = current_row
        if len(self.childs) > 1:
            for current_child in self.childs[1:]:
                if current_child.y1 - current_row_y1 > min_h_of_cell * 0.95:
                    current_row += 1
                current_child.row = current_row
                current_row_y1 = current_child.y1
                prev_child = current_child
            prev_child.row = current_row
        self.child_max_row = current_row

    def get_structure_of_col(self):
        if not self.childs:
            return
        list_w = [elem.x2 - elem.x1 for elem in self.childs]
        min_w_of_cell = min(list_w)
        self.childs.sort(key=lambda c: c.x1)
        current_col = 0
        prev_child = self.childs[0]
        current_col_x1 = prev_child.x1
        prev_child.col = current_col
        if len(self.childs) > 1:
            for current_child in self.childs[1:]:
                if current_child.x1 - current_col_x1 > min_w_of_cell * 0.95:
                    current_col += 1
                    current_col_x1 = current_child.x1
                current_child.col = current_col
                prev_child = current_child
            prev_child.col = current_col
        self.child_max_col = current_col

    def to_dataframe(self, i):
        if not self.childs or (self.child_max_col == 1 and self.child_max_row == 1):
            return
        df = pd.DataFrame(columns=range(self.child_max_col), index=range(self.child_max_row))
        for box in self.childs:
            df.loc[box.row, box.col] = box.text
        return df

    def to_json(self):
        predicted_boxes_dict = {
            "type": "text",
            "text": self.text, "row": self.row, "col": self.col, "xmin": self.x1, "ymin": self.y1,
            "xmax": self.x2, "ymax": self.y2, "score": self.score, "std": self.std
        }
        json_list = [predicted_boxes_dict]
        if not self.childs:
            return predicted_boxes_dict
        table = {"type": "table", "cells": [child.to_json() for child in self.childs]}
        json_list.append(table)
        return json_list

    def recognize_and_get_structure_of_table(self):
        self.recognize()
        self.get_structure_of_col()
        self.get_structure_of_row()


file = r"all_files/7902 ADMIRAL STAR от 03.06.2022.pdf-001.jpg"
img = cv2.imread(file, 0)
# thresholding the image to a binary image
thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# inverting the image
img_bin = 255 - img_bin
# Length(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1] // 150
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
# Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bitxor = cv2.bitwise_xor(img, img_vh)
bitnot = cv2.bitwise_not(bitxor)

# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort all the contours by top to bottom.
# sourcery no-metrics
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
# Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
# Get mean of heights
mean = np.mean(heights)

all_contours = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w != img.shape[1] and h != img.shape[0]:
        all_contours.append([x, y, w, h])

box = [Cell(img, *contour) for contour in all_contours]
list_contours_with_combinations = list(combinations(box, 2))

for cell1, cell2 in list_contours_with_combinations:
    if all(cell1 > cell2):
        if cell2.parent is not None:
            cell3 = cell2.parent
            if all(cell3 > cell1):
                cell3.childs.remove(cell2)
                cell2.parent = cell1
                cell1.childs.append(cell2)
        else:
            cell2.parent = cell1
            cell1.childs.append(cell2)

for elem_of_box in box:
    elem_of_box.recognize_and_get_structure_of_table()

for i, elem_of_box in enumerate(box):
    df = elem_of_box.to_dataframe(i)
    df.to_csv("ocr_table" + "/" + os.path.basename(f"{file}_{elem_of_box.x1}_{elem_of_box.y1}.csv"), encoding="utf-8",
              index=False)

list_all_table = []
for elem_of_box in box:
    if elem_of_box.parent:
        continue
    json_list = elem_of_box.to_json()
    list_all_table.append(json_list)

with open("json" + "/" + os.path.basename(f"{file}.json"), "w", encoding="utf-8") as f:
    json.dump(list_all_table, f, ensure_ascii=False, indent=4, cls=SetEncoder)


# for cell in box:
#     if cell.childs:
#         image = cv2.imread(file)
#         # image = cv2.rectangle(img, (cell.x1, cell.y1), (cell.x2, cell.y2), (0, 255, 0), 2)
#         cv2.putText(image, 'PARENT', (cell.x1, cell.y1 - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
#         for child in cell.childs:
#             # image = cv2.rectangle(img, (child.x1, child.y1), (child.x2, child.y2), (0, 255, 255), 2)
#             cv2.putText(image, f'{child.x1, child.x2, child.y1, child.y2}', (child.x1, child.y1 - 8),
#                         cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.4,
#                         (0, 0, 255), 1)
#
#         cv2.imwrite("test.jpg", image)
#         plotting = plt.imshow(image)
#         plt.show()
