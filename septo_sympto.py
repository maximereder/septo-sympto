import os
import cv2, csv
import numpy as np
import pandas as pd
import argparse
from tensorflow import keras
import torch
from PIL import Image
#from tqdm import tqdm

'''
Deep learning model for the detection of Septoria leaf blotch and Pycnidia on wheat leaves.
This script is based on the work of the following authors:

PhD student: Laura MATHIEU

Deep learning engineer: Maxime REDER
MAIL: maximereder@live.fr
Web site: https://maximereder.fr
'''

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--workspace", dest="workspace", help="Workspace path.", default=os.path.join(os.getcwd(), 'workspace'))
parser.add_argument("-i", "--import", dest="csv_import", help="CSV to import path.", default=os.path.join(os.getcwd(), 'LM1_all__input.csv'))
parser.add_argument("-o", "--output", dest="csv_output", help="CSV to output path.", default=os.path.join(os.getcwd(), 'results.csv'))
parser.add_argument("-e", "--extension", dest="extension", help="Image extension.", default='.tif')
parser.add_argument("-t", "--pycnidia_threshold", dest="pycnidia_threshold", help="Pycnidia confidence threshold.", default=0.3)
parser.add_argument("-s", "--save", dest="save", help="Save True or False", default=True)
args = parser.parse_args()

model_necrosis = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'necrosis-model-375.h5'))
model_pycnidia = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(), 'models', 'pycnidia-sota-6/weights/best.pt'))
save = args.save
extension = args.extension

def predict_pycnidia(image_path, result_image, model_pycnidia):
    results = model_pycnidia(image_path)
    r = results.pandas().xyxy[0]
    cond = (r['confidence'] >= args.pycnidia_threshold)
    xmins = r[cond]['xmin']
    xmaxs = r[cond]['xmax']
    ymins = r[cond]['ymin']
    ymaxs = r[cond]['ymax']

    surface = np.round(np.sum([(xmaxs[i] - xmins[i]) * (ymaxs[i] - ymins[i]) for i in range(len(xmins))]), 4)
    number_of_pycnidia = len(xmins)

    for i in range(len(xmins)):
        cv2.rectangle(result_image, (int(xmins[i])+6, int(ymins[i])+6), (int(xmaxs[i])-6, int(ymaxs[i])-6), (255, 0, 0), 2)

    return result_image, surface, number_of_pycnidia

def predict_necrosis_mask(image, result_image):
    mask = np.squeeze(model_necrosis.predict(np.expand_dims(image, axis=0)))
    mask = np.where(mask > 0.8, 255, 0)
    mask = mask.astype(np.uint8)

    cnts_necrosis_full, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    necrosis_area = 0
    necrosis_number = 0

    cnts_necrosis = []
    for cnt in cnts_necrosis_full:
        area = cv2.contourArea(cnt)
        if area > 300:
            perimeter = cv2.arcLength(cnt, True)
            ratio = round(perimeter / area, 3)
            if ratio < 0.9:
                cnts_necrosis.append(cnt)
                cv2.drawContours(result_image, cnt, -1, (0, 255, 0), 2)
                necrosis_area += area
                necrosis_number += 1

    return result_image, necrosis_area, necrosis_number

def get_leaf_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_leaf = np.array([0, 35, 65])
    high_leaf = np.array([255, 255, 255])
    mask_leaf = cv2.inRange(hsv, low_leaf, high_leaf)
    cnts_leaf, _ = cv2.findContours(mask_leaf, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    leaf_area = 0

    for cnt in cnts_leaf:
        area = cv2.contourArea(cnt)
        leaf_area += area

    return leaf_area

def get_image_informations(directory, image_path, file_name, save):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (3070, 300), interpolation=cv2.INTER_AREA)
    result_image = cv2.resize(image.copy(), (3070, 300), interpolation=cv2.INTER_AREA)

    leaf_area = get_leaf_area(image)

    result_image, necrosis_area, necrosis_number = predict_necrosis_mask(image, result_image)
    necrosis_ratio = round(necrosis_area / leaf_area, 3)

    result_image, pycnidia_area, pycnidia_number = predict_pycnidia(image_path, result_image, model_pycnidia)

    if save:
        cv2.imwrite(os.path.join(directory, 'artifacts', 'OUT', file_name) + '.jpg', result_image)

    row = []
    row.append(file_name)
    row.append(leaf_area)
    row.append(round(leaf_area * np.power(1/145, 2), 2))
    row.append(necrosis_number)
    row.append(necrosis_ratio)
    row.append(round(necrosis_area * np.power(1/145, 2), 2))
    row.append(pycnidia_number)
    row.append(pycnidia_area)
    row.append(round(pycnidia_area * np.power(1/145, 2), 4))

    return row


def crop_images_from_directory(directory):

    print('\033[93m' + '\n' + "CROP IMAGES" + "\033[99m \n")

    end = len([f for f in os.listdir(os.path.join(directory))
               if f.endswith(extension) and os.path.isfile(os.path.join(os.path.join(directory), f))])

    file_id = 0
    for file in os.listdir(os.path.join(directory)):
        if file.endswith(extension):
            file_id += 1
            image = cv2.imread(os.path.join(directory, file))

            number_of_images = len([file for file in os.listdir(os.path.join(directory)) if file.endswith('.tif')])

            if image is not None:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                min_area = 50000
                leaf_area = 0

                low_green = np.array([0, 35, 65])
                high_green = np.array([255, 255, 255])
                mask_leaf = cv2.inRange(hsv, low_green, high_green)
                cnts_leaf, _ = cv2.findContours(mask_leaf, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                i = 0
                for contour in cnts_leaf:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        leaf_area += area
                        i += 1
                        x, y, w, h = cv2.boundingRect(contour)
                        cropped = image[y:y + h, x:x + w]
                        cropped = cv2.resize(cropped, (3070, 300), interpolation=cv2.INTER_AREA)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
                        print("\033[93m" + "Crop {} {}/{}".format(file, str(file_id), number_of_images) + "\033[99m")
                        cv2.imwrite(
                            os.path.join(directory, 'artifacts', 'CROPPED', file.split('.')[0]) + '__{}.jpg'.format(str(i)),
                            cropped)
            else:
                print("Cannot read properly : ", file)


def export_result(directory, reactivip, result):
    react_data = pd.read_csv(reactivip, sep=';', encoding="ISO-8859-1")
    result_data = pd.read_csv(result)

    with open(os.path.join(directory, 'result.csv'), 'w', encoding='UTF8', newline='') as f:
        print('\033[92m' + '\n' + 'Create final result csv')
        writer = csv.writer(f)

        header = []
        header.append('leaf')
        for e in react_data.head().columns:
            header.append(str(e).strip())
        header = header + ['leaf_area_px', 'leaf_area_cm', 'necrosis_number', 'necrosis_area_ratio', 'pycnidia_area_px',
                           'pycnidia_number', 'pycnidia_area_cm']
        writer.writerow(header)

        rows = []
        for r_row in react_data.iterrows():
            for l_row in result_data.iterrows():
                if r_row[1][0].split('__')[0] == l_row[1][0].split('__')[0]:
                    row = []
                    row.append(l_row[1][0])
                    for i in range(0, react_data.columns.size, 1):
                        row.append(r_row[1][i])
                    for i in range(1, 8, 1):
                        row.append(l_row[1][i])
                    rows.append(row)

        for i in range(0, len(rows), 1):
            writer.writerow(rows[len(rows) - 1 - i])

        if save:
            print('\033[92m' + 'Save images to OUT folder.')

        print('\033[92m' + "Done!")


def analyze_images(directory, reactivip, result, save):
    if not os.path.exists(os.path.join(directory, 'artifacts')):
        os.mkdir(os.path.join(directory, 'artifacts'))
    if not os.path.exists(os.path.join(directory, 'artifacts', 'CROPPED')):
        os.mkdir(os.path.join(directory, 'artifacts', 'CROPPED'))
    if not os.path.exists(os.path.join(directory, 'artifacts', 'OUT')) and save:
        os.mkdir(os.path.join(directory, 'artifacts', 'OUT'))

    crop_images_from_directory(directory)

    print('\033[93m' + '\n' + "INFER IMAGES INTO MODELS" + "\033[99m \n")

    with open(result, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        rows = []
        i = 0
        for file in os.listdir(os.path.join(directory, 'artifacts', 'CROPPED')):
            i += 1
            print("\033[93m" + "Analyze {} {}/{}".format(file.split('.')[0] + extension, str(i),
                                            str(len(os.listdir(os.path.join(directory, 'artifacts', 'CROPPED'))))) + "\033[99m")
            row = get_image_informations(directory, os.path.join(directory, 'artifacts', 'CROPPED', file),
                                         file.split('.')[0], save)
            rows.append(row)

        for i in range(0, len(rows), 1):
            writer.writerow(rows[len(rows) - 1 - i])

    export_result(directory, reactivip, result)


if __name__ == '__main__':
    print('\033[92m' + '\n' + "Septo-Sympto V2")
    print('\033[92m' + "AUTHORS: Laura MATHIEU, Maxime REDER")
    
    analyze_images(args.workspace, args.csv_import, args.csv_output, args.save)

