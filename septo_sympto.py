import os
import cv2, csv
import numpy as np
import pandas as pd
import argparse
from tensorflow import keras
import torch
from PIL import Image, ImageDraw
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
parser.add_argument("-w", "--images", dest="images", help="Images folder name.", default='images')
parser.add_argument("-i", "--import", dest="csv_import", help="CSV to import name.", default='LM1_all__input.csv')
parser.add_argument("-o", "--output", dest="csv_output", help="CSV to output name.", default='results.csv')
parser.add_argument("-e", "--extension", dest="extension", help="Image extension.", default='.tif')
parser.add_argument("-is", "--imgsz", dest="imgsz", help="Image size for inference.", default=3070)
parser.add_argument("-d", "--device", dest="device", help="Device : 'cpu' or 'mps' for M1&M2 or 1 ... n for gpus", default='cpu')
parser.add_argument("-pt", "--pycnidia_threshold", dest="pycnidia_threshold", help="Pycnidia confidence threshold.", default=0.3)
parser.add_argument("-pn", "--necrosis_threshold", dest="necrosis_threshold", help="Necrosis confidence threshold.", default=0.8)
parser.add_argument("-sm", "--save-masks", dest="save_masks", help="Save masks", default=False)
parser.add_argument("-s", "--save", dest="save", help="Save True or False", default=True)
args = parser.parse_args()

model_necrosis = keras.models.load_model(os.path.join(os.getcwd(), 'models', 'necrosis-model-375.h5'))
model_pycnidia = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(), 'models', 'pycnidia-model.pt'))
model_pycnidia.to(args.device)
save = args.save
extension = args.extension

def predict_pycnidia(image_path, result_image, model_pycnidia):
    model_pycnidia.conf = float(args.pycnidia_threshold)
    model_pycnidia.max_det = 10000

    results = model_pycnidia(image_path, size=int(args.imgsz))

    r = results.pandas().xyxy[0]
    cond = (r['confidence'] >= float(args.pycnidia_threshold))
    xmins = r[cond]['xmin']
    xmaxs = r[cond]['xmax']
    ymins = r[cond]['ymin']
    ymaxs = r[cond]['ymax']

    surface = np.round(np.sum([(xmaxs[i] - xmins[i]) * (ymaxs[i] - ymins[i]) for i in range(len(xmins))]), 4)
    number_of_pycnidia = len(xmins)

    for i in range(len(xmins)):
        center_coord = (int((xmins[i] + xmaxs[i]) / 2), int((ymins[i] + ymaxs[i]) / 2))
        cv2.circle(result_image, center_coord, 2, (0, 0, 255), 2)
        #cv2.rectangle(result_image, (int(xmins[i])+6, int(ymins[i])+6), (int(xmaxs[i])-6, int(ymaxs[i])-6), (255, 0, 0), 2)

    return result_image, surface, number_of_pycnidia

def convert_pixel_area_to_cm2(pixel_area, dpi):
    return 2.54*(pixel_area / dpi)

def predict_necrosis_mask(image_path, mask_path, result_image):
    image = cv2.imread(image_path)
    mask = np.squeeze(model_necrosis.predict(np.expand_dims(image, axis=0)))
    mask = np.where(mask > float(args.necrosis_threshold), 255, 0)
    mask = mask.astype(np.uint8)

    if args.save_masks:
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        cv2.imwrite(os.path.join(mask_path, image_path.split('/')[-1][:-4]+'.png'), mask)
        
    cnts_necrosis_full, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

def get_image_informations(output_directory, image_path, mask_folder_path, file_name, save):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (3070, 300), interpolation=cv2.INTER_AREA)
    result_image = cv2.resize(image.copy(), (3070, 300), interpolation=cv2.INTER_AREA)

    leaf_area = get_leaf_area(image)

    result_image, necrosis_area, necrosis_number = predict_necrosis_mask(image_path, mask_folder_path, result_image)
    necrosis_ratio = round(necrosis_area / leaf_area, 3)

    result_image, pycnidia_area, pycnidia_number = predict_pycnidia(image_path, result_image, model_pycnidia)

    if save:
        if not os.path.exists(os.path.join(output_directory, 'images')):
            os.mkdir(os.path.join(output_directory, 'images'))
        cv2.imwrite(os.path.join(output_directory, 'images', file_name) + '.jpg', result_image)

    row = []
    '''row.append(file_name)
    row.append(leaf_area)
    row.append(round(leaf_area * np.power(1/145, 2), 4))
    row.append(necrosis_number)
    row.append(necrosis_ratio)
    row.append(round(necrosis_area * np.power(1/145, 2), 4))
    row.append(pycnidia_number)
    row.append(pycnidia_area)
    row.append(round(pycnidia_area * np.power(1/145, 2), 4))'''

    row.append(file_name)
    row.append(leaf_area)
    row.append(round(convert_pixel_area_to_cm2(leaf_area, 1200), 4))
    row.append(necrosis_number)
    row.append(necrosis_ratio)
    row.append(round(convert_pixel_area_to_cm2(necrosis_area, 1200), 4))
    row.append(pycnidia_number)
    row.append(pycnidia_area)
    row.append(round(convert_pixel_area_to_cm2(pycnidia_area, 1200), 4))

    return row


def crop_images_from_directory(image_directory):

    print('\033[93m' + '\n' + "CROP LEAF ON IMAGES" + "\033[99m \n")

    file_id = 0
    for file in os.listdir(image_directory):
        if file.endswith(extension):
            file_id += 1
            image = cv2.imread(os.path.join(image_directory, file))

            number_of_images = len([file for file in os.listdir(image_directory) if file.endswith('.tif')])

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
                            os.path.join(image_directory, 'cropped', file.split('.')[0]) + '__{}.jpg'.format(str(i)),
                            cropped)
            else:
                print("Cannot read properly : ", file)


def export_result(output_directory, data_imported_path, result_rows):
    data_imported_csv = os.path.join(os.getcwd(), data_imported_path)
    data_imported = pd.read_csv(data_imported_csv, sep=';', encoding="ISO-8859-1")

    with open(os.path.join(output_directory, args.csv_output), 'w', encoding='UTF8', newline='') as f:
        print('\033[92m' + '\n' + 'Create final result csv')
        writer = csv.writer(f)

        header = []
        header.append('leaf')

        for e in data_imported.head().columns:
            header.append(str(e).strip())
        header = header + ['leaf_area_px', 'leaf_area_cm', 'necrosis_number', 'necrosis_area_ratio', 'pycnidia_area_px',
                           'pycnidia_number', 'pycnidia_area_cm']
        writer.writerow(header)

        rows = []
        for r_row in data_imported.iterrows():
        
            for i in range(len(result_rows)):
                if r_row[1][0].split('__')[0] == result_rows[i][0].split('__')[0]:
                    row = []
                    row.append(result_rows[i][0].split('__')[0])
                    for z in range(0, data_imported.columns.size, 1):
                        row.append(r_row[1][z])
                    for y in range(1, 8, 1):
                        row.append(result_rows[i][y])
                    rows.append(row)

        for i in range(0, len(rows), 1):
            writer.writerow(rows[len(rows) - 1 - i])

        if save:
            print('\033[92m' + 'Save images to outputs folder.')

        print('\033[92m' + "Done!")


def analyze_images(image_directory, output_directory, reactivip, result_name, save):
    result_csv_path = os.path.join(os.getcwd(), result_name)
    if not os.path.exists(os.path.join(image_directory, 'cropped')):
        os.mkdir(os.path.join(image_directory, 'cropped'))
    if not os.path.exists(output_directory) and save:
        os.mkdir(output_directory)

    cropped_images_directory = os.path.join(image_directory, 'cropped')

    crop_images_from_directory(image_directory)

    print('\033[93m' + '\n' + "INFER IMAGES INTO MODELS" + "\033[99m \n")

    with open(result_csv_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['leaf', 'leaf_area_px', 'leaf_area_cm', 'necrosis_number', 'necrosis_area_ratio', 'pycnidia_area_px',
                         'pycnidia_number', 'pycnidia_area_cm'])
        rows = []
        i = 0
        for file in os.listdir(cropped_images_directory):
            i += 1
            print("\033[93m" + "Analyze {} {}/{}".format(file.split('.')[0] + extension, str(i),
                                            str(len(os.listdir(cropped_images_directory)))) + "\033[99m")
            row = get_image_informations(output_directory, os.path.join(cropped_images_directory, file), os.path.join(os.getcwd(), "masks"),
                                         file.split('.')[0], save)
            rows.append(row)

    export_result(output_directory, reactivip, rows)


if __name__ == '__main__':
    print('\033[92m' + '\n' + "Septo-Sympto V2")
    print('\033[92m' + "AUTHORS: Laura MATHIEU, Maxime REDER")

    #(image_directory, output_directory, reactivip, result_name, save):

    analyze_images(os.path.join(os.getcwd(), args.images), os.path.join(os.getcwd(), 'outputs'), args.csv_import, args.csv_output, args.save)

