import os
import cv2, csv
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tools.metrics import dice_loss, dice_coef, iou
import torch
from tqdm import tqdm

'''
Deep learning model for the detection of Septoria leaf blotch and Pycnidia on wheat leaves.
This script is based on the work of the following authors:
PhD student: Laura MATHIEU
Deep learning engineer: Maxime REDER
MAIL: maximereder@live.fr
Web site: https://maximereder.fr
'''

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--images_input", dest="images_input", help="Images folder input name.", default='images')
parser.add_argument("-i", "--import", dest="csv_import", help="CSV to import name.", default=None)
parser.add_argument("-o", "--output", dest="csv_output", help="CSV to output name.", default='results.csv')
parser.add_argument("-nm", "--necrosis_model", dest="necrosis_model", help="Model path.", default='models/necrosis-model-375.h5')
parser.add_argument("-pm", "--pycnidia_model", dest="pycnidia_model", help="Model path.", default='models/pycnidia-model.pt')
parser.add_argument("-e", "--extension", dest="extension", help="Image extension.", default='.tif')
parser.add_argument("-is", "--imgsz", dest="imgsz", help="Image size for inference.", default=[304, 3072], nargs='+')
parser.add_argument("-d", "--device", dest="device", help="Device : 'cpu' or 'mps' for M1&M2 or 1 ... n for gpus", default='cpu')
parser.add_argument("-pc", "--pixels_for_cm", dest="pixels_for_cm", help="Pixels for 1 cm.", default=472)   
parser.add_argument("-pt", "--pycnidia_threshold", dest="pycnidia_threshold", help="Pycnidia confidence threshold.", default=0.3)
parser.add_argument("-pn", "--necrosis_threshold", dest="necrosis_threshold", help="Necrosis confidence threshold.", default=0.8)
parser.add_argument("-dm", "--draw_mode", dest="draw_mode", help="Draw mode : 'pycnidia' or 'necrosis' or 'all'. Default 'all'", default='all')
parser.add_argument("-sm", "--save-masks", dest="save_masks", help="Save masks", default=False)
parser.add_argument("-ns", "--no-save", dest="no_save", help="No save True or False", default=False)
args = parser.parse_args()

with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model_necrosis = tf.keras.models.load_model(args.necrosis_model)
        
model_pycnidia = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(), args.pycnidia_model))
model_pycnidia.to(args.device)
extension = args.extension
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" Gloabal variables used in the script """

if not os.path.exists(os.path.join(os.getcwd(), 'images')):
    os.makedirs(os.path.join(os.getcwd(), 'images'))

if not os.path.exists(os.path.join(os.getcwd(), 'import')):
    os.makedirs(os.path.join(os.getcwd(), 'import'))

if not os.path.exists(os.path.join(os.getcwd(), 'models')):
    os.makedirs(os.path.join(os.getcwd(), 'models'))

if not os.path.exists(os.path.join(os.getcwd(), 'outputs')):
    os.makedirs(os.path.join(os.getcwd(), 'outputs'))

i = len(os.listdir(os.path.join(os.getcwd(), 'outputs')))
image_directory = os.path.join(os.getcwd(), args.images_input) # Get the path to the images folder
output_directory = os.path.join(os.getcwd(), 'outputs', 'output_{}'.format(i)) # Get the path to the output folder
os.makedirs(output_directory) # Create the output folder
mask_path = os.path.join(output_directory, "masks") # Get the path to the masks folder
data_import_path = args.csv_import # Get the path to the CSV to import
result_name = args.csv_output # Get the name of the CSV to output
no_save = args.no_save # Get the no save value

H = int(args.imgsz[0])
W = int(args.imgsz[1])

def read_image(image_path):
    """Reads and processes an image.
    Parameters:
        path (str): Path to the image file.
    Returns:
        tuple: Original image and processed image.
    """
    x = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def remove_leaf_noise(image):
    if image.shape[0] > image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Set the range of colors to consider as part of the leaf
    low_green = np.array([0, 35, 65])
    high_green = np.array([255, 255, 255])

    # Create a mask for the leaf
    mask = cv2.inRange(hsv, low_green, high_green)

    # Find the contours of the leaf
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Copy pixel from largest contour of original image and paste to new image
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

    # Set (0,0,0) pixel to (255,255,255) to avoid black background
    new_image = cv2.bitwise_and(image, mask)
    new_image[new_image == 0] = 255

    # Convert the image to RGB
    cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    return new_image


def convert_model_to_base_coordinates(xmin, xmax, ymin, ymax, model_width, model_height, base_width, base_height):
    """Converts coordinates from one image size to another.
    Parameters:
        xmin (int): X min coordinate.
        ymin (int): Y min coordinate.
        xmax (int): X max coordinate.
        ymax (int): Y max coordinate.
        model_width (int): Width of the original image.
        model_height (int): Height of the original image.
        base_width (int): Width of the base image.
        base_height (int): Height of the base image.
    
    Returns:
        tuple: New coordinates.
    """

    # Calculate the ratio between the base and model image
    rX = base_width / model_width
    rY = base_height / model_height

    # Calculate the new coordinates
    new_xmin = xmin*rX
    new_xmax = xmax*rX
    new_ymin = ymin*rY
    new_ymax = ymax*rY
    
    return new_xmin, new_xmax, new_ymin, new_ymax

def convert_model_to_base_area(base_area, model_width, model_height, base_width, base_height):
    """Converts area from one image size to another.
    
    Parameters:
        model_area (int): Area of the model image.
        model_width (int): Width of the new image.
        model_height (int): Height of the new image.
        base_width (int): Width of the original image.
        base_height (int): Height of the original image.
    
    Returns:
        int: Area of the original image.
    """
    # Calculate the area of the original image
    old_area = base_area * (base_width * base_height) / (model_width * model_height)
    return old_area

def predict_pycnidia(image_path, result_image, image_not_resized_path):
    """Predicts pycnidia on an image.
    
    Parameters:
        image_path (str): Path to the image file.
        result_image (numpy.ndarray): Image to draw the results on.
        image_not_resized_path (str): Path to the crop image not resized file.
    
    Returns:
        tuple: Image with the results drawn on it and the coordinates of the detected pycnidia.
    """
    # Set the confidence level and maximum number of detections for the model
    model_pycnidia.conf = float(args.pycnidia_threshold)
    model_pycnidia.max_det = 10000

    # Use the model to detect pycnidia on the image
    results = model_pycnidia(image_path, size=W)

    # Get the coordinates of the detected pycnidia
    pycnidia_coords = results.pandas().xyxy[0]

    # Select only the pycnidia with confidence level above the specified threshold
    confident_pycnidia = pycnidia_coords[pycnidia_coords['confidence'] >= float(args.pycnidia_threshold)]
    xmins = confident_pycnidia['xmin']
    xmaxs = confident_pycnidia['xmax']
    ymins = confident_pycnidia['ymin']
    ymaxs = confident_pycnidia['ymax']

    # Get the confidence level of the detected pycnidia
    confidences = confident_pycnidia['confidence']

    # Get the shape of the original image
    original_image_shape = cv2.imread(image_not_resized_path).shape

    # Calculate the total surface of the detected pycnidia
    surface = 0
    for i in range(len(xmins)):
        old_xmins, old_xmaxs, old_ymins, old_ymaxs = convert_model_to_base_coordinates(xmins[i], xmaxs[i], ymins[i], ymaxs[i],  W, H, original_image_shape[1], original_image_shape[0])
        surface += np.round((old_xmaxs - old_xmins) * (old_ymaxs - old_ymins), 4)
        
    # Count the number of detected pycnidia
    number_of_pycnidia = len(xmins)

    # Draw circles at the center of each detected pycnidia on the result image
    if args.no_save == False and args.draw_mode == 'pycnidia' or args.draw_mode == 'all':
        for i in range(len(xmins)):
            center_coord = (int((xmins[i] + xmaxs[i]) / 2), int((ymins[i] + ymaxs[i]) / 2))
            cv2.circle(result_image, center_coord, 8, (255, int(confidences[i]*255), 255), 2)


    # Return the result image with the detected pycnidia marked, the total surface area of the detected pycnidia, and the number of detected pycnidia
    return result_image, surface, number_of_pycnidia

def convert_pixel_area_to_cm2(pixel_area, pixels_for_1_cm):
    """
    Convert pixel area to cm^2.

    Parameters:
    - pixel_area (float): Area in pixels.
    - rule (int): Rule of the image.

    Returns:
    - float: Area in cm^2.
    """
    return (pixel_area / pixels_for_1_cm**2)

def predict_necrosis_mask(image_path, result_image):
    """
    Detect necrosis on an image and draw contours on the image.
    
    Parameters:
    - image_path (str): Path to the image.
    - result_image (np.ndarray): Image to draw the contours on.
    
    Returns:
    - Tuple[np.ndarray, float, int]: Tuple containing the image with the contours drawn, the total area of the detected necrosis, and the number of detected necrosis.
    """
    # Load the image and use the model to predict a mask for the necrosis
    image = cv2.imread(image_path)
    x = cv2.resize(image, (W, H))
    x = x/255.0
    x = x.astype(np.float32)

    mask = np.squeeze(model_necrosis.predict(np.expand_dims(x, axis=0)))
    
    # Convert the mask to a binary image where pixels above the specified threshold are white and others are black
    mask = np.where(mask > float(args.necrosis_threshold), 255, 0)
    mask = mask.astype(np.uint8)
    
    # If the --save-masks flag is set, save the mask image to the specified directory
    if args.save_masks:
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
        cv2.imwrite(os.path.join(mask_path, image_path.split('/')[-1][:-4]+'.png'), mask)
        
    # Find all contours in the mask image
    cnts_necrosis_full, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Initialize variables to store the total area of the detected necrosis and the number of detected necrosis
    necrosis_area = 0
    necrosis_number = 0

    # Filter out contours that are too small or have a high perimeter-to-area ratio
    cnts_necrosis = []
    for cnt in cnts_necrosis_full:
        area = cv2.contourArea(cnt)
        if area > 300:
            perimeter = cv2.arcLength(cnt, True)
            ratio = round(perimeter / area, 3)
            if ratio < 0.9:
                cnts_necrosis.append(cnt)
                # Draw the contour on the result image
                if args.no_save == False and args.draw_mode == 'necrosis' or args.draw_mode == 'all':
                    cv2.drawContours(result_image, cnt, -1, (0, 255, 0), 2)
                necrosis_area += area
                necrosis_number += 1

    # Return the image with the contours drawn, the total area of the detected necrosis, and the number of detected necrosis
    return result_image, necrosis_area, necrosis_number

def get_leaf_area(image):
    """
    Calculate the total area of a leaf in an image.
    
    Parameters:
    - image (np.ndarray): Image of the leaf.
    
    Returns:
    - float: Total area of the leaf.
    """
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Set the lower and upper bounds for the leaf color range in HSV
    low_leaf = np.array([0, 35, 65])
    high_leaf = np.array([255, 255, 255])

    # Create a mask image where only pixels within the leaf color range are white and others are black
    mask_leaf = cv2.inRange(hsv, low_leaf, high_leaf)

    # Find all contours in the mask image
    cnts_leaf, _ = cv2.findContours(mask_leaf, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Initialize a variable to store the total area of the leaf
    leaf_area = 0

    # Add up the areas of all the contours
    for cnt in cnts_leaf:
        area = cv2.contourArea(cnt)
        leaf_area += area

    # Return the total area of the leaf
    return leaf_area

def get_image_informations(image_path, image_not_resized_path, file_name):
    """
    Analyze an image and extract information about the leaf, necrosis, and pycnidia.
    
    Parameters:
    - image_path (str): Path to the image.
    - image_not_resized_path (str): Path to the image that was not resized.
    - file_name (str): Name of the image file.
    
    Returns:
    - List[Union[str, float, int]]: List containing the file name, total area of the leaf, number of necrosis, ratio of necrosis area to leaf area, total area of necrosis, number of pycnidia, and total area of pycnidia.
    """

    # Load the image and resize it
    image = cv2.imread(image_path)
    original_image = cv2.imread(image_not_resized_path)

    # Get the original image shape
    original_image_shape = original_image.shape[:2]

    # Resize the image to the input size of the model
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    result_image = cv2.resize(image.copy(), (W, H), interpolation=cv2.INTER_AREA)

    # Calculate the total area of the leaf
    leaf_area = get_leaf_area(image)

    # Detect necrosis in the image and draw contours on the image
    result_image, necrosis_area, necrosis_number = predict_necrosis_mask(image_path, result_image)
    
    # Calculate the ratio of necrosis area to leaf area
    necrosis_ratio = round(necrosis_area / leaf_area, 3)

    # Detect pycnidia in the image and draw circles on the image
    result_image, pycnidia_area, pycnidia_number = predict_pycnidia(image_path, result_image, image_not_resized_path)

    # If the --save flag is set, save the result image to the specified directory
    if args.no_save == False:
        if not os.path.exists(os.path.join(output_directory, 'images_output')):
            os.mkdir(os.path.join(output_directory, 'images_output'))
        cv2.imwrite(os.path.join(output_directory, 'images_output', file_name) + '.jpg', result_image)

    # Convert the areas to the original image size, not for pycnidia because original pycnidia area is already calculated
    leaf_area = convert_model_to_base_area(leaf_area, W, H, original_image_shape[1], original_image_shape[0])
    necrosis_area = convert_model_to_base_area(necrosis_area, W, H, original_image_shape[1], original_image_shape[0])

    # Convert the areas to cm2
    leaf_area_cm = round(convert_pixel_area_to_cm2(leaf_area, args.pixels_for_cm), 4)
    necrosis_area_cm = round(convert_pixel_area_to_cm2(necrosis_area, args.pixels_for_cm), 4)
    pycnidia_area_cm = round(convert_pixel_area_to_cm2(pycnidia_area, args.pixels_for_cm), 4)

    row = []
    row.append(file_name)
    row.append(leaf_area)
    row.append(leaf_area_cm)
    row.append(necrosis_number)
    row.append(necrosis_ratio)
    row.append(necrosis_area_cm)
    row.append(pycnidia_number)
    row.append(pycnidia_area)
    row.append(pycnidia_area_cm)
    if leaf_area_cm != 0:
        row.append(necrosis_area_cm / leaf_area_cm)
    else:
        row.append(0)

    if necrosis_area_cm != 0:
        row.append(pycnidia_number / necrosis_area_cm)
        row.append(pycnidia_area_cm / necrosis_area_cm)
    else:
        row.append(0)
        row.append(0)

    if pycnidia_number != 0:
        row.append(pycnidia_area_cm / pycnidia_number)
    else:
        row.append(0)

    return row


def crop_images_from_directory():
    """
    Crop the leaf images from a directory of images.
        
    Returns:
    - None
    """
    print("\033[93m" + "\n" + "CROP LEAF ON IMAGES" + "\033[99m \n")
    
    # Get a list of all the files in the directory that have the specified extension
    files = [file for file in os.listdir(image_directory) if file.endswith(extension)]
    
    file_id = 0
    # Iterate through the files
    for file in tqdm(files):
        file_id += 1
        # Load the image
        image = cv2.imread(os.path.join(image_directory, file))
        
        # If the image was loaded successfully
        if image is not None:
            # Convert the image to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            # Set the minimum area for a leaf contour to be considered
            min_area = 50000
            leaf_area = 0

            # Set the range of colors to consider as part of the leaf
            low_green = np.array([0, 35, 65])
            high_green = np.array([255, 255, 255])

            # Create a mask to isolate the leaf in the image
            mask_leaf = cv2.inRange(hsv, low_green, high_green)

            # Find the contours of the leaf in the image
            cnts_leaf, _ = cv2.findContours(mask_leaf, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            i = 0
            # Iterate through the contours
            for contour in cnts_leaf:

                # Calculate the area of the contour
                area = cv2.contourArea(contour)

                # If the contour is larger than the minimum area
                if area > min_area:
                    leaf_area += area
                    i += 1

                    # Get the bounding box of the contour
                    x, y, w, h = cv2.boundingRect(contour)

                    # Crop the image to the bounding box
                    cropped = image[y:y + h, x:x + w]

                    # Save the cropped image to the cropped_not_resized directory
                    cv2.imwrite(os.path.join(image_directory, 'cropped_not_resized', file.split('.')[0]) + '__{}.jpg'.format(str(i)), cropped)

                    # Resize the cropped image
                    cropped = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_AREA)

                    # Remove the noise from the leaf
                    cropped = remove_leaf_noise(cropped)
                    
                    # Draw a white rectangle around the leaf contour on the original image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

                    # Save the cropped image to the cropped directory
                    cv2.imwrite(os.path.join(image_directory, 'cropped', file.split('.')[0]) + '__{}.jpg'.format(str(i)), cropped)
                    
        else:
            print("Cannot read properly : ", file)

def export_result(result_rows):
    """
    Exports the result rows to a CSV file.
    
    Parameters:
    - result_name (str): Name of the result file.
    - result_rows (List[List[Any]]): A list of rows with the results.
    
    Returns:
    - None
    """

    with open(os.path.join(output_directory, result_name), 'w', encoding='UTF8', newline='') as f:
        print('\033[92m' + '\n' + 'Create final result csv')
        writer = csv.writer(f)

        if args.csv_import != None:
            data_imported = pd.read_csv(data_import_path, sep=';', encoding="ISO-8859-1")

            header = []
            header.append('leaf')

            for e in data_imported.head().columns:
                header.append(str(e).strip())
            header = header + ['leaf_area_px', 'leaf_area_cm2', 'necrosis_number', 'necrosis_area_ratio', 'necrosis_area_cm2', 'pycnidia_number',
                               'pycnidia_area_px', 'pycnidia_area_cm2', 'pycnidia_number_per_leaf_cm2', 'pycnidia_number_per_necrosis_cm2', 'pycnidia_area_cm2_per_necrosis_area_cm2', 'pycnidia_mean_area_cm2']
            writer.writerow(header)

            rows = []
            for r_row in data_imported.iterrows():
            
                for i in range(len(result_rows)):
                    if r_row[1][0].split('__')[0] == result_rows[i][0].split('__')[0]:
                        row = []
                        row.append(result_rows[i][0].split('__')[0])
                        for z in range(0, data_imported.columns.size, 1):
                            row.append(r_row[1][z])
                        for y in range(1, 9, 1):
                            row.append(result_rows[i][y])
                        rows.append(row)
        else: 
            header = ['leaf', 'leaf_area_px', 'leaf_area_cm2', 'necrosis_number', 'necrosis_area_ratio', 'necrosis_area_cm2', 'pycnidia_number',
                               'pycnidia_area_px', 'pycnidia_area_cm2', 'pycnidia_number_per_leaf_cm2', 'pycnidia_number_per_necrosis_cm2', 'pycnidia_area_cm2_per_necrosis_area_cm2', 'pycnidia_mean_area_cm2']
            writer.writerow(header)
            rows = result_rows
                            

        for i in range(0, len(rows), 1):
            writer.writerow(rows[len(rows) - 1 - i])

        if args.no_save == False:
            print('\033[92m' + 'Save results to outputs folder.')

        print('\033[92m' + 'Save results at {}'.format(output_directory))
        print('\033[92m' + "Done!")


def analyze_images():
    """
    Analyze the images in the given directory and save the results in the given output directory.
    
    Parameters:
        result_name (str): The name of the result file.
    
    Returns:
        None
    """

    if not os.path.exists(os.path.join(image_directory, 'cropped')):
        os.mkdir(os.path.join(image_directory, 'cropped'))
    if not os.path.exists(os.path.join(image_directory, 'cropped_not_resized')):
        os.mkdir(os.path.join(image_directory, 'cropped_not_resized'))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    cropped_images_directory = os.path.join(image_directory, 'cropped')
    cropped_not_resized_images_directory = os.path.join(image_directory, 'cropped_not_resized')

    # Crop the images
    crop_images_from_directory()

    print('\033[93m' + '\n' + "INFER IMAGES INTO MODELS" + "\033[99m \n")
        
    rows = []
    i = 0
    for file in tqdm(os.listdir(cropped_images_directory)):
        i += 1
        # Infer the image
        row = get_image_informations(os.path.join(cropped_images_directory, file),  os.path.join(cropped_not_resized_images_directory, file),
                                     file.split('.')[0])
        # Add the row to the list of rows
        rows.append(row)

    # Export the results to a CSV file
    export_result(rows)


if __name__ == '__main__':
    print('\033[92m' + '\n' + "Septo-Sympto")
    print('\033[92m' + "Authors: Laura MATHIEU, Maxime REDER")

    # Parse the arguments
    analyze_images()