import json
import numpy as np
from pycocotools import mask
import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--workspace", dest="workspace", help="Workspace containing 'images' and 'masks' folder.", default=os.getcwd())
parser.add_argument("-o", "--output", dest="output", help="Output json name.", default='result.json')
args = parser.parse_args()

workspace = args.workspace
image_path = os.path.join(workspace, 'images')
mask_path = os.path.join(workspace, 'masks')

result = {}

info = {
        "year": "2022",
        "version": "1",
        "description": "Leaf necrosis dataset.",
        "contributor": "Laura MATHIEU & Maxime REDER",
        "url": "https://public.roboflow.ai/object-detection/undefined",
        "date_created": "2022-10-26T18:18:59+00:00"
    }

licenses = [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ],

categories = [
        {
            "id": 0,
            "name": "necrosis",
            "supercategory": ""
        }
    ]

result.update({"info": info})
result.update({"licenses": licenses})
result.update({"categories": categories})

image_id = 0
annotation_id = 0

for file in os.listdir(mask_path):
    if not file.endswith('.DS_Store') and file.endswith('.png'):
        image_name = file[:-4] + '.jpg'
        image_shape = cv2.imread(os.path.join(image_path, image_name)).shape

        m = cv2.imread(os.path.join(mask_path, file))
        m = cv2.resize(m, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        ret,m = cv2.threshold(m,10,255,cv2.THRESH_BINARY)
        
        ground_truth_binary_mask = m
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove small contours
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

        # Reducing the number of points in the contours
        contours = [cv2.approxPolyDP(contour, 0.0005 * cv2.arcLength(contour, True), True) for contour in contours]

        # Convert contours to segmentation
        all = []
        for contour in contours:
            segmentation = []
            for point in contour:
                segmentation.append(point[0][0])
                segmentation.append(point[0][1])
            all.append(np.array(segmentation).tolist())

        images_info = {
            "id": image_id,
            "license": 1,
            "file_name": image_name,
            "height": image_shape[0],
            "width": image_shape[1],
            "date_captured": "2022-10-26T18:18:59+00:00"
        }

        for seg in all:
            annotations_info = {
                    "segmentation": seg,
                    "area": ground_truth_area.tolist(),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": ground_truth_bounding_box.tolist(),
                    "category_id": 0,
                    "id": annotation_id
                }
            if 'annotations' not in result:
                result.update({"annotations": [annotations_info]})
            else: 
                result["annotations"].append(annotations_info)
            annotation_id+=1

        if 'images' not in result:
            result.update({"images": [images_info]})
        else: 
            result["images"].append(images_info)

        image_id+=1
                
        print("{} converted.".format(image_name))

print("Save results.json file at {}".format(workspace))

with open(os.path.join(workspace, args.output), 'w') as outfile:
    json.dump(result, outfile)

print("Done!")