# Leaf Necrosis and Pycnidia Detection using Deep Learning
This project is a deep learning-based tool for detecting and classifying leaf necrosis and pycnidia in images of leaves. The tool is implemented using the OpenCV and TensorFlow libraries, and uses a pre-trained convolutional neural network to analyze the images.

![With pycnidias](/pictures/Cad_Rub_3_Rub_2__1__1__1.webp)

### Necrosis Detection
The script starts by loading a pre-trained U-Net model that can detect necrosis on leaves. It then uses the model to process an image, resizing it and converting it to HSV color space. It then applies a mask to the image to isolate the leaf area and counts the number of necrosis in the image. The necrosis ratio is also calculated as the ratio of necrosis area to leaf area.

### Pycnidia Detection
The script then uses convolutional filters and image processing techniques to detect and classify pycnidia. The pycnidia ratio is also calculated as the ratio of pycnidia area to leaf area.

## Results
The script calculates various statistics about the leaf, such as the leaf area, necrosis area, necrosis ratio, pycnidia area, pycnidia ratio and saves the mask image for further analysis.

```
leaf,leaf_area_px,leaf_area_cm,necrosis_number,necrosis_area_ratio,necrosis_area_cm,pycnidia_area_px,pycnidia_number,pycnidia_area_cm
Ber_Acc_3_Acc_2__1__1,820939.5,1737.6553,3,0.149,259.3319,238,12256.082,25.942
```

## Requirements
The project requires the following libraries to be installed:

- OpenCV
- TensorFlow
- PIL
- numpy
- pandas
- scipy

## Usage
To use the script, run the script with the following command line arguments:

Copy code
```py 
python septo_sympto.py -w <images_folder> -i <csv_import> -o <csv_output> -m <model_path> -e <image_extension> -is <image_size> -d <device> -pt <pycnidia_threshold> -pn <necrosis_threshold> -sm <save_masks> -s <save>
```

- `-w` or `--images` : specify the name of the folder containing the images. Default is 'images'.
- `-i` or `--import` : specify the name of the CSV file to import. Default is None.
- `-o` or `--output` : specify the name of the CSV file to output the results. Default is 'results.csv'.
- `-m` or `--model` : specify the path of the pre-trained model to use. Default is 'models/necrosis-model-375.h5'.
- `-e` or `--extension` : specify the extension of the images. Default is '.tif'.
- `-is` or `--imgsz` : specify the size of the images for inference. Default is [304, 3072].
- `-d` or `--device` : specify the device to use for inference. Can be 'cpu', 'mps' for M1&M2 or a number for specific GPU. Default is 'cpu'.
- `-pt` or `--pycnidia_threshold` : specify the confidence threshold for pycnidia detection. Default is 0.3.
- `-pn` or `--necrosis_threshold` : specify the confidence threshold for necrosis detection. Default is 0.8.
- `-sm` or `--save-masks` : specify if you want to save the masks. Default is False.
- `-s` or `--save` : specify if you want to save the results in the csv file. Default is True.
Note: Please make sure the pre-trained model is in the specified path and the specified folder contains the images with the specified extension.

## Train weights

Two weights are provided in the models folder, `necrosis-model-375.h5` and `pycnidia-model.pt`, respectively for pycnidias detection and necrosis detection.

To train new weights for you application, please use the following tutorials : 
- Pycnidias (YOLOv5) : https://github.com/ultralytics/yolov5
- Necrosis (U-Net) : https://github.com/maximereder/unet

## Authors

This script is based on the work of the following authors:
- Laura MATHIEU, PhD
    - Mail : laura.mathieu@supagro.fr 
    - Website : https://www.linkedin.com/in/laura-mathieu/
- Maxime REDER, Deep Learning Engineer
    - Mail : maximereder@live.fr 
    - Website : https://maximereder.fr

## License
This project is for personal use only and should not be distributed.