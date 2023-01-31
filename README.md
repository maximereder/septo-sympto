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
leaf,leaf_area_px,leaf_area_cm,necrosis_number,necrosis_area_ratio,necrosis_area_cm,pycnidias_number,pycnidias_area_px,pycnidias_area_cm,pycnidias_number_per_leaf_cm2,pycnidias_number_per_necrosis_cm2,pycnidias_area_cm2_per_necrosis_area_cm2,pycnidias_mean_area_cm2
9__1,680969.5,32.3886,2,0.707,22.9083,340,14938.5055,0.7105,10.497520732603448,14.841782236132754,0.03101495964344801,0.0020897058823529414
88__1,648293.5,30.8344,2,0.645,19.8957,614,21092.5784,1.0032,19.91282463741795,30.86093980106254,0.05042295571404876,0.0016338762214983715
14__1,638934.0,30.3893,1,0.855,25.9768,413,13490.1368,0.6416,13.590309747180752,15.898802007945552,0.024698962150842288,0.001553510895883777
20__1,821680.5,39.0811,2,0.417,16.305,13,570.0783,0.0271,0.3326416093712818,0.7973014412756824,0.001662066850659307,0.0020846153846153844
43__1,575263.5,27.3609,2,0.735,20.1063,438,18033.416,0.8577,16.008245342806706,21.784216887244295,0.042658271288103726,0.001958219178082192
51__1,708187.0,33.6831,1,0.333,11.2022,197,7094.4192,0.3374,5.848630322030929,17.585831354555356,0.030119083751405974,0.001712690355329949
25__1,637750.0,30.3329,1,0.644,19.5433,430,18773.0449,0.8929,14.176026690491183,22.002425383635313,0.04568829215127435,0.002076511627906977
11__1,673435.5,32.0302,1,0.911,29.1717,598,25491.0139,1.2124,18.669880300466435,20.499319545998347,0.04156082778857591,0.0020274247491638793
85__1,677470.0,32.2221,1,0.238,7.6772,269,11847.1915,0.5635,8.348307528063039,35.03881623508571,0.07339915594227062,0.002094795539033457
50__1,704669.5,33.5158,4,0.463,15.5219,380,15119.8409,0.7191,11.337936137582872,24.481538986850836,0.046328091277485356,0.0018923684210526315
```

## Requirements
The project requires the following libraries to be installed:

- OpenCV
- TensorFlow
- PIL
- numpy
- pandas
- scipy
- sklearn
- ipython
- matplotlib
- psutil
- seaborn
- tqdm

Download the requirements.txt file and run the following command to install the required libraries:
- [pycnidia-model.pt](https://drive.google.com/file/d/1WLIej7263MieoIrfGBtN7ljiZpE4NZy1/view?usp=share_link)
- [necrosis-model-375.h5](https://drive.google.com/file/d/1BPOsgdUjoA8uCGht4-kL2Er3SbB4JalR/view?usp=share_link)

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
- `-pc` or `pixels_for_cm` : specify the number of pixels per cm. Default is 145.
- `-pt` or `--pycnidia_threshold` : specify the confidence threshold for pycnidia detection. Default is 0.3.
- `-pn` or `--necrosis_threshold` : specify the confidence threshold for necrosis detection. Default is 0.8.
- `dm` or `draw_mode` : specify what to draw on the image. Can be 'pycnidias' 'necrosis'. Default is 'all'.
- `-sm` or `--save-masks` : specify if you want to save the masks. Default is False.
- `-s` or `--save` : specify if you want to save the results in the csv file. Default is True.
Note: Please make sure the pre-trained model is in the specified path and the specified folder contains the images with the specified extension.

## Train weights

Two weights are provided in the models folder, `necrosis-model-375.h5` and `pycnidia-model.pt`, respectively for pycnidias detection and necrosis detection.

To train new weights for you application, please use the following tutorials : 
- Pycnidias (YOLOv5) : https://github.com/ultralytics/yolov5
- Necrosis (U-Net) : https://github.com/maximereder/unet

## YOLOv5 Custom Training

This guide explains how to train your own custom dataset with YOLOv5, quickly. For more information, see the [YOLOv5 documentation](https://github.com/ultralytics/yolov5).

### Before You Start
1. Clone the YOLOv5 repository: git clone https://github.com/ultralytics/yolov5
2. Navigate to the cloned repository: cd yolov5
3. Install the requirements: `pip install -r requirements.txt`

### Train on Custom Data
1. **Create Dataset**: YOLOv5 models must be trained on labelled data. There are two options for creating your dataset:
    - Use [Roboflow](https://roboflow.com/) to label, prepare, and host your custom data automatically in YOLO format.
    - Manually prepare your dataset.

2. **Select a Model**: Select a pretrained model to start training from. For example, YOLOv5s is the second-smallest and fastest model available.
3. **Train**: Train a YOLOv5s model on COCO128 by specifying dataset, batch-size, image size and either pretrained --weights yolov5s.pt (recommended), or randomly initialized --weights '' --cfg yolov5s.yaml (not recommended).

```
python train.py --img <image_size> --batch <batch_size> --epochs <epochs> --data <data_yaml_file_path> --weights <weights_path>
```

Example: `python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt`

You may adapt batch size and image size to your hardware. Indeed, the larger the batch the more memory it consumes. For example, if you have a 16GB GPU, you can train at batch size 32 and image size 640. If you have a 32GB GPU, you can train at batch size 64 and image size 1280. If you have a 64GB GPU, you can train at batch size 128 and image size 2560. If you have a 128GB GPU, you can train at batch size 256 and image size 5120.

4. **Visualize**: Track and visualize model metrics in real time using Comet Logging and Visualization or ClearML Logging and Automation.
    - Comet: `pip install comet_ml`
    - ClearML: `pip install clearml`

Training results are automatically logged with Tensorboard and CSV loggers to runs/train, with a new experiment directory created for each new training as runs/train/exp2, runs/train/exp3, etc.

## U-Net Custom Training

This guide explains how to train your own custom dataset with U-Net, quickly. For more information, see the [U-Net documentation](https://github.com/maximereder/unet)

### Before You Start
1. Clone the U-Net repository: https://github.com/maximereder/unet.git
2. Navigate to the cloned repository: `cd unet`

### Usage
To train a model, run the following command:

```
python train.py --data <data_folder> --csv <csv_output> --model <model_output> --epochs <epochs> --batch-size <batch_size> --img_ext <image_extension> --mask_ext <mask_extension> --imgsz <image_size>
```

You can also specify the following arguments:

- `--data`: Data folder name. Default: data
- `--csv`: CSV to output name. Default: results_unet_train.csv
- `--model`: Model to output name. Default: model.h5
- `--epochs`: Number of epochs for training. Default: 100
- `--batch-size`: Batch size for training. Default: 2
- `--img_ext`: Image extension. Default: .jpg
- `--mask_ext`: Masks extension. Default: .png
- `--imgsz`: Image size for inference. Default: [304, 3072]

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