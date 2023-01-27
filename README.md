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
- sklearn

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
2. Navigate to the cloned repository: cd unet

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