# Auto_annotate
To train any model we need data and this data needs to be annotated. Annotation of image data can be very difficult, because the dataset is huge and it becomes time consuming, to solve that I have created a script in python for auto-annotating the image dataset for YOLO models.

YOLO needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random objects in the image along with the desired objects, and should have a variety of backgrounds and lighting conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture.

[LabelImg](https://github.com/tzutalin/labelImg) is a software which lets you label image data in PASCAL VOC or YOLO format. For YOLO format we basically need the class of the object and the co-ordinates in a .txt file. In this project I have used YOLOv3 model for annotating the objects. The class names and config files are included in the yolo-coco folder. Along with it you will need to download the [yolov3.weights](https://pjreddie.com/darknet/yolo/) and put it in the same folder.

### Project Directory Structure
#### Auto_annotate

  > annotated_images: folder will be created if does not exist after running the script and will include images and .txt files.
  
  > data: It will include a predefined_classes.txt for LabelImg.
  
  > unlabelled_images: It will include all the images you want to auto-annotate.
  
  > yolo-coco: It will include the weights, cfg file and coco.names file for the model.
  
  > auto_annotate.py: script to auto-annotate images.
  
  > LabelImg: exe for manually checking the labeled data.
  
### Usage
To run the script on images run the following command:

> python autoannotate.py --image unlabelled_images --yolo yolo-coco --yoloNames yolo.names --yoloWeights yolov3.weights --yolocfg yolov3.cfg

To run the script on video run the following command:

> python autoannotate.py --image unlabelled_images --yolo yolo-coco --yoloNames yolo.names --yoloWeights yolov3.weights --yolocfg yolov3.cfg --video <path to video> --seconds <frame rate to capture frames from video>
  
Use python or python3 depending on your system.



  
