'''
Author: Swarali Desai
'''
# USAGE
# python autoannotate.py --image images/baggage_claim.jpg --yolo yolo-coco --yoloNames yolo.names --yoloWeights yolov3.weights --yolocfg yolov3.cfg

import numpy as np
import argparse
import time
import cv2
import os
from pascal_voc_writer import Writer
import base64

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to input video")
ap.add_argument("-s", "--seconds",
                help="frame conversion after every specified seconds")
ap.add_argument("-i", "--image_dir",
                help="path to image_dir")
ap.add_argument("-o", "--out_dir",
                help="path to store txt generated")
ap.add_argument("-y", "--yolo", required=True,
                help="base path to YOLO directory")
ap.add_argument("-yn", "--yoloNames", required=True,
                help="class names")
ap.add_argument("-yw", "--yoloWeights", required=True,
                help="weights")
ap.add_argument("-yc", "--yolocfg", required=True,
                help="cfg file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

'''
Function to get annotated images
'''
def get_annotations(image_names_list, annotation_folder_path):
    for j, image_path in enumerate(image_names_list):
        filename, extension = os.path.splitext(image_path)
        print(filename)
        image = cv2.imread(os.path.join(UNANNOTATED_IMAGES_DIR, image_path))
        (H, W) = image.shape[:2]

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # print("yes")
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    # boxes.append([centerX, centerY, width, height])
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
            print("num of detections:", len(idxs))
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = ((float(boxes[i][0] + float(boxes[i][2] / 2))) / W, (float(boxes[i][1] + float(boxes[i][3] / 2))) / H)
                (w, h) = (boxes[i][2] / W, boxes[i][3] / H)
                # print(x, y, w, h)
                with open(os.path.join(annotation_folder_path, filename + '.txt'), 'a') as f:
                    f.write(str(classIDs[i]) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + '\n')
                cv2.imwrite(os.path.join(annotation_folder_path, filename + extension), image)

'''
Function to convert video to frames at specified frame interval
'''
def convert_video_to_frames(image_dir, video_file, seconds):
    # Create video cap object
    video_capture = cv2.VideoCapture(video_file)
    i = 0   # counter
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
    multiplier = fps * seconds
    success = True
    while success:
        frameId = int(round(video_capture.get(1)))
        success, frame = video_capture.read()
        # write frame to directory every specified second
        if frameId % multiplier == 0:
            cv2.imwrite(os.path.join(image_dir, str(i) + '.jpg'), frame)
            i += 1
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Read labels from labels file and store it in a list
    labelsPath = os.path.sep.join([args["yolo"], args["yoloNames"]])
    LABELS = open(labelsPath).read().strip().split("\n")

    # Set random colors for labelled images bounding boxes
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # Get path for weights file and config file
    weightsPath = os.path.sep.join([args["yolo"], args["yoloWeights"]])
    configPath = os.path.sep.join([args["yolo"], args["yolocfg"]])

    # Load model from disk
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Create folder if not exists, in case of video folder will be absent
    if not os.path.exists(os.path.join(os.getcwd(),args["image_dir"])):
        os.makedirs(os.path.join(os.getcwd(),args["image_dir"]))
    UNANNOTATED_IMAGES_DIR = os.path.join(os.getcwd(),args["image_dir"])

    # If video passed as argument convert video to separate frames for processing
    if args["video"]:
        VIDEO = args["video"]
        seconds = args['seconds']
        convert_video_to_frames(UNANNOTATED_IMAGES_DIR, VIDEO)

    TOTAL_IMAGES = len(os.listdir(UNANNOTATED_IMAGES_DIR))
    UNANNOTATED_IMAGES_LIST = os.listdir(UNANNOTATED_IMAGES_DIR)
    ANNOTATED_IMAGES_PATH = "annotated_images"

    # Create folder for annotated images
    if not os.path.exists(ANNOTATED_IMAGES_PATH):
        os.makedirs(ANNOTATED_IMAGES_PATH)

    # Process images to get annotated images
    get_annotations(UNANNOTATED_IMAGES_LIST, ANNOTATED_IMAGES_PATH)


