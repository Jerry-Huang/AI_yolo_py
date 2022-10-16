import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def detect(frame, net, conf_threshold, nms_threshold):
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale=1/255, size=(416, 416))

    class_indexes, scores, boxes = model.detect(frame, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

    return class_indexes, scores, boxes

def box_frame(frame, classe_names, class_indexes, scores, boxes):

    for (class_index, box) in enumerate(boxes):
            (x, y, w, h) = box;
            class_name = classe_names[class_indexes[class_index]]
            score = str(round(scores[class_index], 2))
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
            title_text = to_chinese(class_name) + ' ' + score

            frame = text2chinese(frame, title_text, (x, y-30))
            ##cv2.putText(img=frame, text=title_text, org=(x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0,255,0), thickness=2)
   
    return frame

def to_chinese(text):

    dict = {'PERSON': '人', 'BOTTLE':'瓶子', 'CUP': '杯子'}

    text_upper = text.upper()

    if text_upper in dict:
        return dict[text.upper()]
    else:
        return text

def capture_video(net, classe_names, conf_threshold, nms_threshold):

    capture = cv2.VideoCapture(0)

    while capture.isOpened():
        OK, frame = capture.read()
        if not OK:
            break
    
        frame = cv2.flip(src=frame, flipCode=2)
        class_indexes, scores, boxes = detect(frame, net, conf_threshold, nms_threshold)
        frame = box_frame(frame, classe_names, class_indexes, scores, boxes)
        cv2.imshow('detect', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()

def capture_image(net, classe_names, conf_threshold, nms_threshold, image_path_name):

    frame = cv2.imread(image_path_name)
    frame = cv2.resize(frame, (416, 416))

    class_indexes, scores, boxes = detect(frame, net, conf_threshold, nms_threshold)
    box_frame(frame, classe_names, class_indexes, scores, boxes)

    cv2.imshow('detect', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def text2chinese(frame, text, position, text_color=(0,255,0), text_size=30):
    if (isinstance(frame, np.ndarray)):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(frame)

    font_style = ImageFont.truetype('simsun.ttc', text_size, encoding='utf-8')
    draw.text(position, text, text_color, font=font_style)

    return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':

    basedir = 'E:/AI/yolo'
    yolov3_weights_path_name = os.path.join(basedir, 'yolov3.weights')
    yolov3_config_path_name = os.path.join(basedir, 'yolov3.cfg')
    coco_labels_path_name = os.path.join(basedir, 'coco.names')

    net = cv2.dnn.readNet(yolov3_weights_path_name, yolov3_config_path_name)

    #layer_names = net.getLayerNames()

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.35

    with open(coco_labels_path_name, 'r') as fp:
        classe_names = fp.read().splitlines()

    capture_video(net, classe_names, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)