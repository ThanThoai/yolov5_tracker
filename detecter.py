import torch
import numpy as np

class Detector:

    def __init__(self, ckpt = None, conf = 0.25, iou = 0.5):
        self.model = self.load_model(ckpt, conf = conf, iou = iou)

    
    def load_model(self, ckpt, conf, iou):

        model = torch.hub.load("ultralytics/yolov5", "custom", path = ckpt)
        model.conf = conf
        model.iou = iou
        model.classes = None
        model.multi_label = False
        model.max_det = 1000
        model.agnostic = True
        return model
    
    def predict(self, image):
        image = image[..., ::-1]
        result = self.model(image, size=  1280)
        boxes = []
        # print(result.pandas().xyxy[0])
        for idx, row in result.pandas().xyxy[0].iterrows():
            if row['class'] == 2:
                boxes.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), int(row['class']), float(row['confidence'])])
        return np.array(boxes)
