import cv2
import numpy as np
import time


class detectv5:
    def __init__(self, img_path, model, imgs_w, imgs_h, conf_threshold, score_threshold, nms_threshold):
        self.conf = conf_threshold
        self.score = score_threshold
        self.nms = nms_threshold
        self.img = img_path
        self.model = model
        self.img_w = imgs_w
        self.img_h = imgs_h

    def __call__(self):
        img = cv2.imread(self.img)
        net = cv2.dnn.readNetFromONNX(self.model)
        self.detection(img, net)


    def detection(self, img, net):
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, (self.img_w, self.img_h), swapRB=True, mean=(0, 0, 0), crop=False)
        net.setInput(blob)
        t1 = time.time()
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        t2 = time.time()
        out = outputs[0]
        print('Opencv dnn yolov5 inference time: ', t2 - t1)
        n_detections = out.shape[1]
        height, width = img.shape[:2]
        x_scale = width / self.img_w
        y_scale = height / self.img_h
        conf_threshold = self.conf
        score_threshold = self.score
        nms_threshold = self.nms
        class_ids = []
        score = []
        boxes = []
        for i in range(n_detections):
            detect = out[0][i]
            confidence = detect[4]
            if confidence >= conf_threshold:
                class_score = detect[5:]
                class_id = np.argmax(class_score)
                if class_score[class_id] > score_threshold:
                    score.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                    left = int((x - w / 2) * x_scale)
                    top = int((y - h / 2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
                    cv2.putText(img, f'The brain tumor is detected', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        indices = cv2.dnn.NMSBoxes(boxes, np.array(
            score), conf_threshold, nms_threshold)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(img, (left, top), (left + width,
                          top + height), (0, 0, 255), 3)
            print('predictions: ', box, class_ids[i], score[i])
        while True:
            cv2.imshow("detection result", img)
            key = cv2.waitKey(1)
            if key == 27:  # 27 is the ASCII code for the 'Esc' key
                break
        cv2.destroyAllWindows()


# Specify your image path and other parameters here
image_path = 'test_photos/00000_140.jpg'
weights_path = 'tumor_detector_sagittal.onnx'
imgs_w = 640
imgs_h = 640
conf_thres = 0.7
score_thres = 0.5
nms_thres = 0.5
instance = detectv5(image_path, weights_path, imgs_w, imgs_h,
                    conf_thres, score_thres, nms_thres)
instance()
