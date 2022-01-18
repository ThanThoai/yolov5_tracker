import cv2
import random
import os
from track import ObjectTracker
import detecter
import copy

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

import time

if __name__ == "__main__":

    detector = detecter.Detector(ckpt = "weight/face_detection_v1.0.pt")

    Track = True
    detector.show = False if Track else True
    detector.pause = False

    num_lines = 1
    lineStat = []

    save = True 
    save_dir = "./output"
    os.makedirs(save_dir, exist_ok=True)

    type = 'bytetrack'
    tracker = ObjectTracker(type=type, config = "cfg.yaml")
    conf_thresh = 0.2

    # for video
    pause = True
    test_video = "video/test_2.mp4"
    cap = cv2.VideoCapture(test_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    fourcc = "mp4v"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vid_writer = (
        cv2.VideoWriter(
            os.path.join(save_dir, test_video.split(os.sep)[-1]),
            cv2.VideoWriter_fourcc(*fourcc),
            fps if fps <= 30 else 25,
            (w, h),
        )
        if save
        else None
    )

    frame_num = 0
    if Track:
        cv2.namedWindow("p", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # img, img_raw = detector.predict(frame, auto=True)
        # preds, _ = detector.dynamic_detect(
        #     img, [img_raw], classes=[0], conf_threshold=conf_thresh
        # )
        preds = detector.predict(copy.deepcopy(frame))
        # print(preds)
        if not Track:
            continue
        if len(preds) > 0:
            box = preds[:, :4]
            conf = preds[:, 5]
            cls = preds[:, 4]
            t0 = time.time()
            tracks = tracker.update(bboxes=box, scores=conf, cls=cls, ori_img=copy.deepcopy(frame))
            print(time.time() - t0)
            for i, track in enumerate(tracks):
                box = [int(b) for b in track[:4]]
                # print(box)
                id = track[4]
                pt = ((box[0] + box[2]) // 2, box[3]) 

                plot_one_box(
                    box, frame, label=None, color=(20, 20, 255), line_thickness=2
                )
                # cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), (20, 20, 255))
                text = "{}".format(int(id))
                cv2.putText(
                    frame,
                    text,
                    (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if vid_writer is not None:
            vid_writer.write(frame)

        cv2.imshow("p", frame)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(" ") else False
        if key == ord("q") or key == ord("e") or key == 27:
            break