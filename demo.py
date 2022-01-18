from detector import Yolov5
import cv2
from yolov5.utils.plots import plot_one_box
from trackers.tracker import ObjectTracker
import os


if __name__ == "__main__":
    detector = Yolov5(
        weight_path="./yolov5/weights/yolov5s.pt", device="0", img_hw=(640, 640)
    )

    Track = True
    detector.show = False if Track else True
    detector.pause = False

    num_lines = 1
    lineStat = []

    save = True  
    save_dir = "./output"  
    os.makedirs(save_dir, exist_ok=True)

    type = 'bytetrack'
    tracker = ObjectTracker(type=type)
    conf_thresh = 0.2 if type == "bytetrack" else 0.4

    pause = True
    test_video = "/d/projects/YOLOV5Tracker/test.mp4"
    cap = cv2.VideoCapture(test_video)

    fps = cap.get(cv2.CAP_PROP_FPS)
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

        img, img_raw = detector.preprocess(frame, auto=True)
        preds, _ = detector.dynamic_detect(
            img, [img_raw], classes=[0], conf_threshold=conf_thresh
        )
        if not Track:
            continue
        box = preds[0][:, :4].cpu()
        conf = preds[0][:, 4].cpu()
        cls = preds[0][:, 5].cpu()

        tracks = tracker.update(bboxes=box, scores=conf, cls=cls, ori_img=img_raw)

        for i, track in enumerate(tracks):
            box = [int(b) for b in track[:4]]
            id = track[4]
            pt = ((box[0] + box[2]) // 2, box[3])  

            plot_one_box(
                box, img_raw, label=None, color=(20, 20, 255), line_thickness=2
            )
            # cv2.rectangle(draw_image, (box[0], box[1]), (box[2], box[3]), (20, 20, 255))
            text = "{}".format(id)
            cv2.putText(
                img_raw,
                text,
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if vid_writer is not None:
            vid_writer.write(frame)

        cv2.imshow("p", img_raw)
        key = cv2.waitKey(0 if pause else 1)
        pause = True if key == ord(" ") else False
        if key == ord("q") or key == ord("e") or key == 27:
            break