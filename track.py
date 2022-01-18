from tracker.bytetrack import BYTETracker
import yaml
import numpy as np

class ObjectTracker:


    SUPPORTED = ["bytetrack"]
    def __init__(self, type, config=None):

        if type not in self.SUPPORTED:
            raise ValueError("Not supported!!!!!!!!")

        with open(config, errors='ignore') as f:
            cfg = yaml.safe_load(f)
        if type == 'bytetrack':
            self.Tracker = BYTETracker(**cfg)
            self.args = ["bboxes", "scores"]

        self.type = type


    def update(self, **kwargs):
        outputs = self.Tracker.update(*[kwargs.get(a, None) for a in self.args])
        if self.type == "bytetrack":
            tracks = []
            for output in outputs:
                x1, y1, x2, y2 = output.tlbr
                tracks.append([x1, y1, x2, y2, output.track_id])
            if len(tracks):
                tracks = np.stack(tracks, axis=0)
        return tracks
