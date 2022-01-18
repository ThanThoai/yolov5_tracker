import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanUpdater(object):
    def __init__(self, point):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], 
                              [0, 1, 0, 0]])

        # self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  
        self.kf.P *= 10.

        # self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[2:, 2:] *= 0.01
        self.kf.R *= 1000

        self.kf.x[:2] = point.reshape((2, 1))

        self.time_since_update = 0
        self.hit_streak = 0

    def predict(self):
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.kf.predict()
        pred_x = self.get_state()
        return pred_x

    def update(self, point):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(point)

    def get_state(self):
        return self.kf.x[:2].reshape((1, 2))


class StablePoint(object):
    """
    Using KalmanFilter update points.

    Example with track points:
    ids = np.array(tracks[:, 4])
    boxes = np.array(tracks[:, :4])
    if len(boxes):
        x1, y1, x2, y2 = boxes.T
        w, h = x2 - x1, y2 - y1
    
        points = np.stack([(x2 + x1) / 2, y1 + 5 * h], axis=1)
    
        out = stable_point.update(ids=ids, points=points)
        points = out[:, :2]
        ids = out[:, 2]
    
    for ib, id in enumerate(ids):
        pt = tuple([int(p) for p in points[ib]])
        box = tracks[id]
    """
    def __init__(self, max_age=3, min_hits=3):
        self.kalmans = {}
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0


    def update(self, ids=np.empty((0, 1)), points=np.empty((0, 2))):
        self.frame_count += 1
        ret = []
        for _, v in self.kalmans.items():
            v.predict()
        for i, idx in enumerate(ids):
            if self.kalmans.get(idx, None) is None:
                self.kalmans[idx] = KalmanUpdater(points[i])
            else:
                self.kalmans[idx].update(points[i])

        clear_idx = []
        for idx, kalman in self.kalmans.items():
            point = kalman.get_state()[0]
            if (kalman.time_since_update < 1) and \
                    (kalman.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((point, [idx])).reshape(1, -1))

            if kalman.time_since_update > self.max_age:
                clear_idx.append(idx)

        for idx in clear_idx:
            self.kalmans.pop(idx)
        if len(ret):
            return np.concatenate(ret, axis=0)
        return np.empty((0, 3))

# Example
# ids = np.array(tracks[:, 4])
# boxes = np.array(tracks[:, :4])
# if len(boxes):
#     x1, y1, x2, y2 = boxes.T
#     w, h = x2 - x1, y2 - y1
# 
#     points = np.stack([(x2 + x1) / 2, y1 + 5 * h], axis=1)
# 
#     out = stable_point.update(ids=ids, points=points)
#     points = out[:, :2]
#     ids = out[:, 2]
# 
# for ib, id in enumerate(ids):
#     pt = tuple([int(p) for p in points[ib]])
#     box = tracks[id]