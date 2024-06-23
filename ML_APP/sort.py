import numpy as np

class Sort:
    def __init__(self):
        self.tracks = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        if len(dets) == 0:
            for track in self.tracks:
                track['age'] += 1
                track['missed'] += 1
            return np.array([track['bbox'] for track in self.tracks])
        else:
            new_tracks = []
            for det in dets:
                matched = False
                for track in self.tracks:
                    if self.iou(det, track['bbox']) > 0.5:
                        track['bbox'] = det
                        track['age'] += 1
                        track['missed'] = 0
                        matched = True
                        break
                if not matched:
                    new_tracks.append({'bbox': det, 'age': 1, 'missed': 0, 'id': len(self.tracks) + len(new_tracks)})
            for track in self.tracks:
                if track['missed'] > 5:
                    self.tracks.remove(track)
            self.tracks.extend(new_tracks)
            return np.array([track['bbox'] for track in self.tracks])

    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return intersection / (area_a + area_b - intersection)