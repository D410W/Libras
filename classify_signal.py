import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import ast

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

FINGER_LANDMARK_GROUPS = [
  [WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
  [WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
  [WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
  [WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP],
  [WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
]

FINGER_COLORS = [
  (0, 140, 255),   # Thumb  - orange
  (0, 220, 0),     # Index  - green
  (220, 0, 220),   # Middle - magenta
  (0, 220, 220),   # Ring   - cyan
  (220, 220, 0),   # Pinky  - yellow
]


def detect_hands(frame: numpy.ndarray, palm_detector, hand_pose) -> list[list[(int, int)]]:
  palms_pts = []
  
  palms = palm_detector.infer(frame)
  for palm in palms:
    result = hand_pose.infer(frame, palm)
    if result is None:
      continue
    landmarks = result[4:67].reshape(21, 3)
    pts = [(int(x), int(y)) for x, y, _ in landmarks]
    
    palms_pts.append(pts)
  
  return palms_pts


def draw_hand(frame: numpy.ndarray, pts: list[(int, int)]):
  for finger_idx, chain in enumerate(FINGER_LANDMARK_GROUPS):
    color = FINGER_COLORS[finger_idx]
    chain_pts = [pts[i] for i in chain]
    for a, b in zip(chain_pts, chain_pts[1:]):
      cv2.line(frame, a, b, color, 3, cv2.LINE_AA)
    for pt in chain_pts[1:]:
      cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)

  for i, j in [(INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP)]:
    cv2.line(frame, pts[i], pts[j], (255, 255, 255), 2, cv2.LINE_AA)

  cv2.circle(frame, pts[WRIST], 7, (255, 255, 255), -1, cv2.LINE_AA)

  for tip_id, color in zip(FINGER_TIPS, FINGER_COLORS):
    cv2.circle(frame, pts[tip_id], 9, color, 2, cv2.LINE_AA)
    
def flatten(input: list[tuple[int, int]]):
  x = []
  for tup in input:
    for el in tup:
      x.append(el)
  
  return x

def normalize_hand(points: list[(int, int)]) -> list[(int, int)]:
  INF = 999999
  minp = [0, 0]
  maxp = [INF, INF]
  
  for p in points:
    minp[0] = min(minp[0], p[0])
    minp[1] = min(minp[1], p[1])
    maxp[0] = max(maxp[0], p[0])
    maxp[1] = max(maxp[1], p[1])
    
  ampl = (maxp[0] - minp[0], maxp[1] - minp[1])
  
  norm_points = []
  
  for p in points:
    norm_points.append(
      ((p[0] - minp[0]) / ampl[0], (p[1] - minp[1]) / ampl[1])
    )
  
  return norm_points


def load_knn(classes_path: string) -> KNeighborsClassifier:
  with open(classes_path) as file:
    contents = file.read()
  
  lines = contents.split('\n')[:-1];
    
  x, y = [], []
  for line in lines:
    vec, _, label = line.rpartition(' ')
    x.append(flatten(ast.literal_eval(vec)))
    y.append(int(label))
  
  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(x, y)
  
  return knn
  

def classify_signal(classifier: KNeighborsClassifier, points: list[(int, int)]):
  norm = normalize_hand(points)
  norm = np.array(flatten(norm))
  
  pred = classifier.predict(norm.reshape(1, -1))
  
  print(pred)
  return pred[0]
