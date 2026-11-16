import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import ast
import math

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
NUM_FINGER_LANDMARKS = 21

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


def normalize_hand(points: list[(int, int)]) -> list[(int, int)]:
  xs = [p[0] for p in points]
  ys = [p[1] for p in points]

  min_x, max_x = min(xs), max(xs)
  min_y, max_y = min(ys), max(ys)

  w = max_x - min_x or 1
  h = max_y - min_y or 1

  norm = [((x - min_x) / w, (y - min_y) / h) for x, y in points]
  
  return norm


def make_hand(pts: list[tuple[int, int]]) -> dict:
  avr_x = sum(p[0] for p in pts) / len(pts)
  avr_y = sum(p[1] for p in pts) / len(pts)
  
  return {
    'pts' : np.array(pts),
    'norm' : normalize_hand(pts),
    'avr_pos' : [avr_x, avr_y]
  }


previous_frame = []
current_frame = []


def detect_hands(frame: np.ndarray, palm_detector: MPPalmDet, hand_pose: MPHandPose) -> list[dict]:
  global previous_frame, current_frame
  
  hands = []
  
  palms = palm_detector.infer(frame)
  for palm in palms:
    result = hand_pose.infer(frame, palm)
    if result is None:
      continue
    landmarks = result[4:67].reshape(21, 3)
    pts = [(int(x), int(y)) for x, y, _ in landmarks]
    
    hand = make_hand(pts)
    
    hands.append(hand)
  
  previous_frame = current_frame
  current_frame = hands
  
  # getting speeds for each point
  old_hands_idxs = match_previous_frame_hands(current_frame, previous_frame)
    
  speeds = []
  
  for curr_idx, old_idx in enumerate(old_hands_idxs):
    speed = [[0, 0] for i in range(NUM_FINGER_LANDMARKS)]
    
    if old_idx != -1:
      speed = get_hand_speed(current_frame[curr_idx], previous_frame[old_idx])
    
    speeds.append(speed)
  
  for i, hand in enumerate(current_frame):
    hand['speeds'] = speeds[i]
  
  return current_frame


def get_hand_speed(curr_hand: dict, prev_hand: dict
) -> list[tuple[float, float]]:
  return curr_hand['pts'] - prev_hand['pts']


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
  return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def match_previous_frame_hands(curr_frame, old_frame):
  matches = []
  old_positions = [hand['avr_pos'] for hand in old_frame]
  
  if not old_positions or not curr_frame:
    return [-1] * len(curr_frame)
    
  unmatched_old_indices = list(range(len(old_frame)))
  
  for hand in curr_frame:
    if not unmatched_old_indices:
      matches.append(-1)
      continue
      
    best_idx = min(
      unmatched_old_indices,
      key=lambda idx: distance(hand['avr_pos'], old_positions[idx])
    )
    
    unmatched_old_indices.remove(best_idx)
    matches.append(best_idx)
    
  return matches
  

def draw_hand(frame: np.ndarray, hand: dict):
  pts = hand['pts']
  speeds = hand['speeds']
  
  for finger_idx, chain in enumerate(FINGER_LANDMARK_GROUPS):
    color = FINGER_COLORS[finger_idx]

    for start_idx, end_idx in zip(chain[:-1], chain[1:]):
      cv2.line(frame, pts[start_idx], pts[end_idx], color, 3, cv2.LINE_AA)

    for pt_idx in chain[1:]:
      cv2.circle(frame, pts[pt_idx], 5, color, -1, cv2.LINE_AA)
      cv2.arrowedLine(frame, pts[pt_idx], pts[pt_idx] + speeds[pt_idx], (0, 0, 255), 3)

  mcp_connections = [
      (INDEX_MCP, MIDDLE_MCP),
      (MIDDLE_MCP, RING_MCP),
      (RING_MCP, PINKY_MCP),
  ]
  for i, j in mcp_connections:
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
  

def classify_signal(classifier: KNeighborsClassifier, hand: dict):
  norm = hand['norm']
  
  norm = np.array(flatten(norm))
  
  pred = classifier.predict(norm.reshape(1, -1))
  
  print(pred)
  return pred[0]
