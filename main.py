import os
import sys
import time

import cv2

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
sys.path.insert(0, MODELS)

from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

PALM_MODEL = os.path.join(MODELS, "palm_detection_mediapipe_2023feb.onnx")
POSE_MODEL = os.path.join(MODELS, "handpose_estimation_mediapipe_2023feb.onnx")

BACKEND = cv2.dnn.DNN_BACKEND_OPENCV
TARGET = cv2.dnn.DNN_TARGET_CPU

from classify_signal import *

def main():
  for p in (PALM_MODEL, POSE_MODEL):
    if not os.path.exists(p) or os.path.getsize(p) < 100_000:
      raise RuntimeError(f"Missing or LFS-pointer model file: {p} (run `git lfs pull` in the zoo repo)")

  palm_detector = MPPalmDet(modelPath=PALM_MODEL, nmsThreshold=0.3, scoreThreshold=0.6,
                            backendId=BACKEND, targetId=TARGET)
  hand_pose = MPHandPose(modelPath=POSE_MODEL, confThreshold=0.9,
                         backendId=BACKEND, targetId=TARGET)

  cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

  prev_tick = cv2.getTickCount()
  
  classifier = load_knn('classes.txt')

  hands_to_write = []
  is_writing = False

  while True:
    ok, frame = cap.read()
    if not ok:
      break

    frame = cv2.flip(frame, 1)
    
    for hand in detect_hands(frame, palm_detector, hand_pose):
      draw_hand(frame, hand)
      classify_signal(classifier, hand)
      if is_writing:
        hands_to_write.append(str(normalize_hand(hand)) + '\n')
    
    curr_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
    prev_tick = curr_tick
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Hand Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
    if key == ord("s"):
      filename = f"frame_{int(time.time() * 1000)}.png"
      cv2.imwrite(filename, frame)
      print(f"Saved {filename}")
    if key == ord('w'):
      is_writing = not is_writing
      if not is_writing:
        with open('output_hands.txt', 'w') as f:
          f.writelines(hands_to_write)
        print(f'Written {len(hands_to_write)} lines to \'output_hands.txt\'')
      else:
        print('Logging hands...')
        hands_to_write = []
      

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
