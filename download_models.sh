mkdir -p models && cd models

HF=https://huggingface.co/opencv/opencv_zoo/resolve/main/models
GH=https://raw.githubusercontent.com/opencv/opencv_zoo/main/models

curl -L -O $HF/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx
curl -L -O $HF/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx
curl -L -O $GH/palm_detection_mediapipe/mp_palmdet.py
curl -L -O $GH/handpose_estimation_mediapipe/mp_handpose.py

ls -lh
