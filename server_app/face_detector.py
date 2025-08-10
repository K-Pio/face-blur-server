from ultralytics import YOLO

MODEL_NAME = "yolov11s-face.pt"
MODEL_PATH = "./models/" + MODEL_NAME

def detectFace(imgPath):
    model = YOLO(MODEL_PATH)
    output = model.predict(imgPath)

    bbox = output[0].boxes.xyxy.tolist()[0]
    return bbox