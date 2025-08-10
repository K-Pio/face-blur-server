import sys
from methods import methods
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import os, requests, json
from datetime import datetime

INPUT_DIR = "./data/images/"
OUTPUT_DIR = "./data/blurred/"

BASE_URL = os.getenv("BASE_URL", "http://blur-svc")

def pretty(resp: requests.Response) -> str:
    try:
        body = json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except ValueError:
        body = resp.text
    return f"Status: {resp.status_code}\nHeaders: {dict(resp.headers)}\nBody:\n{body}\n"

def patch_result(rec_id):
    url = os.getenv("TARGET_URL", f"http://blur-svc/records/{rec_id}")

    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    
    data = {
        "job status": "completed",
        "completed_at": timestamp
    }

    resp = requests.patch(url, json=data, timeout=5)
    print(pretty(resp))
    resp.raise_for_status()
    

def main():
    args = sys.argv

    imageName = args[1]
    imagePath = INPUT_DIR + imageName
    bbox_xyxy = [args[2], args[3], args[4], args[5]]
    id_job = args[6]
    method_no = int(args[7] if len(args) > 6 else 1)
    method_no = method_no if (method_no <= 3 and method_no > 0) else 1
    radius = int(args[8] if len(args) > 7 else 15)
    radius = 15 if radius < 0 or radius > 100 else radius

    bbox_xyxy = [int(float(x)) for x in bbox_xyxy]
    bbox_xywh = bbox_xyxy[0:2] + [bbox_xyxy[2]-bbox_xyxy[0]] + [bbox_xyxy[3]-bbox_xyxy[1]]

    try:
        img = np.array(Image.open(imagePath))
        selectedMethod = methods[method_no]

        print("params:\nfilename : {imagePath}; xywh : {bbox_xywh}; \
            method : {selectedMethod}; radius : {radius} \n".format(
            imagePath=imagePath, bbox_xywh=bbox_xywh, selectedMethod=selectedMethod, radius=radius
        ))
        blurred = selectedMethod(img, 
                                 bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3], 
                                 radius)

        plt.imsave(OUTPUT_DIR + imageName, np.array(blurred))

        print('finished')
        patch_result(int(id_job))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()