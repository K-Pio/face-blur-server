from flask import Flask, request, jsonify, send_from_directory
import os, json
from datetime import datetime
from kubernetes import client, config
import yaml
from jinja2 import Environment, FileSystemLoader
from face_detector import detectFace

app = Flask(__name__)
DATA_FILE = 'data.json'
DATA_FILE_PATH = 'data/' + DATA_FILE
os.makedirs('data', exist_ok=True)
if not os.path.isfile(DATA_FILE_PATH):
    with open(DATA_FILE_PATH, 'w') as file:
        json.dump([], file)
else:
    print("jason ok")

IMAGES_DIR = 'data/images/'
BLURRED_DIR = 'data/blurred/'
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(BLURRED_DIR, exist_ok=True)

config.load_incluster_config()

with open("./templates/job.j2.yaml") as f:
    job_manifest = yaml.safe_load(f)

def load_data():
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_json_or_error():
    if not request.is_json:
        return None, (jsonify(error="Content-Type must be application/json"), 415)
    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return None, (jsonify(error="Invalid or non-object JSON"), 400)
    return data, None

@app.route('/blur', methods=["POST"])
def post_blur():
    if 'image' not in request.files:
        return jsonify({'error': 'missed image'}), 400
    
    img = request.files['image']
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    filename = f"{timestamp}_{img.filename}"
    path = os.path.join(IMAGES_DIR, filename)
    img.save(path)

    instructions = request.form.get('instructions')
    if instructions:
        instructions = json.loads(instructions)
        try:
            print(f"my instructions:\n{instructions}")
        except:
            print(instructions)
    else:
        instructions = {}

    data = load_data()
    id_local = len(data)+1
    rec = {'id': id_local, 'filename': filename, 'uploaded': timestamp, 'job status' : "created"}
    data.append(rec)
    save_data(data)

    # face detection
    bbox = detectFace(path)

    blurMethod = instructions.get("blurMethod", 1)
    radius = instructions.get("radius", 10)

    env = Environment(loader=FileSystemLoader("./templates"))
    template = env.get_template("job.j2.yaml")
    rendered = template.render({
        "id": timestamp,
        "param1": filename,
        "param2": bbox[0],
        "param3": bbox[1],
        "param4": bbox[2],
        "param5": bbox[3],
        "param6": id_local,
        "param7": blurMethod,
        "param8": radius,
    })
    job_manifest = yaml.safe_load(rendered)

    batch_api = client.BatchV1Api()
    resp = batch_api.create_namespaced_job(
        namespace="default",
        body=job_manifest
    )
    print(f"Job created: {resp.metadata.name}")

    return jsonify(rec), 202

@app.route('/records/<int:rec_id>', methods=['PATCH'])
def patch_record(rec_id: int):
    savedData = load_data()
    newData, error = get_json_or_error()
    if error:
        return error
    
    for rec in savedData:
        if rec["id"] == rec_id:
            selectedRec = rec
    
    print(selectedRec)
    selectedRec.update(newData)
    print(selectedRec)
    save_data(savedData)

    return jsonify(selectedRec), 200

@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/records', methods=['GET'])
def get_records():
    return jsonify(load_data())

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)