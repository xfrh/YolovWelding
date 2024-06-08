from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from database import Database

import io

app = Flask(__name__)

# 定义类别名称
class_names = {
    0: "air-hole",
    1: "bite-edge",
    2: "broken-arc",
    3: "hollow",
    4: "through",
    5: "tumor",
    6: "slag-inclusion",
    7: "unfused"
}

Database.initialize()
# 加载YOLO模型
model = YOLO("runs/detect/train2/weights/best.pt")

def readb64(base64_string):
    decoded_data = base64.b64decode(base64_string)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    return img

@app.route('/addSample', methods=['POST'])
def addSample():
    data = request.get_json()
    img_base64 = data['image']
    img_class = data['class']
    item_id = Database.insert("sample", {"image": img_base64, "class": img_class})
    if item_id:
        return jsonify({"id": str(item_id.inserted_id), "status_code": "200"})
    else:
        return jsonify({"error": "数据更新失败", "status-code": "400"}), 400

@app.route('/samples', methods=['GET'])
def read_items():
    items = []
    results = Database.find("sample", {})
    for result in results:
        items.append({"image": result['image'], "class": result['class']})
    return jsonify(items)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    img_base64 = data['image']
    img = readb64(img_base64)

    # 检测图像中的对象
    results = model(img)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        class_name = class_names.get(int(class_id), "Unknown")
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(img, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        detections.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "score": float(score),
            "class_id": int(class_id),
            "class_name": class_name
        })

    # 将带有标记的图像转换为Base64编码
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # 返回JSON对象
    response = {
        "detections": detections,
        "image": encoded_image
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

