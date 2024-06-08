import requests
import base64

# 读取图像并进行Base64编码
with open('D:\\dataset\\images\\train\\air-hole\\air-hole-1.png', 'rb') as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# 准备请求数据
data = {
    "image": img_base64
}

# 发送POST请求到/detect端点
response = requests.post('http://localhost:5000/detect', json=data)

# 解析响应
result = response.json()
detections = result['detections']
encoded_image = result['image']
print(len(detections))
# 打印检测结果
for detection in detections:
    print(f"Class: {detection['class_name']}, Score: {detection['score']}")
    print(f"Bounding Box: ({detection['x1']}, {detection['y1']}), ({detection['x2']}, {detection['y2']})")


