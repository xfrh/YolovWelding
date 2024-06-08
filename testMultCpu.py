import cv2
import threading
from ultralytics import YOLO
import os
import queue

# 加载模型
model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'last.pt')
model = YOLO(model_path)
class_names = {
    0: "air-hole",
    1: "bite-edge",
    2: "broken-arc",
    3: "crack",
    4: "hollow-bead",
    5: "overlap",
    6: "slag-inclusion",
    7: "unfused"
}
# 创建队列用于线程间通信
frame_queue = queue.Queue(maxsize=10)


def process_frame():
    while True:
        camera_id, frame = frame_queue.get()
        if frame is None:  # 结束信号
            break
        # 使用 YOLO 模型处理帧
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            class_name = class_names.get(class_id, "Unknown")
            label_text = f"Class: {class_name}, Confidence: {score:.2f}"
            cv2.putText(frame, label_text, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow(f'Camera {camera_id}', frame)
        cv2.waitKey(1)
        frame_queue.task_done()


def capture_video(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_id}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put((camera_id, frame))

    cap.release()
    frame_queue.put((camera_id, None))  # 发送结束信号


def get_camera_ids(max_test=10):
    camera_ids = []
    for i in range(max_test):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_ids.append(i)
                cap.release()
            else:
                print(f"Camera {i} not found or not accessible.")
                break
        except Exception as e:
            print(f"Exception occurred while testing camera {i}: {e}")

    return camera_ids


# 动态搜索连接到PC的摄像头数量
camera_ids = get_camera_ids()

if not camera_ids:
    print("No cameras found.")
else:
    # 创建并启动处理线程
    process_thread = threading.Thread(target=process_frame)
    process_thread.start()

    # 创建并启动捕获线程
    capture_threads = []
    for camera_id in camera_ids:
        thread = threading.Thread(target=capture_video, args=(camera_id,))
        capture_threads.append(thread)
        thread.start()

    # 等待所有捕获线程完成
    for thread in capture_threads:
        thread.join()

    # 等待处理线程完成
    frame_queue.join()
    frame_queue.put((None, None))  # 发送结束信号
    process_thread.join()

cv2.destroyAllWindows()
