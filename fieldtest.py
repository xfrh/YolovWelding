from ultralytics import YOLO
import cv2

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
model = YOLO("runs/detect/train2/weights/best.pt")
results = model("D:\\dataset\\images\\train\\air-hole\\air-hole-1.png")[0]
frame = cv2.imread("D:\\dataset\\images\\train\\air-hole\\air-hole-1.png")
print(len(results.boxes.data))
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    class_name = class_names.get(class_id, "Unknown")
    print(class_name)
    # label_text = f"Class: {class_name}, Confidence: {score:.2f}"
    # cv2.putText(frame, label_text, (int(x1), int(y1 - 10)),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
cv2.imshow('detected', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



