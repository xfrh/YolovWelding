from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a model
model = YOLO("YOLOv8n.yaml")  # build a new model from scratch

# Train the model
results = model.train(data="config.yaml", epochs=10,batch=4,lr=0.01,resume=True)  # train the model

results.mean_results()  # This will print a summary of the training results


