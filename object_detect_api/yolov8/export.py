from ultralytics import YOLO

#print("---> Exporting yolov8n.pt...", flush = True)
#model = YOLO("yolov8n.pt")
#model.export(format = "engine", dynamic = True, batch = 8, workspace = 4, int8 = True, data = "coco.yaml" )

print("---> Exporting yolov8n-seg.pt...", flush = True)
model = YOLO("yolov8n-seg.pt")
model.export(format = "engine", dynamic = True, batch = 8, workspace = 4, int8 = True, data = "coco.yaml" )

#print("---> Exporting yolov8n-pose.pt...", flush = True)
#model = YOLO("yolov8n-pose.pt")
#model.export(format = "engine", dynamic = True, batch = 8, workspace = 4, int8 = True, data = "coco-pose.yaml" )