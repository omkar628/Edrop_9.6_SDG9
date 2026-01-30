from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model (DO NOT TRAIN HERE)
model = YOLO("runs/detect/train/weights/best.pt")

def calculate_severity(img_path, results):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_area = h * w

    if len(results[0].boxes) == 0:
        return "No pothole detected"

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    area = (x2 - x1) * (y2 - y1)
    ratio = area / img_area

    if ratio < 0.02:
        return "Low"
    elif ratio < 0.07:
        return "Moderate"
    else:
        return "Severe"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)

        results = model(img_path, conf=0.25)
        severity = calculate_severity(img_path, results)

        return render_template(
            "result.html",
            image=file.filename,
            severity=severity
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
