from flask import Flask, render_template, request, redirect, url_for, session, flash
from ultralytics import YOLO
import cv2
import os
import json
import datetime
import uuid

app = Flask(__name__)
app.secret_key = "civic_ai_secret_key"
ADMIN_PASSWORD = "1234"

REPORTS_FILE = "reports.json"


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "runs/detect/train/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    # Fallback to base model if best.pt is not found
    MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)

def load_reports():
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_report(report_data):
    reports = load_reports()
    reports.append(report_data)
    with open(REPORTS_FILE, "w") as f:
        json.dump(reports, f, indent=4)

def get_municipal_category(severity):
    if severity == "Severe":
        return {
            "department": "Emergency Response & Public Works",
            "priority": "Critical",
            "action": "Immediate repair needed. Dangerous for vehicles.",
            "simple_msg": "बहुत खराब सड़क - तुरंत मरम्मत की जरूरत है (Very bad road - needs immediate repair)",
            "sla": "24 Hours",
            "officer_phone": "+919999999999" # Placeholder for municipal officer
        }
    elif severity == "Moderate":
        return {
            "department": "Road Maintenance Division",
            "priority": "Medium",
            "action": "Schedule for repair soon. Drive carefully.",
            "simple_msg": "सड़क खराब है - जल्द ही ठीक किया जाएगा (Road is bad - will be fixed soon)",
            "sla": "7-14 Days",
            "officer_phone": "+918888888888"
        }
    elif severity == "Low":
        return {
            "department": "Infrastructure Planning",
            "priority": "Low",
            "action": "Minor issue. Will be addressed in routine work.",
            "simple_msg": "छोटी समस्या - समय पर ठीक किया जाएगा (Small issue - will be fixed in time)",
            "sla": "30 Days",
            "officer_phone": "+917777777777"
        }
    else:
        return {
            "department": "General Services",
            "priority": "None",
            "action": "Road surface looks safe.",
            "simple_msg": "सड़क सुरक्षित है (Road is safe)",
            "sla": "N/A",
            "officer_phone": ""
        }

def calculate_pothole_metrics(img_path, results):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_area = h * w

    if len(results[0].boxes) == 0:
        return 0.0, "None"

    total_pothole_area = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        total_pothole_area += (x2 - x1) * (y2 - y1)
    
    percentage = (total_pothole_area / img_area) * 100
    
    if percentage < 2.0:
        severity = "Low"
    elif percentage < 7.0:
        severity = "Moderate"
    else:
        severity = "Severe"
        
    return round(percentage, 2), severity

def calculate_severity(img_path, results):
    # Backward compatibility wrapper if still used
    p, s = calculate_pothole_metrics(img_path, results)
    return s

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        img_filename = f"{uuid.uuid4()}_{file.filename}"
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], img_filename)
        file.save(img_path)

        results = model(img_path, conf=0.25)
        pothole_percent, severity = calculate_pothole_metrics(img_path, results)
        
        if len(results[0].boxes) == 0:
            severity = "No pothole detected"
            
        municipal_info = get_municipal_category(severity)

        # Create report record
        report = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": img_filename,
            "severity": severity,
            "percentage": pothole_percent,
            "department": municipal_info["department"],
            "priority": municipal_info["priority"],
            "action": municipal_info["action"],
            "status": "Reported" if severity != "No pothole detected" else "No Action"
        }
        
        if severity != "No pothole detected":
            save_report(report)

        return render_template(
            "result.html",
            image=img_filename,
            severity=severity,
            percentage=pothole_percent,
            municipal_info=municipal_info,
            report_id=report["id"]
        )

    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").lower()
    
    # Simple AI Logic for technical doubts
    responses = {
        "bitumen": "Bitumen is the black sticky substance (tar) used to bind stones together to make roads.",
        "pothole": "A pothole is a hole in a road surface that results from the gradual wearing away of the pavement.",
        "repair": "Roads are typically repaired using 'Hot Mix' or 'Cold Mix' asphalt depending on the weather conditions.",
        "sla": "SLA (Service Level Agreement) is the time within which the municipality promises to fix the issue.",
        "severity": "Severity is calculated based on the percentage of the road surface area damaged by the pothole.",
        "pavement": "Pavement refers to the hard surface of the road, usually made of concrete or asphalt.",
        "water": "Water is the biggest enemy of roads. It seeps into cracks and expands, causing the surface to break.",
        "priority": "Priority is decided by the size of the pothole and the danger it poses to high-speed traffic."
    }
    
    for key in responses:
        if key in user_msg:
            return {"response": responses[key]}
            
    return {"response": "I'm your Civic AI Assistant. You can ask me about terms like Bitumen, SLA, Severity, or how road repairs work!"}

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password")
        if password == ADMIN_PASSWORD:
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Incorrect Password")
    return render_template("login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))
    
    reports = load_reports()
    # Unique stats
    stats = {
        "total": len(reports),
        "critical": len([r for r in reports if r["priority"] == "Critical"]),
        "medium": len([r for r in reports if r["priority"] == "Medium"]),
        "low": len([r for r in reports if r["priority"] == "Low"])
    }
    return render_template("dashboard.html", reports=reports[::-1], stats=stats)

@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
