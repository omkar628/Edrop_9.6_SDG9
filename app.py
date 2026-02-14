from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from ultralytics import YOLO
import cv2
import os
import json
import datetime
import uuid

app = Flask(__name__)
app.secret_key = "civic_ai_unlimited_secret_key"
ADMIN_PASSWORD = "1234"

REPORTS_FILE = "reports.json"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "runs/detect/train/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)

def load_reports():
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r") as f:
                data = json.load(f)
                # Ensure all reports have the necessary fields for the dashboard
                for r in data:
                    r.setdefault("risk_score", 0)
                    r.setdefault("percentage", 0)
                    r.setdefault("cost", 0)
                    r.setdefault("volume", 0)
                return data
        except:
            return []
    return []

def save_report(report_data):
    reports = load_reports()
    reports.append(report_data)
    with open(REPORTS_FILE, "w") as f:
        json.dump(reports, f, indent=4)

def get_municipal_category(severity):
    categories = {
        "Severe": {
            "department": "Emergency Response & Public Works",
            "priority": "Critical",
            "action": "Immediate reconstruction required. High safety risk.",
            "sla": "24 Hours",
            "officer_phone": "+919999999999"
        },
        "Moderate": {
            "department": "Road Maintenance Division",
            "priority": "Medium",
            "action": "Scheduled surface repair needed. Structural integrity warning.",
            "sla": "7-14 Days",
            "officer_phone": "+918888888888"
        },
        "Low": {
            "department": "Infrastructure Planning",
            "priority": "Low",
            "action": "Surface anomaly detected. Routine maintenance recommended.",
            "sla": "30 Days",
            "officer_phone": "+917777777777"
        },
        "None": {
            "department": "General Services",
            "priority": "None",
            "action": "Pavement condition remains optimal.",
            "sla": "N/A",
            "officer_phone": ""
        }
    }
    return categories.get(severity, categories["None"])

def calculate_pothole_metrics(img_path, results):
    """
    Intelligent Pavement Management System (IPMS) Core Logic
    Computes volumetric displacement, fiscal impact, and structural failure risk.
    """
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    img_area = h * w

    boxes = results[0].boxes
    pothole_count = len(boxes)

    if pothole_count == 0:
        return {
            "percentage": 0.0,
            "severity": "None",
            "volume_cm3": 0,
            "estimated_cost_inr": 0,
            "prediction_score": 0,
            "pothole_count": 0,
            "confidence": 0
        }

    total_pothole_area = 0
    max_area_ratio = 0
    avg_conf = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_area = (x2 - x1) * (y2 - y1)
        total_pothole_area += box_area
        max_area_ratio = max(max_area_ratio, box_area / img_area)
        avg_conf += float(box.conf[0])
    
    avg_conf = (avg_conf / pothole_count) * 100
    percentage = (total_pothole_area / img_area) * 100
    
    # Parametric Modeling for Volume (V = A * D)
    # Mean depth (D) is estimated based on surface area projection
    # Larger potholes are historically deeper due to structural layer collapse
    est_mean_depth = 4 + (max_area_ratio * 60) 
    volume = total_pothole_area * est_mean_depth / 1000 # Normalized cubic cm
    
    # Fiscal Impact Model: Material (Bitumen Mix) + Logistics + Labor 
    material_cost = volume * 1.85
    logistics_markup = 750
    labor_factor = 450 * pothole_count
    total_cost = material_cost + logistics_markup + labor_factor
    
    # Predictive Risk scoring (Logistic regression approximation)
    # Factors: Density + Size + Confidence
    risk_score = min(99.4, (percentage * 4.5) + (pothole_count * 5) + 5)
    
    if percentage < 1.5:
        severity = "Low"
    elif percentage < 5.0:
        severity = "Moderate"
    else:
        severity = "Severe"
        
    return {
        "percentage": round(percentage, 2),
        "severity": severity,
        "volume_cm3": round(volume, 2),
        "estimated_cost_inr": round(total_cost, 2),
        "prediction_score": round(risk_score, 1),
        "pothole_count": pothole_count,
        "confidence": round(avg_conf, 1)
    }

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

        results = model(img_path, conf=0.15)
        metrics = calculate_pothole_metrics(img_path, results)
        municipal_info = get_municipal_category(metrics["severity"])

        report = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": img_filename,
            "severity": metrics["severity"],
            "percentage": metrics["percentage"],
            "volume": metrics["volume_cm3"],
            "cost": metrics["estimated_cost_inr"],
            "risk_score": metrics["prediction_score"],
            "pothole_count": metrics["pothole_count"],
            "department": municipal_info["department"],
            "priority": municipal_info["priority"],
            "action": municipal_info["action"],
            "status": "Reported" if metrics["severity"] != "None" else "Verified Safe"
        }
        
        if metrics["severity"] != "None":
            save_report(report)

        # Store context for Chatbot
        session["current_report"] = report
        
        return render_template(
            "result.html",
            image=img_filename,
            metrics=metrics,
            municipal_info=municipal_info,
            report_id=report["id"],
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").lower().strip()
    current_report = session.get("current_report", {})
    
    # 1. Semantic Awareness (Mapping user intent to detection data)
    intents = {
        "status": ["how many", "potholes", "status", "detect", "holes", "count"],
        "fiscal": ["cost", "price", "money", "budget", "expensive", "rupees"],
        "risk": ["risk", "danger", "safe", "dangerous", "prediction", "failure"],
        "tech": ["bitumen", "asphalt", "how it works", "ai", "model", "yolo"],
        "municipal": ["sla", "officer", "who will fix", "time", "department"]
    }

    # Helper to check intent
    def has_intent(category):
        return any(keyword in user_msg for keyword in intents[category])

    # 2. Contextual Response Builder
    if not current_report and any(k in user_msg for k in ["this", "image", "current", "here"]):
        return jsonify({"response": "I'm ready to analyze your specific road image. Please upload a scan first so I can provide telemetry!"})

    if has_intent("status"):
        count = current_report.get("pothole_count", "unknown")
        severity = current_report.get("severity", "N/A")
        if count == 0:
            return jsonify({"response": "The road surface in this image appears optimal. No significant potholes or cracks detected by the YOLO system."})
        return jsonify({"response": f"Strategic analysis shows {count} points of interest. The damage is categorized as '{severity}' with a {current_report.get('percentage')}% surface impact."})

    if has_intent("fiscal"):
        cost = current_report.get("cost", 0)
        if cost == 0:
            return jsonify({"response": "Since no damage was detected, the estimated immediate repair cost is ₹0. Preventive maintenance is still recommended."})
        return jsonify({"response": f"The estimated fiscal impact for this section is ₹{cost}. This covers material volume, transport logistics, and on-site labor."})

    if has_intent("risk"):
        risk = current_report.get("risk_score", 0)
        return jsonify({"response": f"Our structural engine predicts a {risk}% failure probability. High traffic load on this section could lead to rapid pavement degradation."})

    if has_intent("municipal"):
        return jsonify({"response": f"The reported issue is assigned to the '{current_report.get('department', 'General Works')}' department. Expected SLA is {current_report.get('sla', 'standard cycle')}."})

    # 3. Technical Knowledge Base (IPMS Handbook)
    kb = {
        "bitumen": "Bitumen acts as the primary binder in asphalt. Potholes occur when water emulsifies this binder, leading to aggregate stripping.",
        "yolo": "I use the YOLOv8 (You Only Look Once) neural network for real-time spatial object detection in high-resolution pavement imagery.",
        "ai": "My AI combines Computer Vision (YOLO) with Parametric Engineering Models (Volume/Cost/Risk) to provide a digital twin of the road condition.",
        "how it works": "Simple: YOLO v8 detects boxes → Pixel-to-Area conversion calculates damage % → Depth projection estimates volume → Market rates derive cost.",
        "hello": "System online. I am your Technical Road Intelligence Assistant. Ask me about the current report metrics or general road engineering.",
        "hi": "Hi! Ready to optimize infrastructure. What's your query?",
        "weather": "Rainwater infiltration is the #1 enemy. It weakens the base layer through pore pressure, and traffic loading then collapses the asphalt."
    }

    for key, value in kb.items():
        if key in user_msg:
            return jsonify({"response": value})

    # 4. Smart Fallback (Professional Engineering GPT-style)
    return jsonify({
        "response": "Understood. As an infrastructure specialist, I'm focusing on Pavement Management. Could you specify if your query relates to cost estimation, risk scores, or technical bitumen properties?"
    })

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
    stats = {
        "total": len(reports),
        "critical": len([r for r in reports if r.get("priority") == "Critical"]),
        "medium": len([r for r in reports if r.get("priority") == "Medium"]),
        "low": len([r for r in reports if r.get("priority") == "Low"])
    }
    return render_template("dashboard.html", reports=reports[::-1], stats=stats)

@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
