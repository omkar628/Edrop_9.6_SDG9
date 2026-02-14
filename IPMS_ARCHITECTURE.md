# Civic AI 2.0: Intelligent Pavement Management System (IPMS)
## System Architecture & Strategic Roadmap

### 1. High-Level Architecture
The system follows a **Cloud-Edge Hybrid Architecture** to ensure real-time detection on-field while maintaining heavy analytical processing in the cloud.

*   **Edge Layer:** Mobile devices, IoT road sensors, and Drones (streaming via RTSP).
*   **Perception Layer:** YOLOv8 (Detection) + MiDaS (Monocular Depth Estimation).
*   **Intelligence Layer:** Predictive models for degradation based on climate and traffic throughput.
*   **Presentation Layer:** GIS-Integrated Dashboard with Contractor Analytics.

---

### 2. Model Pipeline Diagram
`Input Image` → `YOLOv8 (Pothole/Crack Detection)` → `Bounding Box Coordinates`
                                ↓
`Input Image` → `MiDaS (Depth Projection)` → `Depth Map / Surface Normal`
                                ↓
`Fusion Layer` → `Volumetric Calculation (V = Area * Mean Depth)`
                                ↓
`Decision Engine` → `SLA Assignment` + `Cost Estimation` → `GIS Database`

---

### 3. Advanced Modules Implementation Plan

| Feature | Technical Approach | Dependency |
|---------|-------------------|------------|
| **Volumetric Analysis** | Calculate volume using depth estimation integration. | `torch`, `MiDaS` |
| **Predictive Maintenance** | Gradient Boosting model (XGBoost) using Rainfall (mm) and Traffic (PCU) inputs. | `scikit-learn` |
| **GIS Ward Heatmaps** | Leaflet.js with coordinate cluster mapping. | `GeoJSON` |
| **Contractor Analytics** | Performance scoring based on Repair Time vs SLA. | `SQL/JSON Analytics` |
| **Cost Estimation** | Parametric modeling ($ per cubic meter of Bitumen mix). | `Dynamic Pricing API` |

---

### 4. Technical Evaluation Metrics
*   **Detection:** IoU (Intersection over Union) > 0.75, mAP (mean Average Precision) > 0.85.
*   **Volumetric:** RMSE (Root Mean Square Error) for depth prediction.
*   **System:** Latency < 200ms for edge detection.

---

### 5. Deployment Strategy
1.  **Phase 1 (Edge):** Mobile-web app for field workers.
2.  **Phase 2 (GIS):** Centralized ward-wise transparency portal.
3.  **Phase 3 (Predictive):** Automated maintenance scheduling for the upcoming monsoon season.
