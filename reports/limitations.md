# Project Limitations & Academic Engineering Assumptions

**Course**: Work System Design  
**Project**: AI-Based Factory Layout Optimizer  

---

## 1. Computer Vision & Perception Assumptions

1. **Synthetic-to-Real Domain Gap**:
   - The object detector was trained on synthetic 2D top-down diagrams generated via Pillow/OpenCV.
   - Real-world factory floor photographs contain camera optical distortion (barrel/pincushion), perspective vanishing lines, non-uniform ceiling lighting, oil stains, moving personnel, and crane shadows.
   - Deploying on physical camera streams requires homography calibration with four corner ground control points (GCPs) and real training imagery.

2. **Axis-Aligned Bounding Box Simplification**:
   - Equipment footprints are modeled as axis-aligned rectangles.
   - Angular orientations ($\theta \in [0^\circ, 360^\circ]$) and non-convex footprints (L-shaped assembly cells, U-shaped machining lines) are simplified to minimum bounding boxes.

3. **Bounding Box vs True Working Envelope**:
   - An optical bounding box captures the outer steel frame of a machine. It does not automatically reflect operator standing zones, CNC door swings, maintenance access covers, or chip conveyor clearances unless explicitly calibrated by the engineer.

---

## 2. Operations Research & Optimization Assumptions

1. **Static Deterministic Flows**:
   - The material flow matrix $F = [f_{i,j}]$ assumes steady-state, static move counts per shift.
   - In dynamic job shops, flow frequencies fluctuate based on batch scheduling, machine breakdowns, scrap rates, and seasonal product mixes.

2. **Distance as a Proxy for Logistics Cost**:
   - Minimizing travel distance $d(C_i, C_j)$ assumes transportation cost is linearly proportional to distance.
   - In reality, forklift acceleration/deceleration, blind intersection stops, aisle congestion, and vertical lifting times significantly impact total transport time.

3. **Fire Code & Regulatory Compliance**:
   - While geometric clearance buffers and structural pillar avoidance are strictly validated by the engine, official NFPA (National Fire Protection Association) and OSHA egress standards require formal safety engineering sign-off, illuminated exit path widths, and sprinkler drop clearances.
