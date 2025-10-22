# 🚗 Real-Time Smoky Vehicle Detection and License Plate Recognition  

**University:** Department of Electrical Engineering, University of Engineering and Technology (UET) Lahore  
**Supervisor:** Dr. Kashif Javed  
**Project Type:** Final Year Design Project (FYDP)  
**Group No.:** 15  

**Team Members:**  
- **Taha Asif**  — 2021-EE-074 
- **Abad ur Rehman** — 2021-EE-091 
- **Muhammad Bilal** — 2021-EE-068 

---

## 📜 Problem Statement  

Air pollution in cities is rapidly increasing, and one of the major causes is smoke emitted from vehicles.  
Most traffic monitoring systems can track vehicles but **cannot automatically detect smoke emissions** or **identify polluting vehicles** through license plates.  

This gap leaves many polluting vehicles undetected, making manual checks inefficient and impractical for busy roads.  
Hence, there is a need for an **automated system** that can detect, recognize, and record smoke-emitting vehicles.

---

## 💡 Proposed Solution  

The objective of this project is to design an **automated real-time system** that:  
1. Detects vehicles emitting excessive smoke.  
2. Identifies their license plates using OCR technology.  
3. Records smoky vehicle details (license plate, date, time) in a CSV file.  

### **Process Overview**
- Use a camera to monitor vehicle emissions.  
- Detect **vehicles** and **smoke** using trained YOLOv8 models.  
- Detect **license plates** using another YOLOv8 model.  
- Recognize license plate characters using **EasyOCR**.  
- Save the detected smoky vehicle’s plate number in a CSV file for record keeping.

---

## ⚙️ System Architecture  

### **Flow Diagram**
1. Capture live video feed.  
2. Detect smoke and vehicles in real time.  
3. When smoke is detected → trigger license plate recognition.  
4. Save recognized plate data with timestamp.  
5. Display results on screen and update log file.

---

## 🧠 Model Development  

| Task | Model | Description |
|------|--------|-------------|
| Vehicle Detection | YOLOv8 | Trained to detect multiple vehicle types. |
| Smoke Detection | YOLOv8 | Identifies visible smoke emissions from vehicles. |
| License Plate Detection | YOLOv8 | Detects plate regions for OCR input. |
| License Plate Recognition | EasyOCR | Extracts alphanumeric characters from detected plates. |

All three models were trained on custom datasets for optimal detection accuracy.

---

## 🧩 Integration and Output  

- Real-time pipeline combines all models.  
- When smoke is detected, the corresponding vehicle’s license plate is automatically captured.  
- The recognized license plate number, date, and time are **saved in a CSV file**.  
- System can work on **live streams** or **recorded videos**.

---

## 🧰 Hardware Implementation  

As part of Phase II, the trained model was **deployed on an NVIDIA Jetson Nano**, enabling real-time detection on-device without cloud dependency.  
Using a connected live camera feed, the Jetson Nano demonstrated **continuous monitoring** and **efficient processing** suitable for compact surveillance units.

---

## 🌍 Applications  

- **Environmental Monitoring:** Real-time tracking of vehicle emissions to improve air quality.  
- **Traffic Law Enforcement:** Automated reporting of violators to authorities.  
- **Public Health:** Reduction of pollution-related health risks.  
- **Smart City Integration:** Compatible with existing CCTV infrastructure for large-scale deployment.



## 📚 References  

1. [IQAir – 2023 World Air Quality Report](https://www.iqair.com/world-most-polluted-cities)  
2. [The News International – Lahore AQI](https://www.thenews.com.pk/latest/1255848-lahore-records-worlds-worst-air-pollution-level-as-aqi-crosses-600)  
3. [Startup Pakistan – Smog in Pakistan: Impact and Solutions](https://startuppakistan.com.pk/smog-in-pakistan-impact)  
4. [YOLOv4 Paper – J. Redmon & A. Farhadi](https://arxiv.org/abs/2004.10934)  
5. [Tesseract OCR Engine](https://github.com/tesseract-ocr)  
6. [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)  
7. [WHO – Air Quality and Health](https://www.who.int/health-topics/air-pollution)  
8. [Smart Cities World – AI-Driven Environmental Monitoring](https://www.smartcitiesworld.net)  
9. [Iqbal, M. & Siddiqui, S. (2021). AI-based Traffic Surveillance for Monitoring Vehicular Emissions. *Journal of Environmental Engineering.*](https://www.researchgate.net/publication/366644990_Monitoring_Vehicle_Pollution_and_Fuel_Consumption_Based_on_AI_Camera_System_and_Gas_Emission_Estimator_Model)

---

## 🧾 License  
This project is developed as part of the **Final Year Design Project (FYDP)** at **UET Lahore**.  
All rights reserved © 2025 — Group 15.

---

## ✨ Acknowledgements  
Special thanks to **Dr. Kashif Javed** for his supervision and guidance, and to the **Department of Electrical Engineering, UET Lahore** for their support and resources.
