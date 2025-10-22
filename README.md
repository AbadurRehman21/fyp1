# 💨 Real-Time Smoky Vehicle Detection and License Plate Recognition

This project is a real-time smart surveillance system designed to automatically detect vehicles emitting excessive smoke and identify them using License Plate Recognition (LPR). [cite_start]Developed as a Final Year Design Project (FYDP) by a team specializing in Computer Engineering [cite: 13, 16, 17][cite_start], this system aims to assist environmental agencies and law enforcement in tackling air pollution caused by high-emission vehicles[cite: 30, 91].

## ⚠️ Problem Statement

[cite_start]Air pollution in cities is increasing, and vehicles that release smoke are a major cause[cite: 28]. [cite_start]Current traffic monitoring systems can track vehicles but generally cannot automatically detect smoke emissions or identify vehicles through their license plates[cite: 29]. [cite_start]This means many polluting vehicles go unnoticed [cite: 30][cite_start], as manual checks are time-consuming and impractical for busy roads[cite: 31]. [cite_start]This highlights the need for an automated approach to accurately identify and track smoke-emitting vehicles[cite: 32].

## ✨ Proposed Solution & Objective

[cite_start]The objective is to design a system to detect vehicles emitting excessive smoke and identify them using license plate recognition[cite: 38].

### Process Flow

1.  [cite_start]**Capture:** A camera captures real-time video of vehicles[cite: 40, 52].
2.  [cite_start]**Detection:** A machine learning model detects smoke emissions and identifies license plates[cite: 41, 52].
3.  [cite_start]**Recognition:** Optical Character Recognition (OCR) technology extracts text from the detected license plates[cite: 42, 52].
4.  [cite_start]**Reporting:** The system saves the license plate of the smoky vehicle to a CSV file for record-keeping [cite: 43] [cite_start]and sends the processed vehicle data to authorities[cite: 52, 53].

## 🏗️ System Architecture and Technology Stack

[cite_start]The system leverages advanced deep learning models and a robust edge computing platform for real-time performance[cite: 130].

### Hardware Platform

* **Edge Device:** **NVIDIA Jetson Nano**
    * [cite_start]The trained model was deployed on the Jetson Nano as part of Phase II[cite: 129].
    * [cite_start]This enabled real-time processing directly on the device, eliminating the need for cloud support[cite: 130].
    * [cite_start]The Jetson Nano performs continuous monitoring and detection using a live camera feed[cite: 131].
* [cite_start]**Input:** Live Camera Feed (Camera captures video)[cite: 40, 52].

### Software & Models

| Component | Technology | Function | Current Progress |
| :--- | :--- | :--- | :--- |
| **Vehicle Detection** | **YOLOv8** | [cite_start]Trained to detect various types of vehicles on the road[cite: 105]. | [cite_start]Completed[cite: 105]. |
| **Smoke Detection** | **YOLOv8** | [cite_start]Trained to detect smoke emissions from vehicles[cite: 106]. | [cite_start]Completed[cite: 106]. |
| **License Plate Detection** | **YOLOv8** | [cite_start]Trained to detect vehicle number plates and draw bounding boxes[cite: 107, 108]. | [cite_start]Completed[cite: 107, 108]. |
| **Character Recognition** | **EasyOCR** | [cite_start]Integrated to extract and recognize alphanumeric characters from detected license plates[cite: 119]. | [cite_start]Completed[cite: 119]. |
| **Reporting System** | **Python / CSV** | [cite_start]Automatically saves the detected license plate along with the date and time to a CSV file[cite: 126]. | [cite_start]Completed[cite: 126]. |

## 🚀 Key Progress

* [cite_start]**Data Collection:** Collected initial datasets for vehicle emissions (smoke) and number plates to train the models[cite: 99, 100].
* [cite_start]**Model Integration:** Combined the models so that when smoke is detected, the system automatically detects and captures the license plate of the vehicle[cite: 123].
* [cite_start]**Prototype Testing:** Conducted initial tests using both models for real-time smoke detection and number plate identification[cite: 120, 121].
* [cite_start]**Real-Time Performance:** The system demonstrated reliable real-time performance on the Jetson Nano, making it suitable for compact and energy-efficient surveillance[cite: 132].

## 🎯 Applications

* [cite_start]**Environmental Monitoring:** Supports real-time tracking of vehicle emissions to improve air quality[cite: 84, 85].
* [cite_start]**Traffic Law Enforcement:** Automates detection and reporting of violators, assisting traffic authorities in enforcing emission regulations[cite: 86, 91].
* [cite_start]**Public Health Improvement:** Reduces pollution-related health risks, improving respiratory health and life expectancy[cite: 92, 93].
* [cite_start]**Smart City Integration:** Integrates seamlessly with existing CCTV infrastructure for city-wide deployment[cite: 94, 95].

## 👨‍💻 Group Members

* [cite_start]**Taha Asif** (Group Leader) - 2021-EE-074 (Computer Specialization) [cite: 12, 13]
* [cite_start]**Abad ur Rehman** - 2021-EE-091 (Computer Specialization) [cite: 14, 15, 16]
* [cite_start]**Muhammad Bilal** - 2021-EE-068 (Computer Specialization) [cite: 17]

[cite_start]**Project Supervisor:** Dr. Kashif Javed [cite: 5]
[cite_start]**Department:** Electrical Engineering, University of Engineering and Technology Lahore [cite: 6]
```eof
