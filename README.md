# da_app by benl4212 | Project Plan: Simulated LNG Train Control & Monitoring System

**Version 1.3 | Start Date: July 20, 2025 | Last update: July 29, 2025**

Version (1.1) reflects key tooling and workflow adjustments based on initial development and testing, ensuring a more efficient and practical path to completion.

Version (1.2) incorporates a new Instrument Health Monitoring feature, adding a layer of predictive maintenance simulation to the project scope.

This version (1.3) refines the project timeline for a more logical workflow and clarifies the task numbering for better tracking.
# 1 . Project Overview & Key Objectives
This document outlines the plan for the design, development, and implementation of a simulated Liquefied Natural Gas (LNG) train control and monitoring system. My primary objective of this project is to learn and demonstrate a wide range of in-demand skills in industrial automation, control systems, data engineering, and machine learning.
This project is designed to be completed within an aggressive 4-5 week timeline.
## Key Objectives:
* Demonstrate End-to-End System Integration: Integrate edge computing (Raspberry Pi), cloud services (Linode VM, Cloudflare), a web-based UI (Streamlit), and simulated industrial hardware (PLC).
* Showcase Control Systems Knowledge: Design and simulate both Basic Process Control System (BPCS) and Safety Instrumented System (SIS) logic.
* Apply Data & ML in an Industrial Context: Generate realistic sensor data, build a data historian, and apply machine learning for anomaly detection.
* Develop a Professional HMI: Create a user-friendly Human-Machine Interface (HMI) for monitoring and controlling the simulated process.
* Produce Engineering Design Documents: Create standard industry documents, including a P&ID, tag database, and control panel layout.
# 2. Scope & Deliverables
This project encompasses the creation of a multi-component system that simulates the operation and control of an LNG train.
### In Scope:
* All hardware and software setup required for simulation.
* Development of custom scripts for data generation and communication.
* Design of control logic and engineering diagrams using free software.
* End-to-end testing of the integrated system.
### Out of Scope:
* Purchase of any physical industrial hardware (e.g., physical PLCs, sensors).
* Deployment to a production environment.
* Advanced cybersecurity measures beyond the use of Cloudflare Tunnel.
## Deliverables:

* Simulated SCADA/Historian System (on Raspberry Pi 5):
	* A Python-based Modbus TCP server to simulate field device communications.
	* Scripts generating realistic, time-series sensor data (pressure, temp, flow) with trends and anomalies.
	* A lightweight data historian using SQLite for storing time-series data and system tags.

* Enhanced Streamlit Web Application:
  	* Data Dashboard: Visualizing real-time and historical sensor data with interactive plots.
	* ML Dashboard: Displaying anomaly detection results (e.g., F1 scores, PCA, heatmaps).
	* Interactive HMI/Control Page: A graphical overview of the LNG train with on/off controls, real-time sensor readouts, and visual alarms triggered by the ML model.
	* Instrument Health Dashboard: A dedicated page to monitor the health status of all 100+ instruments, with a summary integrated into the main alarm view

* PLC Program Design (Simulated):
	* BPCS Logic: Ladder logic for routine process control and operator alarms.
	* SIS Logic: Ladder logic for critical safety interlocks that act independently to bring the process to a safe state.
	* Implementation will be done in CODESYS

* Power System Design & Simulation:
  	* Conceptual design of a power system for the LNG train utilizing waste heat recovery and hydrogen (H2) feedback for power generation.
	* A basic simulation of the power system dynamics using the PyPSA Python library.

* Engineering Design Documentation:
  	* A Piping & Instrumentation Diagram (P&ID) of the chosen subsystems to model.
	* A structured Tag Database built in SQLite.
	* A Main Control Panel Layout designed in QElectroTech or FreeCAD, including real-world part numbers sourced from online catalogs.
