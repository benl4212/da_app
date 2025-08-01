# Simulated LNG Train Control & Monitoring System: 

Tag Database Version: 2.0 

Last Updated: July 30, 2025. 

___

# 1 Project Overview
This document serves as the primary guide to the tag database for the Simulated LNG Train Control & Monitoring System. The database is a core component of the project, designed to provide a structured, realistic, and comprehensive set of process variables for a simulated industrial environment.The tags defined herein are intended for use across multiple project deliverables, including:
1. D1. A Python-based Modbus TCP server and SQLite data historian.
2. D2. A Streamlit-based HMI and data visualization dashboard.
3. D3. A simulated PLC program for process control and alarming.
4. D4. A PyPSA-based power system simulation modeling plant-wide electrical loads.



## ____ 1.1 Simulated Subsystems ____ 

The simulated LNG train is divided into the following subsystems, each with a designated loop number series.

|Series	|Subsystem Name                    	|Function|
|------	    |----------------------------------	|-------------------------------------------------------------------------|
|100   	    |Filtration, Metering, & Regulation	|Pre-treats and measures the incoming natural gas feed.|
|200   	    |Acid Gas Removal                  	|Removes CO2 and sulfur compounds from the natural gas.|
|300   	    |Dehydration                       	|Removes water from the gas stream to prevent freezing.|
|350   	    |C3MR Liquefaction                 	|The main cryogenic process using refrigeration compressors to produce LNG.|
|400   	    |LNG Storage & Loading             	|Stores the final LNG product and manages loading operations.|
|500   	    |Refrigerant Storage & Makeup      	|Stores and supplies the propane and mixed refrigerants for the C3MR process.|
|600   	    |Heat Transfer Fluid System        	|Uses steam to heat various process streams as required.|
|700   	    |Fractionation Unit                	|A distillation column used to separate hydrocarbon components.|
|800   	    |H2 Storage & SMR                  	|Stores hydrogen and includes a simplified SMR unit for H2 production.|
|900   	    |H2 Fuel Stack & Power System      	|Generates electrical power for the plant using a hydrogen fuel cell stack.|



## ____ 1.2 Power System Simulation (PyPSA) ____

- A key objective of this project is to simulate the plant's electrical grid. The tag database includes specific power consumption tags to support this.
- **Power Generation:** The 900-series tags include detailed total power generation metrics (-MW, -MVAR, E-920-V) from the H2 Fuel Stack.
- **Power Consumption:** 100-900 series tags are listed by the major electrical loads across all subsystems. These tags, ending in -KW, represent the power draw of large motors for pumps and compressors. 
- By combining the generation and consumption tags in a PyPSA model, a realistic simulation of the plant's power balance, load shedding scenarios, and grid stability can be achieved.

___

# 2. BPCS Tag Database Schema

All tags are defined using a consistent schema, detailed below.

|Column     	    |Data Type	|Description|
|----------	    |-----------	|-----------------------------------------------------------------------------------------------------------------|
|TagName    	|TEXT     	|The unique identifier for the tag, following ISA-5.1 convention (e.g., `PT-101`, `LAH-401`). This is the Primary Key.|
|Description	    |TEXT     	    |A clear, human-readable description of the tag's function (e.g., "Storage Tank Level Transmitter").|
|Unit       	    |TEXT     	    |The engineering unit of measurement (e.g., `bar`, `degC`, `%`, `kW`). For alarms, this is "Alarm".|
|D_Type     	    |TEXT     	    |The fundamental data type. `REAL` for analog values, `BOOL` for discrete states.|
|LowLim     	    |REAL     	    |Scaling Limit: The 0% value for HMI visualization (gauges, trends).|
|HighLim    	    |REAL     	    |Scaling Limit: The 100% value for HMI visualization.|
|LL_SP      	    |REAL     	    |Setpoint: The Low-Low alarm setpoint. Triggers a critical alarm.|
|L_SP       	    |REAL     	    |Setpoint: The Low alarm setpoint. Triggers an advisory alarm.|
|H_SP       	    |REAL     	    |Setpoint: The High alarm setpoint. Triggers an advisory alarm.|
|HH_SP      	    |REAL     	    |Setpoint: The High-High alarm setpoint. Triggers a critical alarm.|

- **Note** on Implementation: When importing into a database like SQLite, string values such as - or N/A in setpoint columns must be converted to NULL to be stored in REAL type columns.



## ____ 2.1 Tagging Philosophy ____

The tag database is organized according to standard industry practices to ensure clarity and scalability.

- **ISA-5.1 Standard:** Tag names are based on the ANSI/ISA-5.1 standard, combining an instrument code (e.g., PT for Pressure Transmitter) with a unique loop number (e.g., -101).
- **Logical Grouping:** Each major process unit is assigned a unique 100-series number, which forms the basis for all loop numbers within that subsystem. This provides a clear, organized structure.
- **Alarm Handling:** Alarms are treated as separate, discrete BOOL tags (e.g., PAH-109) derived from a parent analog tag (PT-109). This separates the process value from its alarm status, which is critical for clear logic in the PLC and HMI.4. Subsystem Directory

___

# 3. SIS Tag Database Schema

All tags are defined using a consistent schema, detailed below.

|Column          |Data Type           |Description|
|------------	    |--------------	    |-----------------------------------------------------------------------------------------------------|
|TagName             |TEXT                |The unique identifier for the safety tag (e.g., PST-101A, LSHH-401). This is the Primary Key.|
|Description         |TEXT                |A clear, human-readable description of the tag's safety function.|
|Unit                |TEXT                |The engineering unit of measurement. For alarms and trips, this is "Alarm", "Trip", or "E-Stop".|
|D_Type              |TEXT                |The fundamental data type. REAL for analog sensors, BOOL for discrete states.|
|LowLim              |REAL                |Scaling Limit: The 0% value for HMI visualization.|
|HighLim             |REAL                |Scaling Limit: The 100% value for HMI visualization.|
|LL_SP               |REAL                |Setpoint: The Low-Low trip setpoint.|
|L_SP                |REAL                |Setpoint: The Low advisory alarm setpoint (typically not used in SIS).|
|H_SP                |REAL                |Setpoint: The High advisory alarm setpoint (typically not used in SIS).|
|HH_SP               |REAL                |Setpoint: The High-High trip setpoint.|

**Note**: For most SIS functions, only the `LL_SP` or `HH_SP` is relevant for a trip. Other setpoint columns are typically left as `NULL`.



## ____ 3.1 SIS Philosophy ____

The core philosophy of this SIS design is to ensure a high degree of reliability and independence.

* **Independence**: The SIS functions independently of the BPCS. It uses dedicated sensors, logic solvers, and final control elements to prevent a single point of failure from disabling both control and safety functions.
* **Reliability**: To prevent spurious trips and ensure availability, critical safety functions utilize redundant sensors in a **2-out-of-3 (2oo3) voting** architecture. A trip is only initiated if at least two of the three dedicated safety sensors detect a hazardous condition.
* **Fail-Safe Design**: Final control elements, such as emergency shutdown valves (`SXV`), are designed to fail to their safe state (e.g., fail-closed) upon loss of power or signal.



## ____ 3.2 SIS Tagging Convention _____

To clearly differentiate safety tags from BPCS tags, a specific naming convention is used. This is based on the ISA-5.1 standard but with modifications for safety applications.

|Tag Prefix      |Instrument Type         |Description|
|------------	    |----------------------  |----------------------------------------------------------------------------------------------------|
|PST                 |Pressure Safety Tx      |A pressure transmitter dedicated to a safety function.|
|LST                 |Level Safety Tx         |A level transmitter dedicated to a safety function.|
|TST                 |Temperature Safety Tx   |A temperature transmitter dedicated to a safety function.|
|GDT                 |Gas Detector Transmit.  |A sensor for detecting combustible or toxic gas.|
|PSHH, LSHH, etc.    |Voted Logic Switch      |A BOOL tag representing the output of the 2oo3 voting logic. This tag initiates the trip.|
|PDA, LDA, etc.      |Deviation Alarm         |A BOOL alarm that activates if one redundant sensor deviates significantly from the others.|
|SXV                 |Safety Shutdown Valve   |An emergency shutdown (ESD) valve that acts as the final element to stop a hazardous flow.|
|SAH / SAL           |Safety Alarm            |A dedicated alarm that indicates a specific Safety Instrumented Function (SIF) has been activated.|
|EHS                 |Emergency Hand Switch   |A manual pushbutton for initiating a shutdown.|



## ____ 3.3 Key Safety Instrumented Functions (SIFs) ____

The SIS is built around a series of SIFs, each designed to mitigate a specific hazard.

|SIF ID          |Subsystem Series        |Safety Objective|
|------------	    |----------------------  |-------------------------------------------------------------------------|
|SIF-100             |100 - Gas Inlet         |Prevent over-pressurization of the plant from the feed gas inlet.|
|SIF-200             |200 - Acid Gas Removal  |Prevent over-pressurization of the Amine Contactor.|
|SIF-350             |350 - Liquefaction      |Prevent over-pressure at the Mixed Refrigerant compressor suction.|
|SIF-401             |400 - LNG Storage       |Prevent LNG storage tank overfill.|
|SIF-402             |400 - LNG Storage       |Prevent LNG storage tank over-pressurization.|
|SIF-700             |700 - Fractionation     |Prevent over-pressurization of the fractionation column.|
|SIF-800             |800 - H2 Production     |Detect and isolate flammable gas leaks in the hydrogen production area.|
|SIF-900             |900 - H2 Fuel Stack     |Prevent thermal damage to the fuel cell stack from over-temperature.|



## ____ 3.4 Emergency Shutdown (ESD) System ____

In addition to the automated SIFs, a manual Emergency Shutdown (ESD) system is included.

* **E-Stops (`EHS`)**: These are manually activated pushbuttons placed in critical locations (e.g., Main Control Room, process areas).
* **Function**: Activating an E-stop will trigger a pre-defined shutdown sequence, which may include tripping major rotating equipment and closing key ESD valves (`SXV`) to bring the entire plant or a specific area to a safe state.
