# Stirling Engine Model (Beta Type)

This repository contains the source code for an interactive web application simulating the thermodynamic cycle of a Beta-type Stirling engine. The app is built in Python using the Streamlit framework and performs calculations considering polytropic processes on both the hot and cold sides of the engine.

Live version of the application is available here: https://stirling-engine-model.streamlit.app/

## App Features
* Simulation of the ideal cycle and plotting of p-V diagrams.
* Energy balance calculation (indicated power, heat added and rejected, efficiency, regenerated heat).
* Sensitivity analysis – evaluating how changing a single parameter (e.g., mean pressure or dead volumes) affects overall power and efficiency.
* Estimation of actual real-world power using the Beale number, based on G. Walker's empirical data.
* Live animation of piston kinematics based on specified geometry and phase angle.

## Technologies Used
The script utilizes the following Python libraries:
* streamlit
* numpy
* scipy
* pandas
* plotly
* matplotlib

## Local Setup
If you wish to download and run the application locally on your PC, use the following commands:

1. Clone the repository:
git clone https://github.com/vovota2/Stirling-model.git
cd Stirling-model

2. Install the required packages:
pip install -r requirements.txt

3. Run the application:
streamlit run BETA_1.py

## References
Data for the power estimation and empirical curves of the Beale number are based on the following publication:
* MARTINI, William. Stirling engine design manual, 2004. Reprint of the 1983 edition. Honolulu: University press of the Pacific, ISBN: 1-4102-1604-7.

## License
This project is released under the free GNU GPLv3 license. You may freely use and modify it, but any derivative works must be released under the same license. For full details, see the LICENSE file.
