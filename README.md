# AIS-Data-Preprocessing-and-Vessel-Trajectory-Analysis

This repository contains a Python-based solution for preprocessing AIS (Automatic Identification System) data and performing vessel trajectory analysis. The project is modular, flexible, and designed to handle large datasets, offering insights into vessel behavior and port congestion patterns. It includes data preprocessing, trajectory analysis, and visualizations like heatmaps and port maps.

###  Data Preprocessing: 
Cleans and prepares the raw AIS data for analysis.
Analysis of Vessel Trajectories: Performs the analysis at four levels, from direct observations to strategic insights.
Additionally, the repository includes solutions for handling large datasets efficiently, optimized using Pandas and other libraries.

### Tools and Libraries Used
Python: Main programming language.
Libraries:
Pandas, NumPy
matplotlib, seaborn, folium, geopandas
tqdm
pyarrow, fastparquet (for large datasets)
scikit-learn (for future modeling)
Key Files and Folder Structure
### Main Files
AIS_Data_Preprocessing_and_Vessel_Trajectory_Analysis.ipynb:
This is the core Jupyter notebook that drives the entire analysis process. It combines data cleaning, utility functions, and visualization generation, guiding the user through each step of the analysis. It is the most important file in the project and integrates the results of all other scripts.

helper_utils.py:
A utility script that contains reusable functions for the analysis. It was provided by the assignment team and extended with additional custom functions.

helper_utils_level_1.py to helper_utils_level_4.py:
These scripts represent different levels of analysis, ensuring the modular structure of the code. Each file corresponds to a specific stage of the analysis, from basic observations to more complex strategic insights.

data_cleaning.py:
A preprocessing script designed to handle data cleaning tasks, including a function for datetime conversion.

Static HTML Files:

ports_map.html: Displays all port locations on a map.
vessels_heatmap.html: A heatmap showcasing the trajectories of all vessels.
Documentation:

Data Scientist Assignment.pdf: Task brief detailing the objectives and requirements.
run.txt: This file is crucial for setting up the environment and running the analysis. It provides step-by-step instructions to ensure all necessary dependencies are installed and ready before executing the analysis.
requirements.txt:
A file listing all the dependencies and packages needed to run the analysis smoothly.

Folder Structure
data/: Contains all the raw and processed data files provided for the assignment (this folder is not uploaded to GitHub for privacy and file size reasons, but it should be re-added to your local environment).
Installation and Setup
## **Important**
**1. Run run.txt First**
**Before running any scripts or Jupyter notebooks, you must first execute the steps outlined in run.txt. This file will ensure that the required environment is activated and that all necessary imports are available for the analysis.**

2. Install Dependencies
Make sure to install the dependencies listed in requirements.txt by running the following command:

bash
Copy code
pip install -r requirements.txt
This will install all the necessary packages to execute the scripts and run the analysis successfully.

3. Activate the Environment
Follow the specific instructions in run.txt to activate the environment and set up any additional configuration required for the analysis.

### How to Use
Start with AIS_Data_Preprocessing_and_Vessel_Trajectory_Analysis.ipynb:
This notebook will guide you through the data preprocessing and vessel trajectory analysis. It is designed to be interactive, allowing you to modify parameters as needed for your analysis.

### Explore the Analysis Levels:
The code is structured into different levels of analysis. You can start with basic observations and move on to more complex insights by modifying the parameters in the helper functions.

### Generate and View Visualizations:
After processing the data, the analysis produces static HTML outputs like port maps and vessel heatmaps.

### Notes
The repository is designed to handle large datasets efficiently, with optimizations using Pandas and libraries like pyarrow and fastparquet for handling large data files.
You can easily extend the functionality by adding new helper functions or modifying existing ones.
### Conclusion
This repository offers a comprehensive, modular approach to AIS data preprocessing and vessel trajectory analysis. By following the steps in the Jupyter notebook, you will gain valuable insights into vessel behaviors and port congestion patterns.
