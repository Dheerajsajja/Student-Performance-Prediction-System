# Student Performance Prediction System

## Overview  
This project is an end-to-end data science and web application pipeline that predicts student academic performance. It covers data ingestion, exploratory data analysis (EDA), model training/prediction, and a web dashboard to explore and visualize the data.

## Table of Contents  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Technologies Used](#technologies-used)  
- [Installation & Usage](#installation-usage)  
- [How to Use](#how-to-use)  
- [Contributing](#contributing)  
- [License](#license)  

## Features  
- Clean and preprocess student performance data.  
- Comprehensive exploratory data analysis, with distributions, correlations, outliers, categorical analysis and automatic textual insights.  
- A prediction pipeline to forecast student outcomes based on input features.  
- Flask web application with interactive pages: prediction page, data analysis dashboard.  
- Modular codebase: notebooks for EDA, Python modules for pipeline, web templates for UI.

## Project Structure  

├── notebook/ # Jupyter notebooks (data exploration and prototyping) 
├── src/ # Source code: pipelines, data classes
├── templates/ # HTML templates for Flask UI
├── data_analysis.py # Python module for EDA context generation
├── app.py # Flask application entry point
├── requirements.txt # Python dependencies
└── README.md # This file


## Technologies Used  
- Python 3.x  
- Pandas, NumPy – data manipulation  
- Matplotlib, Seaborn – visualization  
- Sci-Kit Learn (or custom pipeline) – machine learning  
- Flask – web framework  
- HTML + Bootstrap – front-end UI  
- Git & GitHub – version control & repository hosting  

## Installation & Usage  
1. Clone the repository:  
   ```bash
   git clone https://github.com/Dheerajsajja/Student-Performance-Prediction-System.git  
   cd Student-Performance-Prediction-System  


# Create and activate a virtual environment:

python3 -m venv venv  
source venv/bin/activate  # On Windows: venv\Scripts\activate  


# Install required packages:

pip install -r requirements.txt  


#Run the Flask application:

python app.py  


Navigate to http://localhost:8080/ in your browser. Use the navigation to explore the prediction page and the data-analysis dashboard.

# How to Use

Prediction Page: Input student attributes (gender, ethnicity, parental level of education, lunch status, test preparation, reading & writing scores) and submit to get a predicted outcome.

Data Analysis Dashboard: View dataset summary, distributions, correlations, categorical breakdowns, and automated insights. You can optionally choose a categorical column to filter/view.

Use the insights and visuals to understand the data characteristics, potential feature engineering ideas, and model implications.

# Contributing

Contributions are welcome! Feel free to open issues or pull requests for bug fixes, new features, or improvements in EDA, UI, and model performance.
