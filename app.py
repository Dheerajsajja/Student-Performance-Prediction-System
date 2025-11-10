from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from data_analysis import generate_eda_context
application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])




@app.route('/data-analysis', methods=['GET'])
def data_analysis():
    # Path to your dataset
    data_path = '/Users/dheerajsajja/Documents/Student Performance Prediction System/notebook/data/stud.csv'
    
    # Optional query parameters for dynamic exploration
    selected_col = request.args.get('col')          # categorical column for barplot
    target_col = request.args.get('target')         # target column (if you want target-specific plots)

    # Generate full EDA context
    ctx = generate_eda_context(
        csv_path=data_path,
        selected_categorical=selected_col,
        target_col=target_col
    )

    # Pass everything to your template
    return render_template('data_analysis.html', **ctx)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 

