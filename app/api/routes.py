import pickle
import os
import pandas as pd
import numpy as np
from flask import Flask, Blueprint, request, flash, redirect, url_for, render_template, send_file, session
from flask_login import login_required, current_user
from app import db
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
CLEANED_DATA_FOLDER = 'cleaned_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLEANED_DATA_FOLDER'] = CLEANED_DATA_FOLDER

api = Blueprint('api', __name__)

@api.route('/')
@login_required
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return render_template('index.html', title='Dashboard', username=current_user.username)

@api.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if not current_user.is_authenticated:
        flash("Please log in to upload a file.", "error")
        return redirect(url_for('auth.login'))
    
    if request.method == 'GET':
        return render_template('upload_file.html', show_preprocessing_options=False)
    
    if request.method == 'POST':
        step = request.form.get('step')

        if step == 'preprocessing':
            file_path = session.get('uploaded_file_path')
            filename = session.get('uploaded_filename')
            if not file_path or not os.path.exists(file_path):
                flash("Session expired or file missing. Please re-upload.", "error")
                return redirect(url_for('api.upload_file'))

            data = pd.read_csv(file_path)

            if 'drop_missing' in request.form:
                data.dropna(inplace=True)

            if 'apply_scaling' in request.form:
                scaler = StandardScaler()
                numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
                data[numerical_cols] = scaler.fit_transform(data[numerical_cols]) 

            cleaned_data_folder = app.config['CLEANED_DATA_FOLDER']
            os.makedirs(cleaned_data_folder, exist_ok=True)
            # if not os.path.exists(cleaned_data_folder):
            #     os.makedirs(cleaned_data_folder)

            cleaned_file_path = os.path.join(cleaned_data_folder, 'cleaned_' + filename)
            # output = BytesIO()
            data.to_csv(cleaned_file_path, index=False)
            # output.seek(0)

            session['cleaned_data_path'] = cleaned_file_path

            # send_file(output, mimetype="text/csv", download_name="cleaned_data.csv", as_attachment=True)

            # return redirect(url_for('api.model_training'))
            return redirect(url_for('api.visualization'))

        else:
            if 'file' not in request.files:
                flash("No file uploaded. Please upload a CSV file.", "error")
                return redirect(url_for('api.upload_file'))
            
            file = request.files['file']
            if file.filename == '':
                flash("No selected file", "error")
                return redirect(url_for('api.upload_file'))

            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)

            # if not os.path.exists(upload_folder):
            #     os.makedirs(upload_folder)
            
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            try:
                data = pd.read_csv(file_path)

                if data.empty or len(data.columns) == 0:
                    raise ValueError("The uploaded file has no valid columns to parse.")

                
                missing_columns = data.columns[data.isnull().any()].to_list()
                num_missing_rows = data.isnull().any(axis=1).sum()
                numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
                has_numerical_cols = len(numerical_cols) > 0

                session['uploaded_file_path'] = file_path
                session['uploaded_filename'] = file.filename

                message_parts = []

                if missing_columns:
                    message_parts.append(
                        f"Missing data detected in columns: {', '.join(missing_columns)} "
                        f"({num_missing_rows} row(s) affected)."
                    )

                if has_numerical_cols:
                    message_parts.append("Numerical columns detected, scaling option is available.")
                
                if message_parts:
                    flash(message_parts, "warning")

                # missing_msg = {
                #     f"Missing data detected in columns: {', '.join(missing_columns)}. ",
                #     f"Total rows with missing values: {num_missing_rows}."
                # }
                # flash(missing_msg, "warning")
                if missing_columns or has_numerical_cols:
                    return render_template(
                        'upload_file.html',
                        show_preprocessing_options=True,
                        missing_columns=missing_columns,
                        num_missing_rows=num_missing_rows,
                        has_numerical_cols=has_numerical_cols

                    )

                # if 'drop_missing' in request.form:
                #     data = data.dropna(inplace=True)

                # if 'apply_scaling' in request.form:
                #     scaler = StandardScaler()
                #     numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
                #     data[numerical_cols] = scaler.fit_transform(data[numerical_cols]) 

                # cleaned_data_folder = app.config['CLEANED_DATA_FOLDER']
                # if not os.path.exists(cleaned_data_folder):
                #     os.makedirs(cleaned_data_folder)

                # cleaned_file_path = os.path.join(cleaned_data_folder, 'cleaned_' + file.filename)
                # output = BytesIO()
                # data.to_csv(output, index=False)
                # output.seek(0)

                # session['cleaned_data_path'] = cleaned_file_path

                # send_file(output, mimetype="text/csv", download_name="cleaned_data.csv", as_attachment=True)

                # flash("File processed successfully.", "success")
                # send_file(output, mimetype="text/csv", attachment_filename="cleaned_data.csv", as_attachment=True)
                flash("No missing data found. File is ready to be processed.", "success")
                # return redirect(url_for('api.model_training'))
                return redirect(url_for('api.visualization'))
            except Exception as e:
                flash(f"File processing failed: {str(e)}", "error")
                return redirect(url_for('api.upload_file'))
        
    
@api.route('/model-training', methods=['GET', 'POST'])
def model_training():

    if request.method == 'GET':
        return render_template('model_training.html')
    
    cleaned_data_path = session.get('cleaned_data_path')

    if not cleaned_data_path:
        return redirect(url_for('api.upload_file'))
    
    df = pd.read_csv(cleaned_data_path)

    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if request.method == 'POST':
        model_type = request.form.get('model_type')
        best_model = None
        metrics = {}
        regression_metrics = {}
        inertia = None

        if model_type == 'regression':
            model_name = request.form.get('regression_model')
            params = {}

            if model_name == 'linear':
                model = LinearRegression()
                best_model = model.fit(X, y)
            elif model_name == 'ridge':
                model = Ridge()

                try:
                    # Hyperparameters for Ridge Regression
                    alpha_range = [float(x) for x in request.form.get('alpha_range').split(',')]
                    params['alpha'] = alpha_range
                except (ValueError, AttributeError):
                    flash('Invalid alpha range provided', 'error')
                    return redirect(url_for('api.model_training'))

            if params:
                grid_search = GridSearchCV(model, param_grid=params, cv=5)
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
            else:
                best_model = model.fit(X, y)
            
            if best_model:
                y_pred = best_model.predict(X)
                regression_metrics['r2_score'] = r2_score(y, y_pred)
                regression_metrics['mse'] = mean_squared_error(y, y_pred)
                regression_metrics['mae'] = mean_absolute_error(y, y_pred)
                regression_metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
                

                
        elif model_type == 'classification':
            model_name = request.form.get('classification_model')
            params = {}

            if model_name == 'logistic':
                model = LogisticRegression()
                try:
                    C_range = [float(x) for x in request.form.get('C_range').split(',')]
                    params['C'] = C_range
                except (ValueError, AttributeError):
                    flash('Invalid C range provided', 'error')
                    return redirect(url_for('api.model_training'))

            elif model_name == 'svc':
                model = SVC()
                try:
                    svc_C_range = [float(x) for x in request.form.get('svc_C_range').split(',')]
                    kernel = request.form.get('svc_kernel')
                    params['C'] = svc_C_range
                    params['kernel'] = [kernel]
                except (ValueError, AttributeError):
                    flash('Invalid SVC parameters provided', 'error')
                    return redirect(url_for('api.model_training'))
            
            if params:
                grid_search = GridSearchCV(model, param_grid=params, cv=5)
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
            
            if best_model:
                y_pred = best_model.predict(X)
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['precision'] = precision_score(y, y_pred, average='weighted')
                metrics['recall'] = recall_score(y, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
            
        
        elif model_type == 'clustering':
            model_name = request.form.get('clustering_model')

            if model_name == 'kmeans':
                try:
                    n_clusters = [int(x) for x in request.form.get('kmeans_clusters ').split(',')]
                    model = KMeans()
                    params = {'n_clusters': n_clusters}
                    grid_search = GridSearchCV(model, param_grid=params, cv=5)
                    grid_search.fit(X)
                    best_model = grid_search.best_estimator_

                    inertia = best_model.inertia_
                except (ValueError, AttributeError):
                    flash('Invalid KMeans parameters provided', 'error')
                    return redirect(url_for('api.model_training'))
            
            elif model_name == 'hierarchical':
                try:
                    n_clusters = [int(x) for x in request.form.get('hierarchical_clusters').split(',')]
                    linkage = request.form.get('hierarchical_linkage')  # linkage: ward, complete, average, single
                    model = AgglomerativeClustering()
                    params = {'n_clusters': n_clusters, 'linkage': [linkage]}

                    for cluster_count in n_clusters:
                        model = AgglomerativeClustering(n_clusters=cluster_count, linkage=linkage)
                        model.fit(X)
                        best_model = model

                except (ValueError, AttributeError):
                    flash('Invalid parameters for Hierarchical Clustering', 'error')
                    return redirect(url_for('api.model_training'))


        
        if not best_model:
            flash('Model training failed or no model was selected', 'error')
            return redirect(url_for('api.model_training'))
        

        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'trained_model.pkl')
        with open(model_path, 'wb') as model_file:
            pickle.dump(best_model, model_file)
        
        return render_template(
            'model_results.html', 
            model=best_model, 
            regression_metrics=regression_metrics, 
            metrics=metrics, 
            inertia=inertia
        )
    

@api.route('/predict', methods=['GET', 'POST'])
def predict():
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'trained_model.pkl')

    if not os.path.exists(model_path):
        flash("Model not found. Please train the model first", "error")
        return redirect(url_for('api.model_training'))
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    input_data = request.get_json()

    if not input_data:
        flash("No input data provided.", "error")
        return redirect(url_for('api.predict'))
    
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)

        flash(f"Prediction result: {prediction.tolist()}", "success")
        return redirect(url_for('api.model_training'))

    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('api.'))


@api.route('/visualization', methods=['GET', 'POST'])
def visualization():

    message_parts = []
    
    cleaned_file_path = session.get('cleaned_data_path')
    if not cleaned_file_path or not os.path.exists(cleaned_file_path):
        message_parts.append("Cleaned data not found. Please upload and preprocess a file first.")
        flash(message_parts, "error")
        return redirect(url_for('api.upload_file'))
    
    data = pd.read_csv(cleaned_file_path)
    column_names = data.columns.to_list()
    # pie_data = None
    selected_column = None
    chart_type = None
    chart_data = {}

    if request.method == 'POST':
        selected_column = request.form.get('selected_column')

        if selected_column and selected_column in data.columns:
            unique_vals = data[selected_column].nunique()
            dtype = data[selected_column].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                chart_type = 'bar'
                counts, bins = np.histogram(data[selected_column].dropna(), bins=10)
                
                chart_data = {
                    'labels': [f'{round(bins[i], 2)} - {round(bins[i+1], 2)}' for i in range(len(counts))],
                    'values': counts.tolist()
                }
            # pie_data = data[selected_column].value_counts().to_dict()
            elif unique_vals <= 20:
                chart_type = 'pie'
                value_counts = data[selected_column].value_counts()
                chart_data = value_counts.to_dict()
            else:
                flash(f"The selected column has {unique_vals} unique values. Visualization is skipped to avoid clutter.", "warning")

    
    return render_template(
        'visualization.html',
        column_names=column_names,
        # pie_data=pie_data,
        selected_column=selected_column,
        chart_data = chart_data,
        chart_type=chart_type
    )




 


    
