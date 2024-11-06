from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from flask_login import login_required, current_user
from app import db
import pandas as pd

api = Blueprint('api', __name__)

@api.route('/')
@login_required
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return render_template('index.html', title='Dashboard', username=current_user.username)

@api.route('/upload', methods=['POST'])
def upload_file():
    # if not current_user.is_authenticated:
    #     return redirect(url_for('auth.login'))
    
    if 'file' not in request.files:
        return render_template('upload_file.html', error="No file uploaded")
    
    file = request.files['file']

    try:
        data = pd.read_csv(file)
        return render_template('upload_file.html', message="File processed successfully")
    except Exception as e:
        return render_template('upload_file.html', error=str(e))
    
