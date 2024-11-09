from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required
from app import db
from app.models.user import User
from werkzeug.security import generate_password_hash, check_password_hash
# from .login_form import LoginForm
# from .register_form import RegisterForm

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('User registered successfully!')
        return redirect(url_for('api.index'))
    
    return render_template('register.html')


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user is None:
            flash("Email does not exist.", "error")
            return redirect(url_for('auth.login'))

        
        if not check_password_hash(user.password_hash, password):
            flash("Incorrect password.", "error")
            return redirect(url_for('auth.login'))

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('api.index'))
    
        return redirect(url_for('auth.login'))
    
    return render_template('login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


