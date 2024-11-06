# losda_data_science_backend

# to create your virtual-environment
python3 -m venv 'folder_name'

# to install the requirements dependency
pip install -r requirements.txt

# Initialize migrations
flask db init

# Create migration scripts
flask db migrate -m "Initial migration"

# Apply the migrations to the database
flask db upgrade

# To run flask for this project
flask run

#  To build the Tailwind CSS:
npx tailwindcss -i ./app/static/css/styles.css -o ./app/static/css/output.css --watch


