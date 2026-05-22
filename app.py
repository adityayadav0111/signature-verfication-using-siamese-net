from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
import base64
import torch
from PIL import Image
import numpy as np
from datetime import timedelta
from torch import nn
from io import BytesIO
import tempfile
import uuid
import atexit
import shutil


# Defining the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Siamese Network class (same as in training file)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(25600, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Loading the trained Siamese model
model_path = 'siamese_model.pth' 
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Setting the model to evaluation mode

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '1234'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Setting session timeout

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/signatures'  # Folder to store uploaded signature images
db = SQLAlchemy(app)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Defining User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    signature_path = db.Column(db.String(200))  # Path to the reference signature image

# Creating the database and tables
with app.app_context():
    db.create_all()

# Preprocess image function (matches training pipeline)
def preprocess_image(image_path, target_size=(100, 100)):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img.to(device)

# Comparing signatures function
def compare_signatures(model, image1, image2, max_distance=5.0):  # Adjust max_distance
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        normalized_score = 1 - (euclidean_distance.item() / max_distance)
        normalized_score = max(0, min(1, normalized_score))  # Clamp to [0, 1]

    return normalized_score


def find_most_similar_image(model, uploaded_image_path, signatures_folder):
    """
    Compare the uploaded image with all images in the signatures folder.
    Returns the path of the most similar image and the similarity score.
    """
    max_similarity_score = 0 # Initialize with a very low value
    most_similar_image_path = None

    # Iterate through all images in the signatures folder
    for filename in os.listdir(signatures_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            reference_image_path = os.path.join(signatures_folder, filename)
            
            # Compare the uploaded image with the current reference image
            similarity_score = compare_signatures(model, uploaded_image_path, reference_image_path)
            
            # Update the most similar image if the current score is higher
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                most_similar_image_path = reference_image_path

    return most_similar_image_path, max_similarity_score

# Classify signature function
def classify_signature(model, image_path, reference_path, max_distance=10.0):
    """
    Comparing two signatures and return the normalized similarity score.
    """
    similarity_score = compare_signatures(model, image_path, reference_path, max_distance)
    return similarity_score

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()

        if user and user.password == password:  # Compare plain text passwords
            session['user_id'] = user.id  # Store user ID in session
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        password = request.form.get('password')

        # Check if the email is already registered
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered!', 'error')
        else:
            # Create a new user
            new_user = User(full_name=full_name, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)


@app.route('/upload', methods=['GET', 'POST'])
def upload_signature():
    if 'user_id' not in session:
        flash('Please login.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)

        # Reading the uploaded file into memory (without saving it)
        file_data = file.read()
        file_stream = BytesIO(file_data)  # Create a file-like object in memory

        # Preprocessing the uploaded image for comparison
        uploaded_image = Image.open(file_stream).convert('L')  # Convert to grayscale
        uploaded_image = uploaded_image.resize((100, 100))  # Resize to match model input
        uploaded_image = np.array(uploaded_image) / 255.0  # Normalize to [0, 1]
        uploaded_image = torch.from_numpy(uploaded_image).float()  # Convert to tensor
        uploaded_image = uploaded_image.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

        # Find the most similar image in the signatures folder
        signatures_folder = app.config['UPLOAD_FOLDER']
        max_similarity_score = 0 # Initialize with a very low value
        most_similar_image_path = None

        for filename in os.listdir(signatures_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                reference_image_path = os.path.join(signatures_folder, filename)

                # Preprocess the reference image
                reference_image = preprocess_image(reference_image_path)

                # Compare the uploaded image with the current reference image
                with torch.no_grad():
                    output1, output2 = model(uploaded_image, reference_image)
                    euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                    similarity_score = 1 - (euclidean_distance.item() / 10.0)  # Normalize to [0, 1]
                    similarity_score = max(0, min(1, similarity_score))  # Clamp to [0, 1]

                # Update the most similar image if the current score is higher
                if similarity_score > max_similarity_score:
                    max_similarity_score = similarity_score
                    most_similar_image_path = reference_image_path

        # Determine the status based on the similarity score
        if max_similarity_score >= 0.98:
            status = "Signature Matched"
        elif max_similarity_score < 0.5:
            status = "Rejected"
        else:
            status = "Signature did not match"

        # Save the uploaded image only after comparison (if needed)
        filename = secure_filename(file.filename)
        verification_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(verification_image_path, 'wb') as f:
            f.write(file_data)  # Save the uploaded image to the folder

        # Render the result template
        return render_template(
            'upload_result.html',
            reference_image=most_similar_image_path.replace("\\", "/") if most_similar_image_path else None,
            verification_image=verification_image_path.replace("\\", "/"),
            similarity_score=max_similarity_score,
            status=status,
            user=session.get('user_id')
        )

    return render_template('upload.html')

@app.route('/canvas')
def canvas():
    if 'user_id' not in session:
        flash('Please login to draw a signature.', 'error')
        return redirect(url_for('login'))

    return render_template('canvas.html')


@app.route('/upload_canvas', methods=['POST'])
def upload_canvas():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login to compare signatures.'}), 401

    data = request.json
    image_data = data['image'].split(',')[1]  # Remove the data URL prefix
    file_data = base64.b64decode(image_data)  # Decode the base64 image data

    # Process the drawn signature in memory
    file_stream = BytesIO(file_data)  # Create a file-like object in memory
    drawn_image = Image.open(file_stream).convert('L')  # Convert to grayscale
    drawn_image = drawn_image.resize((100, 100))  # Resize to match model input
    drawn_image = np.array(drawn_image) / 255.0  # Normalize to [0, 1]
    drawn_image = torch.from_numpy(drawn_image).float()  # Convert to tensor
    drawn_image = drawn_image.unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

    # Compare the drawn signature with images in the folder
    signatures_folder = app.config['UPLOAD_FOLDER']
    max_similarity_score = -1  # Initialize with a very low value
    most_similar_image_path = None

    for filename in os.listdir(signatures_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
            reference_image_path = os.path.join(signatures_folder, filename)

            # Preprocess the reference image
            reference_image = preprocess_image(reference_image_path)

            # Compare the drawn image with the current reference image
            with torch.no_grad():
                output1, output2 = model(drawn_image, reference_image)
                euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
                similarity_score = 1 - (euclidean_distance.item() / 10.0)  # Normalize to [0, 1]
                similarity_score = max(0, min(1, similarity_score))  # Clamp to [0, 1]

            # Update the most similar image if the current score is higher
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                most_similar_image_path = reference_image_path

    # Determine the status
    status = "Signature Matched" if max_similarity_score >= 0.95 else "Signature did not match"

    # Store the result in the session for the canvas_result page
    session['canvas_result'] = {
        'reference_image': most_similar_image_path.replace("\\", "/"),  # Store the relative path
        'verification_image': f"data:image/png;base64,{image_data}",  # Base64-encoded image
        'similarity_score': max_similarity_score,
        'status': status
    }

    return jsonify({
        'message': 'Signature compared successfully!',
        'redirect': url_for('canvas_result')  # Redirect to the canvas result page
    })

@app.route('/save_canvas', methods=['POST'])
def save_canvas():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login to save a signature.'}), 401

    data = request.json
    image_data = data['image'].split(',')[1]  # Remove the data URL prefix
    file_data = base64.b64decode(image_data)  # Decode the base64 image data

    # Save the drawn signature to the signatures folder
    filename = f"canvas_signature_{session['user_id']}.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'wb') as f:
        f.write(file_data)

    return jsonify({
        'message': 'Signature saved successfully!'
    })

@app.route('/compare_canvas', methods=['POST'])
def compare_canvas():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login to compare signatures.'}), 401

    data = request.json
    image_data = data['image'].split(',')[1]
    file_data = base64.b64decode(image_data)

    # Save the drawn signature
    filename = f"canvas_signature_{uuid.uuid4().hex}.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'wb') as f:
        f.write(file_data)

    # Reload the model (optional, but recommended)
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Compare the drawn signature with stored signatures
    signatures_folder = app.config['UPLOAD_FOLDER']
    most_similar_image_path = None
    max_similarity_score = 0

    # Calculate distances and store them
    distances = []
    for filename in os.listdir(signatures_folder):
        reference_image_path = os.path.join(signatures_folder, filename)
        if reference_image_path == file_path:
            continue
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            similarity_score = compare_signatures(model, file_path, reference_image_path)
            distances.append(1 - similarity_score)  # Store distance (1 - similarity)
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                most_similar_image_path = reference_image_path

    # Calculate mean and standard deviation of distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Set dynamic threshold (e.g., mean + 2 * std)
    threshold = mean_distance + 2 * std_distance

    # Normalize distance to a similarity score
    normalized_distance = 1 - ((1 - max_similarity_score) / (mean_distance + 3 * std_distance))

    # Only proceed if normalized distance is greater than 0.95
    if normalized_distance >= 0.95:
        session['canvas_result'] = {
            'reference_image': most_similar_image_path.replace("\\", "/"),
            'verification_image': file_path.replace("\\", "/"),
            'similarity_score': normalized_distance,
            'status': "Signature Matched"
        }
        return jsonify({'redirect': url_for('canvas_result')})
    else:
        return jsonify({'error': 'No matching signature found with similarity score greater than 0.95.'}), 400
    
@app.route('/canvas_result')
def canvas_result():
    if 'user_id' not in session:
        flash('Please login to view the result.', 'error')
        return redirect(url_for('login'))

    # Retrieve the comparison results from the session
    canvas_result = session.get('canvas_result')
    if not canvas_result:
        flash('No comparison result found.', 'error')
        return redirect(url_for('dashboard'))

    # Only display the result if the similarity score is greater than 0.95
    if canvas_result['similarity_score'] >= 0.98:
        # Get the user's name
        user = User.query.get(session['user_id'])

        # Render the canvas_result template with the comparison results
        return render_template(
            'canvas_result.html',
            reference_image=canvas_result['reference_image'],
            verification_image=canvas_result['verification_image'],  # Path to the saved signature
            similarity_score=canvas_result['similarity_score'],
            status=canvas_result['status'],
            user=user  # Pass the user object to display the name
        )
    else:
        flash('No matching signature found with similarity score greater than 0.95.', 'error')
        return redirect(url_for('dashboard'))
    


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/about')
def about():
    if 'user_id' not in session:
        flash('Please login to access the about page.', 'error')
        return redirect(url_for('login'))
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 