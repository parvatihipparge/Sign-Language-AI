
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, flash
)
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, timezone
from sqlalchemy import desc, func, text
from functools import wraps
from types import SimpleNamespace
from werkzeug.security import generate_password_hash, check_password_hash

import os
import base64
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from PIL import Image


# =========================================
# CONFIGURATION
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIM = 126
NUM_CLASSES = 37

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "checkpoint.pth"
)

CLASS_NAMES = [
    "Hello", "Goodbye", "ThankYou", "Please",
    "Yes", "No", "Maybe", "OK",
    "Help", "Water", "Food", "Bathroom",
    "Happy", "Sad", "Love", "Sorry",
    "Today", "Tomorrow", "Later", "Now",
    "What", "Where", "When", "How",
    "One", "Two", "Three", "Four", "Five",
    "Want", "Need", "Give", "Take",
    "Mother", "Father", "Family", "Friend"
]


# =========================================
# FLASK CONFIGURATION
# =========================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "sign-language-ai-development-secret-key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sign_language.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================================
# DATABASE MODELS
# =========================================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )


class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=True
    )

    label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    landmarks_detected = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.String(500), nullable=True)

    timestamp = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    session_id = db.Column(db.String(50), nullable=True)

    def to_dict(self):

        return {
            "id": self.id,
            "user_id": self.user_id,
            "label": self.label,
            "confidence": self.confidence,
            "landmarks_detected": self.landmarks_detected,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id
        }


class GestureInfo(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(50), nullable=True)
    image_path = db.Column(db.String(200), nullable=True)

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )


class Statistics(db.Model):

    id = db.Column(db.Integer, primary_key=True)
    total_predictions = db.Column(db.Integer, default=0)
    successful_predictions = db.Column(db.Integer, default=0)
    failed_predictions = db.Column(db.Integer, default=0)

    last_updated = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )


# =========================================
# LOGIN REQUIRED DECORATOR
# =========================================

def login_required(route_function):

    @wraps(route_function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            if request.path.startswith("/api/") or request.path == "/predict":

                return jsonify({
                    "success": False,
                    "error": "Please log in to continue."
                }), 401

            return redirect(url_for("login"))

        return route_function(*args, **kwargs)

    return wrapper


# =========================================
# AUTHENTICATION ROUTES
# =========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password or not confirm_password:

            flash("Please fill in all fields.", "error")
            return render_template("register.html")

        if len(name) > 100 or len(email) > 150:

            flash("Name or email is too long.", "error")
            return render_template("register.html")

        if len(password) < 8:

            flash("Password must contain at least 8 characters.", "error")
            return render_template("register.html")

        if password != confirm_password:

            flash("Passwords do not match.", "error")
            return render_template("register.html")

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:

            flash("An account with this email already exists.", "error")
            return render_template("register.html")

        try:

            new_user = User(
                name=name,
                email=email,
                password=generate_password_hash(password)
            )

            db.session.add(new_user)
            db.session.commit()

            flash("Registration successful! Please log in.", "success")

            return redirect(url_for("login"))

        except Exception as e:

            db.session.rollback()
            print("Registration error:", e)

            flash("Something went wrong. Please try again.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:

            flash("Please enter your email and password.", "error")
            return render_template("login.html")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session.clear()
            session["user_id"] = user.id
            session["user_name"] = user.name

            flash("Welcome back, " + user.name + "!", "success")

            return redirect(url_for("index"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out successfully.", "success")

    return redirect(url_for("login"))


# =========================================
# USER PROFILE
# =========================================

@app.route("/profile")
@login_required
def profile():

    user = User.query.get(session["user_id"])

    if not user:
        session.clear()
        return redirect(url_for("login"))

    total_predictions = Prediction.query.filter_by(
        user_id=user.id
    ).count()

    successful_predictions = Prediction.query.filter_by(
        user_id=user.id,
        landmarks_detected=True
    ).count()

    failed_predictions = Prediction.query.filter_by(
        user_id=user.id,
        landmarks_detected=False
    ).count()

    return render_template(
        "profile.html",
        user=user,
        total_predictions=total_predictions,
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions
    )

# =========================================
# EDIT PROFILE
# =========================================

@app.route("/edit-profile", methods=["GET", "POST"])
@login_required
def edit_profile():

    user = User.query.get(session["user_id"])

    if not user:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        current_password = request.form.get("current_password", "")

        if not name or not email or not current_password:
            flash("Please fill in all fields.", "error")
            return render_template("edit_profile.html", user=user)

        if len(name) > 100 or len(email) > 150:
            flash("Name or email is too long.", "error")
            return render_template("edit_profile.html", user=user)

        if not check_password_hash(user.password, current_password):
            flash("Your current password is incorrect.", "error")
            return render_template("edit_profile.html", user=user)

        existing_user = User.query.filter(
            User.email == email,
            User.id != user.id
        ).first()

        if existing_user:
            flash("This email address is already registered.", "error")
            return render_template("edit_profile.html", user=user)

        try:

            user.name = name
            user.email = email

            db.session.commit()

            session["user_name"] = user.name

            flash("Your profile has been updated successfully!", "success")

            return redirect(url_for("profile"))

        except Exception as e:

            db.session.rollback()
            print("Profile update error:", e)

            flash("Unable to update your profile. Please try again.", "error")

    return render_template("edit_profile.html", user=user)


# =========================================
# CHANGE PASSWORD
# =========================================

@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():

    user = User.query.get(session["user_id"])

    if not user:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":

        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not current_password or not new_password or not confirm_password:
            flash("Please fill in all fields.", "error")
            return render_template("change_password.html")

        if not check_password_hash(user.password, current_password):
            flash("Your current password is incorrect.", "error")
            return render_template("change_password.html")

        if len(new_password) < 8:
            flash("New password must contain at least 8 characters.", "error")
            return render_template("change_password.html")

        if new_password != confirm_password:
            flash("New passwords do not match.", "error")
            return render_template("change_password.html")

        if check_password_hash(user.password, new_password):
            flash("Your new password must be different from your current password.", "error")
            return render_template("change_password.html")

        try:

            user.password = generate_password_hash(new_password)

            db.session.commit()

            flash("Your password has been changed successfully!", "success")

            return redirect(url_for("profile"))

        except Exception as e:

            db.session.rollback()
            print("Password update error:", e)

            flash("Unable to change your password. Please try again.", "error")

    return render_template("change_password.html")


# =========================================
# DELETE USER ACCOUNT
# =========================================

@app.route("/delete-account", methods=["GET", "POST"])
@login_required
def delete_account():

    user = User.query.get(session["user_id"])

    if not user:
        session.clear()
        return redirect(url_for("login"))

    if request.method == "POST":

        current_password = request.form.get("current_password", "")
        confirmation = request.form.get("confirmation", "")

        if not current_password or not confirmation:
            flash("Please fill in all fields.", "error")
            return render_template("delete_account.html")

        if not check_password_hash(user.password, current_password):
            flash("Your current password is incorrect.", "error")
            return render_template("delete_account.html")

        if confirmation != "DELETE":
            flash("Please type DELETE exactly to confirm account deletion.", "error")
            return render_template("delete_account.html")

        try:

            user_id = user.id

            # Delete only this user's prediction records.
            Prediction.query.filter_by(
                user_id=user_id
            ).delete(synchronize_session=False)

            # Delete the user account.
            db.session.delete(user)

            db.session.commit()

            # Clear the logged-in session.
            session.clear()

            flash(
                "Your account and personal prediction history have been permanently deleted.",
                "success"
            )

            return redirect(url_for("login"))

        except Exception as e:

            db.session.rollback()

            print("Account deletion error:", e)

            flash(
                "Unable to delete your account. Please try again.",
                "error"
            )

    return render_template("delete_account.html")



# =========================================
# MODEL ARCHITECTURE
# =========================================

class LandmarkTransformer(nn.Module):

    def __init__(
        self,
        input_dim=126,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=37
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(0.3),

            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        return self.network(x)


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = LandmarkTransformer(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES
).to(DEVICE)

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Checkpoint not found: {MODEL_PATH}"
    )

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

print("Trained model loaded successfully!")
print("Device:", DEVICE)


# =========================================
# MEDIAPIPE SETUP
# =========================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# =========================================
# EXTRACT TWO-HAND LANDMARKS
# =========================================

def extract_landmarks(results):

    hand_data = {
        "Left": None,
        "Right": None
    }

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            hand_label = (
                handedness.classification[0].label
            )

            if hand_label in hand_data:

                hand_data[hand_label] = hand_landmarks

    landmarks = []

    for hand_label in ["Left", "Right"]:

        if hand_data[hand_label] is not None:

            for lm in hand_data[hand_label].landmark:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

        else:

            landmarks.extend([0.0] * 63)

    return landmarks


# =========================================
# INITIALIZE GESTURES
# =========================================

def initialize_gestures():

    categories = {

        "Hello": "Basic Greetings",
        "Goodbye": "Basic Greetings",
        "ThankYou": "Basic Greetings",
        "Please": "Basic Greetings",

        "Yes": "Common Responses",
        "No": "Common Responses",
        "Maybe": "Common Responses",
        "OK": "Common Responses",

        "Help": "Basic Needs",
        "Water": "Basic Needs",
        "Food": "Basic Needs",
        "Bathroom": "Basic Needs",

        "Happy": "Emotions",
        "Sad": "Emotions",
        "Love": "Emotions",
        "Sorry": "Emotions",

        "Today": "Time Related",
        "Tomorrow": "Time Related",
        "Later": "Time Related",
        "Now": "Time Related",

        "What": "Questions",
        "Where": "Questions",
        "When": "Questions",
        "How": "Questions",

        "One": "Numbers",
        "Two": "Numbers",
        "Three": "Numbers",
        "Four": "Numbers",
        "Five": "Numbers",

        "Want": "Common Actions",
        "Need": "Common Actions",
        "Give": "Common Actions",
        "Take": "Common Actions",

        "Mother": "Family",
        "Father": "Family",
        "Family": "Family",
        "Friend": "Family"
    }

    for name, category in categories.items():

        if not GestureInfo.query.filter_by(name=name).first():

            gesture = GestureInfo(
                name=name,
                category=category,
                description=f"Sign for {name.lower()} in sign language."
            )

            db.session.add(gesture)

    db.session.commit()


# =========================================
# HOME PAGE
# =========================================

@app.route("/")
@login_required
def index():

    return render_template("index.html")


# =========================================
# DETECTION PAGE
# =========================================

@app.route("/detect")
@login_required
def detect():

    return render_template("detect.html")


# =========================================
# GESTURES PAGE
# =========================================

@app.route("/gestures")
@login_required
def gestures():

    category = request.args.get("category")

    if category:

        gesture_list = GestureInfo.query.filter_by(
            category=category
        ).order_by(GestureInfo.name).all()

    else:

        gesture_list = GestureInfo.query.order_by(
            GestureInfo.name
        ).all()

    categories = db.session.query(
        GestureInfo.category
    ).distinct().all()

    categories = [cat[0] for cat in categories]

    return render_template(
        "gestures.html",
        gestures=gesture_list,
        categories=categories,
        selected_category=category
    )


# =========================================
# USER HISTORY PAGE
# =========================================

@app.route("/history")
@login_required
def history():

    page = request.args.get("page", 1, type=int)

    predictions = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        desc(Prediction.timestamp)
    ).paginate(
        page=page,
        per_page=10,
        error_out=False
    )

    return render_template(
        "history.html",
        predictions=predictions
    )


# =========================================
# PREVIOUS HISTORY PAGE
# =========================================

@app.route("/previous-history")
@login_required
def previous_history():

    page = request.args.get("page", 1, type=int)

    predictions = Prediction.query.filter(
        Prediction.user_id.is_(None)
    ).order_by(
        desc(Prediction.timestamp)
    ).paginate(
        page=page,
        per_page=10,
        error_out=False
    )

    return render_template(
        "previous_history.html",
        predictions=predictions
    )


# =========================================
# USER STATISTICS PAGE
# =========================================

@app.route("/statistics")
@login_required
def statistics():

    user_id = session["user_id"]

    total_predictions = Prediction.query.filter_by(
        user_id=user_id
    ).count()

    successful_predictions = Prediction.query.filter_by(
        user_id=user_id,
        landmarks_detected=True
    ).count()

    failed_predictions = Prediction.query.filter_by(
        user_id=user_id,
        landmarks_detected=False
    ).count()

    stats = SimpleNamespace(
        total_predictions=total_predictions,
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions
    )

    raw_stats = db.session.query(
        Prediction.label,
        func.count(Prediction.id).label("count")
    ).filter(
        Prediction.user_id == user_id,
        Prediction.landmarks_detected == True
    ).group_by(
        Prediction.label
    ).order_by(
        desc(func.count(Prediction.id))
    ).limit(10).all()

    gesture_stats = [
        (item.label, item.count)
        for item in raw_stats
    ]

    week_ago = (
        datetime.now(timezone.utc)
        - timedelta(days=7)
    )

    recent_predictions = Prediction.query.filter(
        Prediction.user_id == user_id,
        Prediction.timestamp >= week_ago
    ).count()

    return render_template(
        "statistics.html",
        stats=stats,
        gesture_stats=gesture_stats,
        recent_predictions=recent_predictions
    )


# =========================================
# ABOUT PAGE
# =========================================

@app.route("/about")
@login_required
def about():

    return render_template("about.html")


# =========================================
# PREDICTION API
# =========================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    data = request.get_json(silent=True)

    if not data or "image" not in data:

        return jsonify({
            "error": "No image provided"
        }), 400

    session_id = data.get("session_id", "unknown")
    user_id = session["user_id"]

    try:

        image_data = data["image"].split(",")[-1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image_np = np.array(image)

        results = hands.process(image_np)

        if not results.multi_hand_landmarks:

            prediction = Prediction(
                user_id=user_id,
                label="No Hand Detected",
                confidence=0.0,
                landmarks_detected=False,
                session_id=session_id
            )

            db.session.add(prediction)
            db.session.commit()

            return jsonify({
                "error": "No hand detected",
                "label": "No Hand Detected",
                "confidence": 0.0
            })

        landmarks = extract_landmarks(results)

        if len(landmarks) != INPUT_DIM:
            raise ValueError("Incorrect landmark dimension")

        input_tensor = torch.tensor(
            landmarks,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        label = CLASS_NAMES[predicted.item()]
        confidence_value = float(confidence.item())

        prediction = Prediction(
            user_id=user_id,
            label=label,
            confidence=confidence_value,
            landmarks_detected=True,
            session_id=session_id
        )

        db.session.add(prediction)
        db.session.commit()

        return jsonify({
            "label": label,
            "confidence": confidence_value,
            "landmarks_detected": True,
            "hands_detected": len(results.multi_hand_landmarks)
        })

    except Exception as e:

        db.session.rollback()

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================================
# USER PREDICTION HISTORY API
# =========================================

@app.route("/api/predictions", methods=["GET"])
@login_required
def get_predictions():

    limit = request.args.get("limit", 50, type=int)

    limit = max(1, min(limit, 200))

    predictions = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        desc(Prediction.timestamp)
    ).limit(limit).all()

    return jsonify([
        prediction.to_dict()
        for prediction in predictions
    ])


# =========================================
# CLEAR CURRENT USER HISTORY
# =========================================

@app.route("/api/clear_history", methods=["POST"])
@login_required
def clear_history():

    try:

        Prediction.query.filter_by(
            user_id=session["user_id"]
        ).delete()

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Your prediction history has been cleared."
        })

    except Exception as e:

        db.session.rollback()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================
# DATABASE INITIALIZATION AND MIGRATION
# =========================================

with app.app_context():

    db.create_all()

    # Add user_id to an existing prediction table
    # without deleting existing prediction records.

    columns = db.session.execute(
        text("PRAGMA table_info(prediction)")
    ).fetchall()

    column_names = [column[1] for column in columns]

    if "user_id" not in column_names:

        db.session.execute(
            text(
                "ALTER TABLE prediction "
                "ADD COLUMN user_id INTEGER "
                "REFERENCES user(id)"
            )
        )

        db.session.commit()

        print("Prediction table migrated successfully!")

    initialize_gestures()

    print("Database initialized successfully!")


# =========================================
# RUN APPLICATION
# =========================================

if __name__ == "__main__":

    app.run(debug=True)
