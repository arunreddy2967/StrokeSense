from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
import joblib
import json
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps


app = Flask(__name__)

app.secret_key = "strokesense-secret-key"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_model.pkl"
MODEL_INFO_PATH = "models/model_info.json"

model = joblib.load(MODEL_PATH)

with open(MODEL_INFO_PATH, "r") as file:
    model_info = json.load(file)

THRESHOLD = float(
    model_info.get("threshold", 0.60)
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    try:

        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="strokesense123",
            database="strokesense"
        )

        return connection

    except mysql.connector.Error as error:

        print(
            "Database connection error:",
            error
        )

        return None

# ============================================================
# LOGIN REQUIRED
# ============================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            return redirect(
                url_for("account")
            )

        return function(*args, **kwargs)

    return decorated_function


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# ACCOUNT / SIGN IN / PROFILE
# ============================================================

@app.route(
    "/account",
    methods=["GET", "POST"]
)
def account():

    if (
        "user_id" in session
        and request.method == "GET"
    ):

        connection = get_db_connection()

        if connection is None:

            return render_template(
                "account.html",
                logged_in=True,
                error="Database connection failed."
            )

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                user_id,
                full_name,
                email,
                phone,
                created_at
            FROM users
            WHERE user_id = %s
            """,
            (session["user_id"],)
        )

        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user is None:

            session.clear()

            return redirect(
                url_for("account")
            )

        return render_template(
            "account.html",
            logged_in=True,
            user=user
        )


    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not email or not password:

            return render_template(
                "account.html",
                logged_in=False,
                error="Please enter your email and password."
            )

        connection = get_db_connection()

        if connection is None:

            return render_template(
                "account.html",
                logged_in=False,
                error="Database connection failed."
            )

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                user_id,
                full_name,
                email,
                phone,
                password
            FROM users
            WHERE email = %s
            """,
            (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        connection.close()

        if user is None:

            return render_template(
                "account.html",
                logged_in=False,
                error="Invalid email or password."
            )

        if not check_password_hash(
            user["password"],
            password
        ):

            return render_template(
                "account.html",
                logged_in=False,
                error="Invalid email or password."
            )

        session["user_id"] = user["user_id"]
        session["full_name"] = user["full_name"]
        session["email"] = user["email"]

        return redirect(
            url_for("dashboard")
        )


    registered = request.args.get(
        "registered",
        ""
    )

    if registered == "success":

        return render_template(
            "account.html",
            logged_in=False,
            message="Account created successfully. Please sign in."
        )


    return render_template(
        "account.html",
        logged_in=False
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        full_name = request.form.get(
            "full_name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if (
            not full_name
            or not email
            or not phone
            or not password
            or not confirm_password
        ):

            return render_template(
                "register.html",
                error="Please fill in all fields."
            )

        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        if len(password) < 6:

            return render_template(
                "register.html",
                error="Password must contain at least 6 characters."
            )

        connection = get_db_connection()

        if connection is None:

            return render_template(
                "register.html",
                error="Database connection failed."
            )

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT user_id
            FROM users
            WHERE email = %s
            """,
            (email,)
        )

        existing_user = cursor.fetchone()

        if existing_user is not None:

            cursor.close()
            connection.close()

            return render_template(
                "register.html",
                error="An account with this email already exists."
            )

        hashed_password = generate_password_hash(
            password
        )

        cursor.execute(
            """
            INSERT INTO users (
                full_name,
                email,
                phone,
                password
            )
            VALUES (
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                full_name,
                email,
                phone,
                hashed_password
            )
        )

        connection.commit()

        cursor.close()
        connection.close()

        return redirect(
            url_for(
                "account",
                registered="success"
            )
        )

    return render_template(
        "register.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("home")
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

@app.route(
    "/risk-assessment",
    methods=["GET", "POST"]
)
@login_required
def risk_assessment():

    if request.method == "POST":

        try:

            gender = request.form[
                "gender"
            ]

            age = float(
                request.form["age"]
            )

            hypertension = int(
                request.form["hypertension"]
            )

            heart_disease = int(
                request.form["heart_disease"]
            )

            ever_married = request.form[
                "ever_married"
            ]

            work_type = request.form[
                "work_type"
            ]

            residence_type = request.form[
                "Residence_type"
            ]

            avg_glucose_level = float(
                request.form[
                    "avg_glucose_level"
                ]
            )

            bmi_value = request.form.get(
                "bmi",
                ""
            ).strip()

            smoking_status = request.form[
                "smoking_status"
            ]

            if bmi_value == "":
                bmi = None
            else:
                bmi = float(
                    bmi_value
                )

            input_data = pd.DataFrame(
                [{
                    "gender": gender,
                    "age": age,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "ever_married": ever_married,
                    "work_type": work_type,
                    "Residence_type": residence_type,
                    "avg_glucose_level": avg_glucose_level,
                    "bmi": bmi,
                    "smoking_status": smoking_status
                }]
            )

            probability = float(
                model.predict_proba(
                    input_data
                )[0][1]
            )

            risk_percentage = (
                probability * 100
            )

            if probability >= THRESHOLD:
                risk_category = "HIGH RISK"
            else:
                risk_category = "LOW RISK"

            connection = get_db_connection()

            if connection is None:
                return "Database connection failed"

            cursor = connection.cursor()

            query = """
                INSERT INTO assessments (
                    user_id,
                    gender,
                    age,
                    hypertension,
                    heart_disease,
                    ever_married,
                    work_type,
                    residence_type,
                    avg_glucose_level,
                    bmi,
                    smoking_status,
                    risk_probability,
                    risk_category
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
            """

            values = (
                session["user_id"],
                gender,
                age,
                hypertension,
                heart_disease,
                ever_married,
                work_type,
                residence_type,
                avg_glucose_level,
                bmi,
                smoking_status,
                probability,
                risk_category
            )

            cursor.execute(
                query,
                values
            )

            connection.commit()

            assessment_id = cursor.lastrowid

            cursor.close()
            connection.close()

            return render_template(
                "result.html",
                risk_probability=round(
                    risk_percentage,
                    2
                ),
                risk_category=risk_category,
                assessment_id=assessment_id
            )

        except Exception as error:

            print(
                "Prediction error:",
                error
            )

            return (
                f"Error during risk assessment: "
                f"{error}"
            )

    return render_template(
        "assessment.html"
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
@login_required
def history():

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT
            assessment_id,
            assessment_date,
            gender,
            age,
            avg_glucose_level,
            bmi,
            risk_probability,
            risk_category
        FROM assessments
        WHERE user_id = %s
        ORDER BY assessment_id DESC
        """,
        (session["user_id"],)
    )

    assessments = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "history.html",
        assessments=assessments
    )


# ============================================================
# ASSESSMENT DETAILS
# ============================================================

@app.route(
    "/assessment/<int:assessment_id>"
)
@login_required
def assessment_details(
    assessment_id
):

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT *
        FROM assessments
        WHERE assessment_id = %s
        AND user_id = %s
        """,
        (
            assessment_id,
            session["user_id"]
        )
    )

    assessment = cursor.fetchone()

    cursor.close()
    connection.close()

    if assessment is None:

        return (
            "Assessment not found."
        ), 404

    return render_template(
        "assessment_details.html",
        assessment=assessment
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    user_id = session["user_id"]


    # --------------------------------------------------------
    # TOTAL ASSESSMENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM assessments
        WHERE user_id = %s
        """,
        (user_id,)
    )

    total_result = cursor.fetchone()

    total_assessments = int(
        total_result["total"]
    )


    # --------------------------------------------------------
    # GET ASSESSMENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            assessment_id,
            assessment_date,
            gender,
            age,
            hypertension,
            heart_disease,
            avg_glucose_level,
            bmi,
            risk_probability,
            risk_category
        FROM assessments
        WHERE user_id = %s
        ORDER BY assessment_id DESC
        """,
        (user_id,)
    )

    assessments = cursor.fetchall()

    cursor.close()
    connection.close()


    # --------------------------------------------------------
    # RISK COUNTS
    # --------------------------------------------------------

    high_risk = sum(
        1
        for item in assessments
        if item["risk_category"] == "HIGH RISK"
    )

    low_risk = sum(
        1
        for item in assessments
        if item["risk_category"] == "LOW RISK"
    )


    # --------------------------------------------------------
    # AVERAGE RISK AS PERCENTAGE
    # --------------------------------------------------------

    if total_assessments > 0:

        average_risk = (
            sum(
                float(
                    item["risk_probability"]
                )
                for item in assessments
            )
            / total_assessments
        ) * 100

        average_risk = round(
            average_risk,
            2
        )

    else:

        average_risk = 0


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    risk_distribution_data = [
        high_risk,
        low_risk
    ]


    # --------------------------------------------------------
    # AGE VS RISK
    # --------------------------------------------------------

    age_risk_data = []

    for item in assessments:

        age_risk_data.append(
            {
                "x": float(
                    item["age"]
                ),
                "y": round(
                    float(
                        item["risk_probability"]
                    ) * 100,
                    2
                )
            }
        )


    # --------------------------------------------------------
    # GLUCOSE VS RISK
    # --------------------------------------------------------

    glucose_risk_data = []

    for item in assessments:

        glucose_risk_data.append(
            {
                "x": float(
                    item["avg_glucose_level"]
                ),
                "y": round(
                    float(
                        item["risk_probability"]
                    ) * 100,
                    2
                )
            }
        )


    # --------------------------------------------------------
    # BMI VS RISK
    # --------------------------------------------------------

    bmi_risk_data = []

    for item in assessments:

        if item["bmi"] is not None:

            bmi_risk_data.append(
                {
                    "x": float(
                        item["bmi"]
                    ),
                    "y": round(
                        float(
                            item["risk_probability"]
                        ) * 100,
                        2
                    )
                }
            )


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        assessments=assessments,

        total=total_assessments,

        total_assessments=total_assessments,

        high_risk=high_risk,

        low_risk=low_risk,

        average_risk=average_risk,

        risk_distribution_data=risk_distribution_data,

        age_risk_data=age_risk_data,

        glucose_risk_data=glucose_risk_data,

        bmi_risk_data=bmi_risk_data
    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
@login_required
def analytics():

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    user_id = session["user_id"]


    # --------------------------------------------------------
    # TOTAL ASSESSMENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM assessments
        WHERE user_id = %s
        """,
        (user_id,)
    )

    total_result = cursor.fetchone()

    total = int(
        total_result["total"]
    )


    # --------------------------------------------------------
    # GET DATA
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            assessment_id,
            age,
            hypertension,
            heart_disease,
            avg_glucose_level,
            bmi,
            smoking_status,
            risk_probability,
            risk_category
        FROM assessments
        WHERE user_id = %s
        ORDER BY assessment_id DESC
        """,
        (user_id,)
    )

    assessments = cursor.fetchall()

    cursor.close()
    connection.close()


    # --------------------------------------------------------
    # RISK COUNTS
    # --------------------------------------------------------

    high_risk_count = sum(
        1
        for item in assessments
        if item["risk_category"] == "HIGH RISK"
    )

    low_risk_count = sum(
        1
        for item in assessments
        if item["risk_category"] == "LOW RISK"
    )


    # --------------------------------------------------------
    # RISK PERCENTAGES
    # --------------------------------------------------------

    if total > 0:

        high_risk_percentage = round(
            (high_risk_count / total) * 100,
            2
        )

        low_risk_percentage = round(
            (low_risk_count / total) * 100,
            2
        )

        average_risk = (
            sum(
                float(
                    item["risk_probability"]
                )
                for item in assessments
            )
            / total
        ) * 100

        average_risk = round(
            average_risk,
            2
        )

    else:

        high_risk_percentage = 0
        low_risk_percentage = 0
        average_risk = 0


    # --------------------------------------------------------
    # AGE VS RISK
    # --------------------------------------------------------

    age_data = []

    for item in assessments:

        age_data.append(
            {
                "x": float(
                    item["age"]
                ),
                "y": round(
                    float(
                        item["risk_probability"]
                    ) * 100,
                    2
                )
            }
        )


    # --------------------------------------------------------
    # GLUCOSE VS RISK
    # --------------------------------------------------------

    glucose_data = []

    for item in assessments:

        glucose_data.append(
            {
                "x": float(
                    item["avg_glucose_level"]
                ),
                "y": round(
                    float(
                        item["risk_probability"]
                    ) * 100,
                    2
                )
            }
        )


    # --------------------------------------------------------
    # BMI VS RISK
    # --------------------------------------------------------

    bmi_data = []

    for item in assessments:

        if item["bmi"] is not None:

            bmi_data.append(
                {
                    "x": float(
                        item["bmi"]
                    ),
                    "y": round(
                        float(
                            item["risk_probability"]
                        ) * 100,
                        2
                    )
                }
            )


    # --------------------------------------------------------
    # HYPERTENSION
    # --------------------------------------------------------

    hypertension_data = [
        {
            "label": "No",
            "value": sum(
                1
                for item in assessments
                if int(item["hypertension"]) == 0
            )
        },
        {
            "label": "Yes",
            "value": sum(
                1
                for item in assessments
                if int(item["hypertension"]) == 1
            )
        }
    ]


    # --------------------------------------------------------
    # HEART DISEASE
    # --------------------------------------------------------

    heart_disease_data = [
        {
            "label": "No",
            "value": sum(
                1
                for item in assessments
                if int(item["heart_disease"]) == 0
            )
        },
        {
            "label": "Yes",
            "value": sum(
                1
                for item in assessments
                if int(item["heart_disease"]) == 1
            )
        }
    ]


    # --------------------------------------------------------
    # SMOKING STATUS
    # --------------------------------------------------------

    smoking_statuses = [
        "never smoked",
        "formerly smoked",
        "smokes",
        "Unknown"
    ]

    smoking_data = []

    for status in smoking_statuses:

        smoking_data.append(
            {
                "label": status,
                "value": sum(
                    1
                    for item in assessments
                    if item["smoking_status"] == status
                )
            }
        )


    # --------------------------------------------------------
    # ANALYTICS
    # --------------------------------------------------------

    return render_template(
        "analytics.html",

        total=total,

        high_risk_percentage=high_risk_percentage,

        low_risk_percentage=low_risk_percentage,

        average_risk=average_risk,

        age_data=age_data,

        glucose_data=glucose_data,

        bmi_data=bmi_data,

        hypertension_data=hypertension_data,

        heart_disease_data=heart_disease_data,

        smoking_data=smoking_data
    )


# ============================================================
# PATIENTS
# ============================================================

@app.route("/patients")
@login_required
def patients():

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT
            assessment_id,
            assessment_date,
            age,
            gender,
            avg_glucose_level,
            bmi,
            risk_probability,
            risk_category
        FROM assessments
        WHERE user_id = %s
        ORDER BY assessment_id DESC
        """,
        (session["user_id"],)
    )

    patients_data = cursor.fetchall()

    cursor.close()
    connection.close()

    return render_template(
        "patients.html",
        patients=patients_data
    )


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
@login_required
def reports():

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT
            assessment_id,
            assessment_date,
            age,
            gender,
            avg_glucose_level,
            bmi,
            risk_probability,
            risk_category
        FROM assessments
        WHERE user_id = %s
        ORDER BY assessment_id DESC
        """,
        (session["user_id"],)
    )

    assessments = cursor.fetchall()

    cursor.close()
    connection.close()


    total = len(
        assessments
    )

    high_risk = sum(
        1
        for item in assessments
        if item["risk_category"] == "HIGH RISK"
    )

    low_risk = sum(
        1
        for item in assessments
        if item["risk_category"] == "LOW RISK"
    )


    if total > 0:

        average_risk = (
            sum(
                float(
                    item["risk_probability"]
                )
                for item in assessments
            )
            / total
        ) * 100

        average_risk = round(
            average_risk,
            2
        )

    else:

        average_risk = 0


    return render_template(
        "reports.html",
        assessments=assessments,
        total=total,
        high_risk=high_risk,
        low_risk=low_risk,
        average_risk=average_risk
    )


# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings")
@login_required
def settings():

    connection = get_db_connection()

    if connection is None:
        return "Database connection failed"

    cursor = connection.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM assessments
        WHERE user_id = %s
        """,
        (session["user_id"],)
    )

    result = cursor.fetchone()

    assessment_records = int(
        result["total"]
    )

    cursor.close()
    connection.close()


    return render_template(
        "settings.html",

        version="1.0",

        model_name=model_info.get(
            "model",
            "Logistic Regression"
        ),

        threshold=THRESHOLD * 100,

        database="MySQL",

        dataset="Stroke Prediction Dataset",

        assessment_records=assessment_records,

        accuracy=model_info.get(
            "accuracy",
            0
        ) * 100,

        precision=model_info.get(
            "precision",
            0
        ) * 100,

        recall=model_info.get(
            "recall",
            0
        ) * 100,

        f1_score=model_info.get(
            "f1_score",
            0
        ) * 100,

        roc_auc=model_info.get(
            "roc_auc",
            0
        ) * 100
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "=============================================="
    )

    print(
        "StrokeSense Flask + ML Model Connected Successfully"
    )

    print(
        "Model:",
        model_info.get(
            "model",
            "Unknown"
        )
    )

    print(
        "Threshold:",
        THRESHOLD
    )

    print(
        "Model loaded:",
        "Yes" if model is not None else "No"
    )

    print(
        "=============================================="
    )

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )