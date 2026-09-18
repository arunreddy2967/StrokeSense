Absolutely. Paste everything below into README.md and save with Cmd + S.

# StrokeSense: ML-Driven Stroke Risk Prediction
## Project Overview
StrokeSense is a machine learning-based web application developed to estimate an individual's stroke risk using selected demographic, medical, and lifestyle parameters.
The system uses machine learning classification techniques to analyze factors such as:
- Age
- Gender
- Hypertension
- Heart Disease
- Average Glucose Level
- BMI
- Smoking Status
- Marital Status
- Work Type
- Residence Type
The application provides an estimated risk probability and categorizes the assessment as **Low Risk** or **High Risk**.
> **Disclaimer:** StrokeSense is an academic and preliminary risk-assessment project. It is not a medical diagnostic system and does not replace professional medical advice or diagnosis.
---
## Objectives
The main objectives of StrokeSense are:
1. To develop a machine learning model for stroke risk prediction.
2. To preprocess and analyze healthcare data.
3. To handle missing values and imbalanced data.
4. To compare multiple machine learning classification algorithms.
5. To select an effective model based on evaluation metrics.
6. To develop a simple web-based risk assessment system.
7. To provide users with quick preliminary risk estimation.
8. To maintain assessment history for registered users.
9. To provide dashboard and analytics features.
---
## Machine Learning
The project compares the following classification algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
### Final Model
The final model used by the application is:
**Logistic Regression**
Configuration:
- C = 0.01
- Class Weight = Balanced
- Maximum Iterations = 2000
- Decision Threshold = 0.60
The trained model is stored in:
```text
models/best_model.pkl

Model information is stored in:

models/model_info.json

⸻

Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

Final Logistic Regression Performance

Metric	Value
Accuracy	81.9961%
Precision	18.6916%
Recall	80.00%
F1-Score	30.3030%
ROC-AUC	84.0679%
Decision Threshold	60%

The dataset contains significantly fewer stroke cases than non-stroke cases. Therefore, accuracy alone is not sufficient for evaluating the model. Recall, precision, F1-score, ROC-AUC, and the confusion matrix are also considered.

⸻

Dataset

The project uses the Stroke Prediction Dataset by Fedesoriano from Kaggle.

Dataset file:

data/healthcare-dataset-stroke-data.csv

Dataset Information

* Total records: 5,110
* Original attributes: 12
* Stroke cases: 249
* Non-stroke cases: 4,861
* Missing BMI values: 201

The id column is removed during model training.

Dataset Attributes

Attribute	Description
id	Unique record identifier
gender	Gender of the individual
age	Age
hypertension	Hypertension status
heart_disease	Heart disease status
ever_married	Marital status
work_type	Type of occupation
Residence_type	Urban or Rural
avg_glucose_level	Average glucose level
bmi	Body Mass Index
smoking_status	Smoking status
stroke	Target variable

Target:

0 = No Stroke
1 = Stroke

⸻

Data Preprocessing

The following preprocessing steps are used:

1. Remove the id column.
2. Separate input features and target variable.
3. Split the dataset into training and testing sets.
4. Handle missing BMI values using median imputation.
5. Scale numerical features using StandardScaler.
6. Encode categorical features using OneHotEncoder.
7. Handle class imbalance using class weighting.
8. Train and evaluate multiple classification models.

The preprocessing steps are included in the machine learning pipeline so that the same transformations are applied during prediction.

⸻

System Architecture

                    USER
                      |
                      v
              Web Application
                      |
                      v
             Health Information
                      |
                      v
             Data Preprocessing
                      |
                      v
           Machine Learning Model
                      |
                      v
              Risk Probability
                      |
                      v
             Risk Classification
                /           \
               /             \
              v               v
          LOW RISK        HIGH RISK
               \             /
                \           /
                  v       v
                MySQL Database
                      |
                      v
             History / Dashboard

⸻

Application Features

Home

Provides an introduction to StrokeSense and navigation to the main sections.

User Registration

Users can create an account using:

* Full Name
* Email
* Phone Number
* Password

User Login

Registered users can log in to access their account and assessment information.

Risk Assessment

Users can enter health-related information and submit it for stroke risk estimation.

Assessment Result

The result page displays:

* Assessment ID
* Estimated Risk Percentage
* Decision Threshold
* Risk Category

Risk categories:

LOW RISK
HIGH RISK

Assessment History

Users can view their previous assessments, including:

* Assessment ID
* Date
* Age
* Gender
* Glucose Level
* BMI
* Risk Percentage
* Risk Category

Assessment Details

Users can open an individual assessment to view the complete information used for that assessment.

Dashboard

The dashboard provides:

* Total Assessments
* High Risk Assessments
* Low Risk Assessments
* Average Risk

It also provides visualizations such as:

* Risk Distribution
* Age vs Risk
* Glucose Level vs Risk
* BMI vs Risk

Analytics

The analytics page provides additional analysis using charts and assessment statistics.

Account

Users can view their account information and log out.

About

The About page provides information about:

* StrokeSense
* Brain Stroke
* Stroke Risk Factors
* FAST Warning Signs
* Stroke Risk Assessment
* System Workflow

⸻

Database

The project uses MySQL to store user accounts and assessment records.

Database Name

strokesense

Users Table

users

Main fields:

user_id
full_name
email
phone
password
created_at

Assessments Table

assessments

Main fields:

assessment_id
user_id
assessment_date
gender
age
hypertension
heart_disease
ever_married
work_type
residence_type
avg_glucose_level
bmi
smoking_status
risk_probability
risk_category

⸻

Technologies Used

Programming Language

* Python

Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

Web Development

* Flask
* HTML
* CSS
* JavaScript

Database

* MySQL

Data Visualization

* Matplotlib
* Seaborn
* JavaScript Charts

Development Tools

* Visual Studio Code
* Terminal
* Web Browser
* Python Virtual Environment

⸻

Project Structure

stroke/
│
├── README.md
├── requirements.txt
├── app.py
├── preprocess.py
├── train.py
├── tune.py
├── threshold.py
├── save_model.py
├── predict.py
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_info.json
│
├── results/
│   ├── model_results.csv
│   ├── tuned_model_results.csv
│   └── threshold_results.csv
│
├── templates/
│   ├── index.html
│   ├── account.html
│   ├── register.html
│   ├── assessment.html
│   ├── result.html
│   ├── history.html
│   ├── assessment_details.html
│   ├── dashboard.html
│   ├── analytics.html
│   ├── patients.html
│   ├── reports.html
│   ├── settings.html
│   └── about.html
│
├── static/
│   └── style.css
│
└── venv/

⸻

Installation

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd stroke

2. Create Virtual Environment

python3 -m venv venv

3. Activate Virtual Environment

source venv/bin/activate

4. Install Requirements

pip install -r requirements.txt

⸻

MySQL Setup

Make sure MySQL is installed and running.

Start MySQL:

brew services start mysql

Open MySQL:

mysql -u root -p

Create the database:

CREATE DATABASE strokesense;
USE strokesense;

Create the users table:

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    phone VARCHAR(20) NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

Create the assessments table:

CREATE TABLE assessments (
    assessment_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    assessment_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    gender VARCHAR(20),
    age FLOAT,
    hypertension INT,
    heart_disease INT,
    ever_married VARCHAR(10),
    work_type VARCHAR(30),
    residence_type VARCHAR(20),
    avg_glucose_level FLOAT,
    bmi FLOAT,
    smoking_status VARCHAR(30),
    risk_probability FLOAT,
    risk_category VARCHAR(20)
);

⸻

Database Configuration

The Flask application connects to MySQL using:

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="YOUR_MYSQL_PASSWORD",
    database="strokesense"
)

Replace YOUR_MYSQL_PASSWORD with your local MySQL password.

Do not publish your actual MySQL password on GitHub.

⸻

How to Run the Project

Open Terminal and run:

cd ~/stroke

Activate the virtual environment:

source venv/bin/activate

Start MySQL:

brew services start mysql

Run the Flask application:

python app.py

The application will run at:

http://127.0.0.1:5001

Open the above address in your browser.

⸻

Application Workflow

1. Register Account
        ↓
2. Login
        ↓
3. Open Risk Assessment
        ↓
4. Enter Health Information
        ↓
5. Submit Assessment
        ↓
6. Machine Learning Prediction
        ↓
7. Calculate Risk Probability
        ↓
8. Apply Decision Threshold
        ↓
9. Display Risk Category
        ↓
10. Store Assessment in MySQL
        ↓
11. View History / Dashboard / Analytics

⸻

Model Training Workflow

Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Threshold Analysis
   ↓
Final Model
   ↓
Save Model

⸻

Training Commands

Preprocess the dataset:

python preprocess.py

Train the machine learning models:

python train.py

Perform hyperparameter tuning:

python tune.py

Perform threshold analysis:

python threshold.py

Save the final model:

python save_model.py

Run a sample prediction:

python predict.py

These training commands are only required when retraining or updating the machine learning model. For normal application use, run:

python app.py

⸻

Results Files

The results directory contains model evaluation results.

model_results.csv

Contains the initial comparison of the machine learning models.

tuned_model_results.csv

Contains results after hyperparameter tuning.

threshold_results.csv

Contains results for different decision thresholds.

⸻

Security

The application includes user authentication.

Passwords are stored using password hashing instead of plain-text passwords.

Users can access their assessment history after logging in.

Protected pages require user authentication.

⸻

Future Enhancements

Possible future improvements include:

* Larger and more diverse healthcare datasets
* Additional health-related features
* Improved model calibration
* Additional machine learning algorithms
* PDF assessment reports
* Cloud deployment
* Improved database security
* Model monitoring
* Periodic model retraining

⸻

Health Disclaimer

StrokeSense provides a machine-learning-based preliminary risk estimate using the information entered by the user.

The result should not be interpreted as a medical diagnosis.

If a person experiences possible stroke warning signs or other emergency symptoms, immediate professional medical attention should be sought.


⸻

Project Information

Project Title: StrokeSense: ML-Driven Stroke Risk Prediction

Domain: Machine Learning / Healthcare

Project Type: Academic Project

Degree: B.Tech – Computer Science and Engineering

⸻

Conclusion

StrokeSense demonstrates the integration of machine learning, web development, and database technologies to provide preliminary stroke-risk estimation based on selected health parameters.

The project combines:

Machine Learning
       +
Data Preprocessing
       +
Flask Web Application
       +
MySQL Database
       +
User Authentication
       +
Dashboard & Analytics

StrokeSense is developed as an academic decision-support and preliminary risk-assessment system and does not replace professional medical diagnosis.
