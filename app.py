from flask import Flask, render_template, request
import joblib
import json

app = Flask(__name__)

# Load models
disease_models = {
    "breast_cancer": joblib.load("models/breast_cancer.pkl"),
    "liver_disease": joblib.load("models/Liver_disease.pkl"),
    "heart_disease": joblib.load("models/Heart_Disease.pkl"),
    "kidney_disease": joblib.load("models/Kidney_Disease.pkl"),
}

try:
    with open("cure_data.json", "r") as f:
        treatment_data = json.load(f)
except FileNotFoundError:
    print("Error: cure_data.json not found!")
    treatment_data = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select_disease')
def select_disease():
    return render_template('select_disease.html')

def predict_breast_cancer():
    fields = ["clump_thickness", "uniform_cell_size", "uniform_cell_shape", "marginal_adhesion", "single_epithelial_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses"]
    data = [int(request.form[field]) for field in fields]
    result = disease_models['breast_cancer'].predict([data])[0]
    output = "Benign (Non-cancerous)" if result == 2 else "Malignant (Cancerous)"

    features_table = [
        {"name": "Clump Thickness", "range": "1 - 4", "value": data[0], "min_threshold": 0, "max_threshold": 4},
        {"name": "Uniformity of Cell Size", "range": "1 - 4", "value": data[1], "min_threshold": 0, "max_threshold": 4},
        {"name": "Uniformity of Cell Shape", "range": "1 - 4", "value": data[2], "min_threshold": 0, "max_threshold": 4},
        {"name": "Marginal Adhesion", "range": "1 - 4", "value": data[3], "min_threshold": 0, "max_threshold": 4},
        {"name": "Single Epithelial Cell Size", "range": "1 - 4", "value": data[4], "min_threshold": 0, "max_threshold": 4},
        {"name": "Bare Nuclei", "range": "1 - 4", "value": data[5], "min_threshold": 0, "max_threshold": 4},
        {"name": "Bland Chromatin", "range": "1 - 4", "value": data[6], "min_threshold": 0, "max_threshold": 4},
        {"name": "Normal Nucleoli", "range": "1 - 4", "value": data[7], "min_threshold": 0, "max_threshold": 4},  # FIXED
        {"name": "Mitoses", "range": "1 - 4", "value": data[8], "min_threshold": 0, "max_threshold": 4},
    ]
    
    return "Breast Cancer", output, features_table


def predict_liver_disease():
    fields = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Proteins", "Albumin", "Albumin_and_Globulin_Ratio"]
    data = [float(request.form[field]) for field in fields]
    result = disease_models['liver_disease'].predict([data])[0]
    output = "Liver Disease Detected" if result == 1 else "No Liver Disease"

    features_table = [
    {"name": "Total Bilirubin", "range": "0.1 - 1.2 mg/dL", "value": data[2], "min_threshold": 0.1, "max_threshold": 1.2},
    {"name": "Direct Bilirubin", "range": "0.0 - 0.3 mg/dL", "value": data[3], "min_threshold": 0.0, "max_threshold": 0.3},
    {"name": "Alkaline Phosphotase", "range": "44 - 147 IU/L", "value": data[4], "min_threshold": 44, "max_threshold": 147},
    {"name": "Alamine Aminotransferase", "range": "7 - 56 IU/L", "value": data[5], "min_threshold": 7, "max_threshold": 56},
    {"name": "Aspartate Aminotransferase", "range": "10 - 40 IU/L", "value": data[6], "min_threshold": 10, "max_threshold": 40},
    {"name": "Total Proteins", "range": "6.3 - 7.9 g/dL", "value": data[7], "min_threshold": 6.3, "max_threshold": 7.9},
    {"name": "Albumin", "range": "3.5 - 5.0 g/dL", "value": data[8], "min_threshold": 3.5, "max_threshold": 5.0},
    {"name": "A/G Ratio", "range": "0.8 - 2.0", "value": data[9], "min_threshold": 0.8, "max_threshold": 2.0},
    ]

    return "Liver Disease", output, features_table

def predict_heart_disease():
    fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    data = [float(request.form[field]) for field in fields]
    result = disease_models['heart_disease'].predict([data])[0]
    output = "Heart Disease Detected" if result == 1 else "No Heart Disease"

    features_table = [
        {"name": "Age", "range": "29 - 50 yrs", "value": data[0], "min_threshold": 29, "max_threshold": 50},
        {"name": "Chest Pain Type", "range": "1 - 2 (Mild)", "value": data[2], "min_threshold": 1, "max_threshold": 2},
        {"name": "Serum Cholesterol", "range": "126 - 200 mg/dL", "value": data[4], "min_threshold": 126, "max_threshold": 200},
        {"name": "Resting ECG", "range": "0 (Normal)", "value": data[6], "min_threshold": 0, "max_threshold": 0},
        {"name": "Exercise-Induced Angina", "range": "0 (No)", "value": data[8], "min_threshold": 0, "max_threshold": 0},
        {"name": "Slope of ST Segment", "range": "1 (Normal)", "value": data[10], "min_threshold": 1, "max_threshold": 1},
        {"name": "Thalassemia", "range": "3 (Normal)", "value": data[12], "min_threshold": 3, "max_threshold": 3},
        {"name": "Resting Blood Pressure", "range": "<120/80 mmHg", "value": data[3], "min_threshold": None, "max_threshold": 120},
        {"name": "Fasting Blood Sugar", "range": "<100 mg/dL", "value": data[5], "min_threshold": None, "max_threshold": 100},
        {"name": "Max Heart Rate", "range": "120 - 200 bpm", "value": data[7], "min_threshold": 120, "max_threshold": 200},
        {"name": "Old Peak (ST Depression)", "range": "0 - 1 mm", "value": data[9], "min_threshold": 0, "max_threshold": 1},
        {"name": "Major Vessels (Fluoroscopy)", "range": "0 - 1", "value": data[11], "min_threshold": 0, "max_threshold": 1},
    ]

    return "Heart Disease", output, features_table


def predict_kidney_disease():
    fields = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad", "appet", "pe", "ane"]
    data = [float(request.form[field]) for field in fields]
    result = disease_models['kidney_disease'].predict([data])[0]
    output = "Kidney Disease Detected" if result == 1 else "No Kidney Disease"

    features_table = [
        {"name": "Age", "range": "Varies", "value": data[0], "min_threshold": 0, "max_threshold": None},
        {"name": "Blood Pressure (BP)", "range": "<120/80 mmHg", "value": data[1], "min_threshold": 80, "max_threshold": None},
        {"name": "Specific Gravity (SG)", "range": "1.005 - 1.030", "value": data[2], "min_threshold": 1.005, "max_threshold": 1.030},
        {"name": "Albumin (AL)", "range": "0 - 5", "value": data[3], "min_threshold": 0, "max_threshold": 5},
        {"name": "Sugar (SU)", "range": "0 (No sugar)", "value": data[4], "min_threshold": 0, "max_threshold": None},
        {"name": "Red Blood Cells (RBC)", "range": "None", "value": data[5], "min_threshold": 0, "max_threshold": None},
        {"name": "Pus Cells (PC)", "range": "0 - 5 per HPF", "value": data[6], "min_threshold": 0, "max_threshold": 5},
        {"name": "Pus Cell Clumps (PCC)", "range": "None", "value": data[7], "min_threshold": 0, "max_threshold": None},
        {"name": "Bacteria (BA)", "range": "None", "value": data[8], "min_threshold": 0, "max_threshold": None},
        {"name": "Blood Glucose Random (BGR)", "range": "70 - 140 mg/dL", "value": data[9], "min_threshold": 70, "max_threshold": 140},
        {"name": "Blood Urea (BU)", "range": "Varies", "value": data[10], "min_threshold": 0, "max_threshold": None},
        {"name": "Serum Creatinine (SC)", "range": "0.6 - 1.2 mg/dL", "value": data[11], "min_threshold": 0.6, "max_threshold": 1.2},
        {"name": "Sodium (Na+)", "range": "135 - 145 mEq/L", "value": data[12], "min_threshold": 135, "max_threshold": 145},
        {"name": "Potassium (K+)", "range": "3.5 - 5.0 mEq/L", "value": data[13], "min_threshold": 3.5, "max_threshold": 5.0},
        {"name": "Hemoglobin (Hb)", "range": "13 - 17 g/dL (Men), 12 - 15 g/dL (Women)", "value": data[14], "min_threshold": 12, "max_threshold": 17},
        {"name": "Packed Cell Volume (PCV)", "range": "40 - 50% (Men), 35 - 45% (Women)", "value": data[15], "min_threshold": 35, "max_threshold": 50},
        {"name": "WBC Count", "range": "4,000 - 11,000 cells/μL", "value": data[16], "min_threshold": 4000, "max_threshold": 11000},
        {"name": "RBC Count", "range": "4.2 - 6.1 million/μL", "value": data[17], "min_threshold": 4.2, "max_threshold": 6.1},
        {"name": "Hypertension (HTN)", "range": "0 (No)", "value": data[18], "min_threshold": 0, "max_threshold": None},
        {"name": "Diabetes Mellitus (DM)", "range": "0 (No)", "value": data[19], "min_threshold": 0, "max_threshold": None},
        {"name": "Coronary Artery Disease (CAD)", "range": "0 (No)", "value": data[20], "min_threshold": 0, "max_threshold": None},
        {"name": "Appetite", "range": "0 (Good)", "value": data[21], "min_threshold": 0, "max_threshold": None},
        {"name": "Pedal Edema (PE)", "range": "0 (No)", "value": data[22], "min_threshold": 0, "max_threshold": None},
        {"name": "Anemia (ANE)", "range": "0 (No)", "value": data[23], "min_threshold": 0, "max_threshold": None},
    ]

    return "Kidney Disease", output, features_table


@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']

    if disease == 'breast_cancer':
        disease_name, result, features_table = predict_breast_cancer()
    elif disease == 'liver_disease':
        disease_name, result, features_table = predict_liver_disease()
    elif disease == 'heart_disease':
        disease_name, result, features_table = predict_heart_disease()
    elif disease == 'kidney_disease':
        disease_name, result, features_table = predict_kidney_disease()
    else:
        return "Invalid disease selection", 400

    if "No" not in result:  # If the disease is detected
        return render_template('result.html', disease=disease_name, result=result, features=features_table, show_cure=True)
    else:
        return render_template('result.html', disease=disease_name, result=result, features=features_table, show_cure=False)

@app.route('/cure')
def cure():
    disease = request.args.get('disease', '').lower().replace(" ", "_")  # Normalize disease name
    treatment_info = treatment_data.get(disease, {"title": "No information available", "description": "", "methods": []})

    return render_template('cure.html', disease=disease.replace("_", " ").title(), treatment=treatment_info)


if __name__ == '__main__':
    app.run(debug=True)
