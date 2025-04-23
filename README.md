# Medical Diagnosis using Machine Learning

This folder contains code and resources for a medical diagnosis application using machine learning. The application is built using Flask (a Python web framework) and several machine learning models to predict the likelihood of various diseases.

##  Files and Directories

* `app.py`: This is the main Flask application file. It handles the routing and logic for the web application, including loading the machine learning models and making predictions based on user input[cite: 1].
* `cure_data.json`: This file likely contains data related to cures or treatments for the diagnosed diseases.
* `requirements.txt`: This file lists the Python packages required to run the application.
* `data/`: This directory contains the datasets used for training the machine learning models.
    * `Heart Disease Dataset.csv`: Dataset for heart disease classification.
    * `indian_liver_patient.csv`: Dataset for liver disease classification[cite: 73].
    * `kidney_disease.csv`: Dataset for kidney disease classification[cite: 145].
* `notebooks/`: This directory contains Jupyter Notebooks used for developing and training the machine learning models.
    * `Breast_Cancer_Detection_patched.ipynb`: Notebook for breast cancer detection.
    * `Heart Disease Classification_patched.ipynb`: Notebook for heart disease classification.
    * `Kidney Disease Classification_patched.ipynb`: Notebook for kidney disease classification.
    * `Liver Disease Classifiaction_patched.ipynb`: Notebook for liver disease classification.
* `static/css/`: This directory contains the CSS stylesheet (`style.css`) for the web application's front-end[cite: 1].
* `templates/`: This directory contains the HTML templates for the web application.
    * `cure.html`:  Template for displaying cure/treatment information.
    * `home.html`: Main landing page of the web application.
    * `result.html`: Template to display the diagnosis results.
    * `select_disease.html`:  Template for selecting the disease to diagnose.

##  Datasets

The application uses the following datasets, located in the `data/` directory:

* **Heart Disease Dataset**:  This dataset is used for classifying the likelihood of heart disease[cite: 1].
* **Indian Liver Patient Dataset**: This dataset is used for classifying liver disease[cite: 73].
* **Kidney Disease Dataset**: This dataset is used for classifying kidney disease[cite: 145].

##  Machine Learning Models

The machine learning models are developed and trained in the Jupyter Notebooks within the `notebooks/` directory.  The notebooks cover:

* **Breast Cancer Detection**:  The notebook (`Breast_Cancer_Detection_patched.ipynb`) focuses on detecting breast cancer.
* **Heart Disease Classification**:  The notebook (`Heart Disease Classification_patched.ipynb`) focuses on classifying heart disease[cite: 1, 46, 47].
* **Kidney Disease Classification**: The notebook (`Kidney Disease Classification_patched.ipynb`) focuses on classifying kidney disease.
* **Liver Disease Classification**: The notebook (`Liver Disease Classifiaction_patched.ipynb`) focuses on classifying liver disease[cite: 2, 3, 4].

##  Web Application

The web application is built using Flask, with HTML templates for the front-end and CSS for styling[cite: 1].

* **Flask Application (`app.py`)**:  The core of the web application, handling user requests, processing data, and displaying results[cite: 1].
* **HTML Templates (`templates/`)**:  These templates define the structure and content of the web pages.
* **CSS Styling (`static/css/style.css`)**:  This stylesheet provides the visual styling for the web pages.

##  Requirements

The `requirements.txt` file (currently empty) would typically list the necessary Python packages. You can install them using pip:

```bash
pip install -r requirements.txt
