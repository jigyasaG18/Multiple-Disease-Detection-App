# Multiple Disease Detection System

## Project Overview
The **Multiple Disease Detection System** aims to leverage cutting-edge machine learning techniques for the early detection and prediction of prevalent diseases such as diabetes, heart disease, and Parkinson's disease. The core objective of this project is to create an intuitive predictive model that helps healthcare professionals make informed decisions while diagnosing these diseases based on various health metrics.

The system is designed to handle input features including but not limited to patient demographics, medical history, and symptoms. By analyzing these features, multiple models train to classify and predict the probability of disease presence effectively.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Directory Structure](#directory-structure)
- [Contributors](#contributors)
- [License](#license)
- [Future Work](#future-work)

## Features
- **Disease Prediction**: Predicts the likelihood of diabetes, heart disease, and Parkinson's disease based on patient data.
- **Model Accuracy Reporting**: Displays detailed accuracy metrics for training and test datasets.
- **Modular Architecture**: Each disease has its own model, allowing for specialized adjustments and enhancements.
- **User-Friendly Interface**: Utilizes Jupyter Notebooks for exploratory data analysis, model training, and predictions.
- **Visualization Tools**: Graphs and charts to visualize model performance and data distributions, making it easier to interpret results.
- **Machine Learning Pipeline**: Includes preprocessing steps like data cleaning and normalization, ensuring that the data fed into the models is of high quality.

## Technologies Used
This project employs a variety of technologies and libraries that are essential for data science and machine learning:

- **Python**: Programming language used for building the project and data manipulation.
- **Pandas**: Library for data manipulation and analysis, particularly useful for tabular data operations.
- **NumPy**: Provides support for numerical operations, particularly for arrays and mathematical functions.
- **scikit-learn**: A machine learning library that provides simple and efficient tools for predictive data analysis including model selection and evaluation.
- **Jupyter Notebook**: Interactive environment for developing and documenting the process of data exploration and model building.
- **Google Colab**: A cloud platform for running Jupyter notebooks, facilitating collaboration and access to additional resources.
- **Joblib**: Used for saving and loading trained model objects to enable easy reuse without retraining.

## Installation
To set up the project on your local machine or the cloud, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/multiple-disease-detection.git
   cd multiple-disease-detection
   ```

2. **Set up a Python virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Run the initial data analysis and model training scripts** found in the notebooks under `colab_files_to_train_models` directory.

## Usage
The Multiple Disease Detection System can be utilized in two main ways: **Training** new models on new data and **Making Predictions** using the existing models.

### Training Models
You can train models on health data with the following steps:
1. Load the dataset from the `dataset` directory.
2. Preprocess the data by handling missing values, encoding categorical features, and normalizing numerical values.
3. Split the data into training and testing sets.
4. Train models (e.g., Logistic Regression, Decision Trees, etc.) and evaluate their performance using accuracy scores.
5. Save the trained models for future use.

#### Example of Training Code Snippet
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
data = pd.read_csv('dataset/diabetes_data.csv')
X = data.drop('Outcome', axis=1)
Y = data['Outcome']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the model
import joblib
joblib.dump(model, 'saved models/diabetes_model.sav')
```

### Making Predictions
To make predictions with pre-trained models:
1. Load the appropriate model from the `saved models` directory.
2. Prepare the new patient data.
3. Pass the data to the model and get predictions.

#### Example of Prediction Code Snippet
```python
# Load the saved model for diabetes prediction
model = joblib.load('saved models/diabetes_model.sav')

# Prepare new patient data
new_data = pd.DataFrame({
    'Feature1': [value1],
    'Feature2': [value2],
    ...
})

# Make predictions
predictions = model.predict(new_data)
print('Predicted Outcome: ', predictions)
```

## Model Performance
The effectiveness of different models has been evaluated on training and test datasets. Here are the accuracy metrics:

### Diabetes Model
- **Training Data Accuracy**: 85.12%
- **Test Data Accuracy**: 81.97%

### Heart Disease Model
- **Training Data Accuracy**: 87.18%
- **Test Data Accuracy**: 82.50%  (example value—replace with actual)

### Parkinson's Disease Model
- **Training Data Accuracy**: 78.34%
- **Test Data Accuracy**: 77.27%

Each model's performance can be improved by conducting hyperparameter tuning or using ensemble methods.

## Directory Structure
Here’s a brief on the main components of the project:

```
.
├── colab_files_to_train_models     # Contains Jupyter notebooks for training models
├── dataset                          # Raw datasets used for training
│   ├── diabetes_data.csv         
│   ├── heart_disease_data.csv      
│   └── parkinsons_data.csv         
├── saved models                     # Directory storing trained models
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
├── requirements.txt                 # List of required libraries
└── README.md                        # Project documentation
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details on usage and distribution.

## Future Work
- **Expand Disease Models**: Train models for additional diseases and improve the current models with more features.
- **Implement User Interface**: Create a web application for easier access to the prediction system.
- **Model Interpretability**: Utilize libraries like SHAP or LIME to understand the decisions made by machine learning models, increasing trust and transparency.
- **Deploy as an API**: Set up a RESTful API for real-time predictions on new data.
- **Integration with Electronic Health Records (EHR)**: Explore avenues for integrating this model into healthcare systems for real-time monitoring and predictions.
