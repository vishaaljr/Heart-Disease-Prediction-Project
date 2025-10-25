# Heart Disease Prediction

## Project Overview

This repository contains a machine learning model for predicting heart disease based on various health factors. The model is trained on a dataset containing information such as age, sex, cholesterol levels, blood pressure, and other relevant medical data. This project aims to provide a tool for preliminary risk assessment, which can assist in making informed decisions about healthcare.

## Key Features & Benefits

*   **Predictive Modeling:** Utilizes machine learning algorithms to predict the likelihood of heart disease.
*   **Data-Driven Insights:** Leverages historical health data to identify key risk factors.
*   **Easy to Use:** The pre-trained model and associated code make it straightforward to generate predictions.
*   **Educational Resource:** Provides a practical example of applying machine learning to healthcare.

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Python:** (Version 3.6 or higher)
*   **pip:** Python package installer

The following Python libraries are required. They can be installed using `pip`.

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/vishaaljr/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage Examples

1.  **Running the Jupyter Notebook:**

    Open the `Heart_diseases_prediction.ipynb` notebook using Jupyter Notebook or JupyterLab. Follow the instructions within the notebook to load the data, train the model (if necessary), and generate predictions.

    ```bash
    jupyter notebook Heart_diseases_prediction.ipynb
    ```

2.  **Using the Pre-trained Model:**

    The repository includes a pre-trained model (`heart_model.pkl`). You can load this model in your Python script to make predictions on new data.

    ```python
    import pandas as pd
    import numpy as np
    import pickle

    # Load the pre-trained model
    with open('heart_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Example data (replace with your own data)
    data = {'age': [67], 'sex': [1], 'cp': [4], 'trestbps': [120], 'chol': [229], 'fbs': [0], 'restecg': [2], 'thalach': [129], 'exang': [1], 'oldpeak': [2.6], 'slope': [2], 'ca': [2], 'thal': [7]}
    df = pd.DataFrame(data)

    # Make a prediction
    prediction = model.predict(df)

    print(f"Heart Disease Prediction: {prediction}") # 0 means no heart disease, 1 means heart disease
    ```

## Configuration Options

There are no specific configuration files in this project.  The model parameters were pre-defined during development of the `Heart_diseases_prediction.ipynb` notebook. Fine-tuning of model parameters can be performed in the notebook, then the `heart_model.pkl` file can be regenerated.

## Contributing Guidelines

We welcome contributions to improve this project! Please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix:** `git checkout -b feature/your-feature-name`
3.  **Make your changes and commit them with descriptive commit messages.**
4.  **Push your changes to your forked repository:** `git push origin feature/your-feature-name`
5.  **Submit a pull request to the main branch of this repository.**



