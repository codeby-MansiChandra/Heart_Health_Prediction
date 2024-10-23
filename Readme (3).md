# Heart Health Prediction

This repository contains a Python script and datasets for predicting heart health risk using machine learning.

##Files in the Repository

- `data (2).csv`: Contains health-related data such as blood pressure, cholesterol levels, glucose levels, smoking habits, etc.
- `Data_heart.xlsx`: Contains cardiac-related data including age, sex, chest pain type, cholesterol levels, etc.
- `MainCode.py`: Python script to train an XGBoost classifier and predict heart health risk based on user input.
- Using Machine learning.ipynb 

## Dataset Details

`data (2).csv`

Columns:
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Age
- Gender
- Height
- Weight
- High low-density lipoprotein (LDL) cholesterol
- Low-density lipoprotein (LDL) cholesterol
- Blood Glucose Level
- Active Smoking
- Passive Smoking
- Obesity
- Diet
- Physical Activity
- Result

`Data_heart.xlsx`

## Columns:
- age
- sex
- cp (chest pain type)
- trestbps (resting blood pressure)
- Serum_Cholesterol
- thalach (maximum heart rate achieved)
- restecg (resting electrocardiographic results)
- exang (exercise-induced angina)
- ST depression
- thal
- diagnosis

## Script Details

`MainCode.py`

The script performs the following tasks:
1. **Load Data**: Reads the CSV data file.
    ```python
    df = pd.read_csv("data.csv")
    X = df.drop('Result', axis='columns')
    y = df['Result']
    ```

2. **Split Data**: Splits the data into training and testing sets.
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1)
    ```

3. **Train Model**: Trains an XGBoost classifier model.
    ```python
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    ```

4. **Predict Heart Health Risk**: Predicts the risk based on user input.
    ```python
    patient_data = input("Enter the patients data: ")
    arr = patient_data.split(",")
    arr = [int(x) for x in arr]
    predicted_output = model.predict([arr])
    ```

5. **Output Result**: Prints the prediction result and model accuracy.
    ```python
    if predicted_output[0] == 0:
        print('Heart health is in High risk')
    elif predicted_output[0] == 1:
        print('Heart health is Vulnerable')
    elif predicted_output[0] == 2:
        print('Heart health is Partially prone')
    elif predicted_output[0] == 3:
        print('Heart health is in low risk')
    elif predicted_output[0] == 4:
        print('Heart health is good')
    else:
        print('Invalid data')

    print('Accuracy level of this model:', accuracy)
    ```

## Jupyter Notebook
The Jupyter Notebook HeartHealthPrediction.ipynb provides an interactive way to explore the data, perform data preprocessing, train the model, and visualize the results.

How to Open the Jupyter Notebook
Ensure you have Jupyter Notebook installed. If not, install it using:

sh
Copy code
pip install notebook
Navigate to the directory containing the notebook file:

sh
Copy code
cd path/to/your/repository
Launch Jupyter Notebook:

sh
Copy code
jupyter notebook
In the opened browser window, click on HeartHealthPrediction.ipynb to open and interact with the notebook.

## How to Use

1. Ensure all required packages are installed:
   pip install pandas scikit-learn xgboost

2. Run the script:
   python MainCode.py

3. Follow the prompt to input patient data in the specified format.
License

