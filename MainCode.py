import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
X = df.drop('Result', axis='columns')
y = df['Result']
ycpy = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.1)
#-----------------------------Accuracy Calculation--------------------------------
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
accuracy= model.score(X_test, y_test)
#-----------------------------Predict value---------------------------------------
patient_data = input("Enter the patients data: ")
arr = patient_data.split(",")
arr = [int(x) for x in arr]  
predicted_output=model.predict([arr])

array_value = predicted_output[0]
if array_value == 0:
    print('Heart health is in High risk')
elif array_value == 1:
    print('Heart health is Vulnerable')
elif array_value == 2:
    print('Heart health is Partially prone')
elif array_value == 3:
    print('Heart health is in low risk')    
elif array_value == 4:
    print('Heart health is good')    
else:
    print('Invalid data')

print('Accuracy level of this model:', accuracy)
