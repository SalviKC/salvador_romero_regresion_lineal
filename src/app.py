import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score


data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")

data_uniques = data.drop_duplicates()

data_uniques["sex"] = pd.factorize(data_uniques["sex"])[0]
data_uniques["smoker"] = pd.factorize(data_uniques["smoker"])[0]
data_uniques["region"] = pd.factorize(data_uniques["region"])[0]

col = ["age", "bmi", "smoker", "charges"]
new_data = data_uniques[col]

scaler = MinMaxScaler()
scal_features = scaler.fit_transform(new_data[col])
df_scal = pd.DataFrame(scal_features, index = new_data.index, columns = col)
df_scal["charges"] = new_data["charges"]

X = df_scal.drop("charges", axis=1)
Y = df_scal["charges"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=8)

model = LinearRegression()

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = r2_score(y_test, y_pred)
print(score)