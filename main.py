import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Load data features into X and labels into Y
df = pd.read_csv('salaries.csv')


X = df[['Age', 'War/Game21', 'War/Game19/20']]
df['War/Game19/20'] = df['War/Game19/20'] * .67
y = df[['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

print(X_train)
print(X_test)

model = LinearRegression();
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# Visualize actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlim(2, 35)  # Set the x-axis limits
plt.ylim(2, 35)  # Set the y-axis limits



plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs. Predicted Salary')
plt.show()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
