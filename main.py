import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data features into X and labels into Y
features = pd.read_csv('salaries.csv')

# Extract player names, features, and labels
players = features['Player']
X = features[['War/Game21', 'War/Game19/20']]
y = features['Salary']

# Split the data
X_train, X_test, y_train, y_test, players_train, players_test = train_test_split(X, y, players, test_size=0.2, random_state=48)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')



# Generate x-values
x_values = np.linspace(0, 35, 100)  # Adjust the range as needed

# Corresponding y-values for y = x
y_values = x_values

# Plot the line y = x
plt.plot(x_values, y_values, color='red', label='y = x')

# Visualize actual vs. predicted values with player names
plt.scatter(y_test, y_pred)

# Set the axis
plt.xlim(0, 35)
plt.ylim(0, 35)

# Add names to the graph
for player, actual, predicted in zip(players_test, y_test, y_pred):
    plt.text(actual, predicted, player, fontsize=8, color='blue')

plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs. Predicted Salary with Player Names')
plt.show()