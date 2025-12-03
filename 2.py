import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Data Loading ---
df = pd.read_csv("car data.csv")

# --- 2. Feature Engineering: Car Age ---
# Calculate the car's age based on the current year
current_year = pd.Timestamp.now().year
df['Car_Age'] = current_year - df['Year']
df.drop('Year', axis=1, inplace=True)

# --- 3. Correlation Analysis and Plotting ---
# Calculate the correlation matrix for numerical features
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_matrix_consolidated.png')
plt.close()

# --- 4. Data Preprocessing: One-Hot Encoding ---
# List categorical columns
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']

# Apply one-hot encoding and drop the first category (to avoid multicollinearity)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop the 'Car_Name' column as it has too many unique values
df_encoded.drop('Car_Name', axis=1, inplace=True)

# --- 5. Model Training and Evaluation ---

# Define features (X) and target (y)
X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Random Forest Regressor Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print("-" * 45)

# --- 6. Feature Importance Analysis and Plotting ---
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances.values, y=sorted_importances.index, hue=sorted_importances.index, palette='viridis', legend=False)
plt.title('Feature Importance for Car Price Prediction (Consolidated)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_consolidated.png')
plt.close()