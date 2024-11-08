# Machine-Learning
Examine Data | Build Framework | Evaluating and Comparing Regression Models
I'll start by examining the data in the file to understand the available features and structure. After that, I'll help you build a framework for evaluating and comparing regression models based on your outlined objective. Let me load and review the contents of the dataset first.
import pandas as pd

# Load the dataset to examine its structure and contents
file_path = '/content/Student_Performance.csv'
data = pd.read_csv(file_path)

# Display the first few rows and a summary of the dataset to understand its structure
data.head(), data.describe()

The dataset includes the following features:

1.Hours Studied: Continuous numerical feature.
2.Previous Scores: Continuous numerical feature representing past academic scores.
3.Extracurricular Activities: Categorical feature ("Yes" or "No").
4.Sleep Hours: Continuous numerical feature indicating hours of sleep.
5.Sample Question Papers Practiced: Continuous numerical feature.
The target variable is Performance Index, a continuous value indicating student performance.

To proceed, I’ll:
1.Preprocess the data, including encoding categorical features.
2.Split the data into training and test sets.
3.Develop and evaluate different regression models (e.g., Linear Regression, Random Forest regression).
4.Compare models based on accuracy and other metrics to select the best-performing model.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode the categorical variable 'Extracurricular Activities'
data['Extracurricular Activities'] = LabelEncoder().fit_transform(data['Extracurricular Activities'])

# Define features and target variable
X = data.drop(columns=['Performance Index'])
y = data['Performance Index']

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the train and test sets to confirm the split
X_train.shape, X_test.shape, y_train.shape, y_test.shape

The dataset has been split into training and test sets, with 8,000 samples for training and 2,000 for testing. Now, I'll proceed to develop and evaluate several regression models, starting with common choices like Linear Regression and Random Forest regression. This will give us an initial comparison based on metrics such as Mean Absolute Error (MAE) and R-squared (R²).

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize models
linear_reg = LinearRegression()
random_forest = RandomForestRegressor(random_state=42)

# Train models
linear_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_reg.predict(X_test)
y_pred_rf = random_forest.predict(X_test)

# Evaluate models
linear_metrics = {
    "MAE": mean_absolute_error(y_test, y_pred_linear),
    "R2": r2_score(y_test, y_pred_linear)
}

rf_metrics = {
    "MAE": mean_absolute_error(y_test, y_pred_rf),
    "R2": r2_score(y_test, y_pred_rf)
}

linear_metrics, rf_metrics

The initial model evaluations show the following performance:

1.Linear Regression:
Mean Absolute Error (MAE): 1.61
R-squared (R²): 0.989

2.Random Forest Regressor:
Mean Absolute Error (MAE): 1.81
R-squared (R²): 0.986

Both models perform well, with Linear Regression slightly outperforming Random Forest on MAE and R². Based on these results, the Linear Regression model may be a strong candidate for the final model due to its simplicity and competitive performance.

def predict_performance_index(hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_papers_practiced):
    """
    Predicts the Performance Index of a student based on the given features.

    Parameters:
    - hours_studied (float): Number of hours the student studied.
    - previous_scores (float): Student's previous scores.
    - extracurricular_activities (str): "Yes" if the student participates in extracurricular activities, otherwise "No".
    - sleep_hours (float): Number of hours the student sleeps.
    - sample_papers_practiced (float): Number of sample question papers practiced.

    Returns:
    - Predicted Performance Index (float).
    """
    # Encode the extracurricular activities feature
    extracurricular_encoded = 1 if extracurricular_activities.lower() == "yes" else 0

    # Arrange input data for prediction
    input_data = [[hours_studied, previous_scores, extracurricular_encoded, sleep_hours, sample_papers_practiced]]

    # Predict and return the performance index
    predicted_index = linear_reg.predict(input_data)
    return predicted_index[0]

# Example usage: predict the performance index for a new data point
example_prediction = predict_performance_index(6, 85, "Yes", 7, 4)
example_prediction

