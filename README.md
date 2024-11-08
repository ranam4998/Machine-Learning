# Machine-Learning
Examine Data | Build Framework | Evaluating and Comparing Regression Models
I'll start by examining the data in the file to understand the available features and structure. After that, I'll help you build a framework for evaluating and comparing regression models based on your outlined objective. Let me load and review the contents of the dataset first.
import pandas as pd

# Load the dataset to examine its structure and contents
file_path = '/content/Student_Performance.csv'
data = pd.read_csv(file_path)

# Display the first few rows and a summary of the dataset to understand its structure
data.head(), data.describe()

