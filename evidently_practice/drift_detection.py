import pandas as pd
import numpy as np
from evidently import Report, Dataset
from evidently.presets import DataDriftPreset


# -------------------------------------------Step 1: Create two simple Pandas DataFrames
# Reference Data (Simulating Training Data Distribution)
# Let's create a dataset with 1000 samples and 3 features.
# Feature 'numerical_feature_1' will be normally distributed.
# Feature 'numerical_feature_2' will have a slightly different distribution.
# Feature 'categorical_feature_1' will be categorical.

np.random.seed(42) # for reproducibility

reference_data = pd.DataFrame({
    'numerical_feature_1': np.random.normal(loc=50, scale=10, size=1000),
    'numerical_feature_2': np.random.normal(loc=100, scale=20, size=1000),
    'categorical_feature_1': np.random.choice(['A', 'B', 'C'], size=1000, p=[0.5, 0.3, 0.2]),
    'target': np.random.randint(0, 2, size=1000) # A dummy target column
})

print("--- Reference Data Info ---")
print(reference_data.head())
print(reference_data.describe())
print("\n")

# Current Data (Simulating New Production Data with Drift)
# We will introduce drift in 'numerical_feature_1' (shifted mean)
# and 'numerical_feature_2' (increased variance).
# Also, change the distribution of 'categorical_feature_1'.

current_data = pd.DataFrame({
    'numerical_feature_1': np.random.normal(loc=55, scale=10, size=1000), # Mean shifted
    'numerical_feature_2': np.random.normal(loc=100, scale=30, size=1000), # Variance increased
    'categorical_feature_1': np.random.choice(['A', 'B', 'C', 'D'], size=1000, p=[0.3, 0.3, 0.2, 0.2]), # New category 'D' and changed proportions
    'target': np.random.randint(0, 2, size=1000) # Dummy target
})

print("--- Current Data Info ---")
print(current_data.head())
print(current_data.describe())
print("\n")


# -----------------------------------Step 2: Generate a Data Drift Report
# We use the 'Report' class from evidently, and pass a list of 'Metric Presets'.
# DataDriftPreset is a pre-built set of metrics specifically for detecting data drift.
# It automatically applies various statistical tests and visualizations.

report = Report(metrics=[
    DataDriftPreset(),
])

# Run the report, comparing current_data to reference_data
data_drift_report = report.run(current_data=current_data, reference_data=reference_data)

#When run() is executed, Evidently performs all the calculations defined within the DataDriftPreset 
# and stores the results, including the visualizations, within the data_drift_report object.



# ---------------------------------------------Step 3: Examine the Report (Display and Save)


# To save the report as an HTML file (highly recommended for sharing or offline viewing):
report_file_name = "data_drift_report.html"
data_drift_report.save_html(report_file_name)
print(f"--- Data Drift Report saved to {report_file_name} ---")


