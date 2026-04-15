# messy_car_data

# Used Car Price Prediction - Data Cleaning & Baseline

This notebook performs initial data exploration, systematic cleaning, and computes a simple baseline model (mean prediction) using MAE for the used car dataset.

### Tasks Completed:
- **Task 1**: Explored the dataset using `df.info()`, `df.describe()`, and `df.shape`. Identified missing values, duplicates, inconsistent text, and non-numeric mileage.
- **Task 2**: Cleaned the data by dropping null target rows, imputing missing features, standardizing the `brand` column, extracting numeric values from `mileage`, and removing duplicates.
- **Task 3**: Built a baseline model that predicts the mean `selling_price` for every record and calculated its MAE on the cleaned dataset.

**Key Takeaway**: The baseline MAE serves as a benchmark. Any real ML model must aim to achieve a lower MAE than this to add value.

Submitted as part of Masai School ML Curriculum
