# Appliances-Energy-Prediction

## Project Description
This project aimed to predict household energy consumption by exploring various regression models. The dataset, sourced from Kaggle, included temperature, humidity, and energy data for different appliances in a house, with additional weather parameters from a nearby station. The focus areas were the kitchen, laundry, and living room.

## Technologies and Tools
- R: Used for data preprocessing, exploratory data analysis (EDA), and predictive modeling.
- Dataset: Sourced from Kaggle, containing temperature, humidity, and energy data for different appliances, along with additional weather parameters.

## Variable Introduction and Definitions
- Appliances: Energy use in Wh.
- Lights: Energy use of light fixtures in the house in Wh.
- Temperature and Humidity Variables: Measured for different rooms and outdoor areas.
  
## Implementation

### Data Preprocessing
- Missing Value Handling: Addressed missing values in the dataset.
- Data Transformation: Applied transformations to normalize and scale the data.
- Outlier Removal: Identified and removed outliers to ensure data quality.
### Exploratory Data Analysis (EDA)
- Visualization: Created visualizations to understand the distributions and relationships between variables.
- Handling Degenerate Predictors: Removed predictors with little to no variance.
- Skewness Adjustment: Applied transformations to correct skewed data distributions.

### Predictive Modeling
- Model Selection: Trained and evaluated various regression models, both linear and non-linear.
- Principal Component Regression: Achieved an RMSE of 0.138 and an R-squared value of 0.658.
- Logistic Regression: Achieved an RMSE of 0.166 and an R-squared value of 0.1.

# Conclusion
Exploratory Data Analysis: Conducted a thorough EDA to understand the data and its underlying patterns.
Data Cleaning: Addressed missing data, degenerate predictors, skewness, and outliers.
Model Training and Evaluation: Trained various regression models and evaluated their performance using RMSE and R-squared metrics.
Top-Performing Models: Principal Component Regression and Logistic Regression emerged as the best models.
