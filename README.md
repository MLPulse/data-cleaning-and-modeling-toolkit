# Data Cleaning and Modeling Toolkit

## Overview
This project provides a comprehensive toolkit for data cleaning, visualization, and machine learning model training. It is designed to handle common data preprocessing tasks such as missing data imputation, feature scaling, encoding, and model evaluation. The toolkit also supports multiple machine learning models and offers visualization tools for better data understanding.

## Features
- **Data Cleaning**: Handles missing data by visualizing and imputing missing values.
- **Feature Engineering**: Provides tools for scaling, encoding, and selecting features.
- **Modeling**: Includes methods to train and evaluate machine learning models (e.g., Random Forest, XGBoost, Stacking).
- **Visualization**: Offers data visualization tools to explore missing data, feature importance, and more.

## Directory Structure
```bash
data-cleaning-and-modeling-toolkit/
│
├── data/                  # Directory for raw and processed data
│   ├── raw/               # Raw data files (e.g., Train.csv, Test.csv)
│   ├── processed/         # Processed or cleaned data files
│
├── src/                   # Source code directory
│   ├── __init__.py        # Initialize the module
│   ├── data_processing.py # Data cleaning and processing scripts
│   ├── modeling.py        # Machine learning model training and evaluation
│   └── visualization.py   # Data visualization functions
│
├── notebooks/             # Jupyter notebooks for EDA and experiments
│   └── data_analysis.ipynb
│
├── tests/                 # Unit tests for the code
│   └── test_data_processing.py
│
├── requirements.txt       # Required Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Files to be ignored by Git
```

## Installation

### 1. Clone the repository:
```bash
   git clone https://github.com/your-username/data-cleaning-and-modeling-toolkit.git
   cd data-cleaning-and-modeling-toolkit
```

### 2.Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3.Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1.Data Processing
```bash
from src.data_processing import filter_dataframe_by_missing_data, create_imputation_models, impute_data

# Load raw data
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# Filter data by missing values
filtered_train_df = filter_dataframe_by_missing_data(train_df, threshold=25)

# Create imputation models
categorical_imputer, label_encoders, regressors, categorical_cols, numeric_cols = create_imputation_models(filtered_train_df)

# Impute missing values in the test dataset
imputed_test_df = impute_data(test_df, categorical_imputer, label_encoders, regressors, categorical_cols, numeric_cols)
```

### 2.Model Training
```bash
from src.modeling import train_model, test_model

# Train the model
trained_model = train_model(train_df_processed, selected_features, 'data/processed')

# Test the model
test_model(trained_model, test_df_processed, selected_features, 'data/processed')
```

### 3.Visualizing Missing Data
```bash
from src.visualization import visualize_missing_data

# Visualize missing data in the train set
visualize_missing_data(train_df)
```

## File Structure

- **data/**: Contains raw and processed data files.
- **src/**: Source code for data processing, model training, and visualization.
- **notebooks/**: Jupyter notebooks for experiments and exploration.
- **tests/**: Unit tests for the codebase.
- **requirements.txt**: Lists all the required dependencies.
- **.gitignore**: Specifies which files and directories Git should ignore.
- **README.md**: Provides an overview and usage instructions for the project.
- **LICENSE**: Contains the license information for the project (MIT by default).




