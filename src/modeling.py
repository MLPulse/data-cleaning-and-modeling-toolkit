import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

def train_model(train_df, selected_features, results_data_dir):
    """Train a Decision Tree Regressor and log results to MLflow."""
    X = train_df[selected_features]
    y = train_df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    
    mlflow.set_experiment("decision-tree-regressor-experiment")
    
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

    mlflow.log_metric("rmse", rmse_val)
    mlflow.sklearn.log_model(model, "Decision Tree Regression Model")

    return model

def test_model(model, test_df, selected_features, results_data_dir):
    """Test the model on the test data and log results to MLflow."""
    X_test = test_df[selected_features]
    y_pred_test = model.predict(X_test)

    results = pd.DataFrame({
        'id': test_df['id'],
        'predicted': y_pred_test
    })
    results.to_csv(f'{results_data_dir}/test_predictions.csv', index=False)
    mlflow.log_artifact(f'{results_data_dir}/test_predictions.csv')
m
