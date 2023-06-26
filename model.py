import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pickle
from xgboost import XGBRegressor

def train_model():
    # Load the Boston Housing dataset
    dataset = pd.read_csv('Boston.csv')

    # Split the data into features (X) and target variable (y)
    X = dataset.drop('medv', axis=1)
    y = dataset['medv']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = XGBRegressor()
    # model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model as a pickle file
    with open('regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    acc_xgb = metrics.r2_score(y_test, y_test_pred)
    return acc_xgb 


if __name__ == '__main__':
    acc_xgb = train_model()
    print(f'Accuracy: {acc_xgb}')
