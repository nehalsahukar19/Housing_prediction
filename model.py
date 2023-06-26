import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

def train_model():
    # Load the Boston Housing dataset
    dataset = pd.read_csv('Boston.csv')

    # Split the data into features (X) and target variable (y)
    X = dataset.drop('medv', axis=1)
    y = dataset['medv']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model as a pickle file
    with open('regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

if __name__ == '__main__':
    mse = train_model()
    print(f'Mean Squared Error: {mse}')
