# XGBoost Regression by Zonghan Li in SEA Group

# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Model training
def train_model(X_train, y_train):
    # Define the model
    xgbr = xgb.XGBRegressor(random_state=RandomState)

    # Define the grid
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'alpha': list(range(0, 6)),
        'lambda': list(range(0, 6)),
        'gamma': [i / 10.0 for i in range(0, 7)],
        'eta':[0.1, 0.3, 0.5, 0.7, 0.9],
        'subsample': [i / 100.0 for i in range(70, 96, 5)],
        'colsample_bytree': [i / 100.0 for i in range(70, 96, 5)],
        'max_depth': list(range(3, 10)),
        'min_child_weight': list(range(1, 6)),
        'learning_rate': [0.1, 0.01, 0.03, 0.05, 0.07, 0.09]
    }

    # Define K-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=RandomState)

    # Grid searching and K-fold validation
    grid_search = GridSearchCV(xgbr, param_grid=param_grid, n_jobs=-1, scoring='neg_mean_squared_error', cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    xgbr = xgb.XGBRegressor(random_state=RandomState, **best_params, early_stopping_rounds=10, eval_metric="rmse")
    xgbr.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    return xgbr, grid_result, best_params, grid_result.best_estimator_

def evaluate_over_fitting(xgbr, X_train, y_train, X_test, y_test):
    y_train_pred = xgbr.predict(X_train)
    y_pred = xgbr.predict(X_test)
    # 计算训练集和测试集的均方根误差
    RMSE_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    if RMSE > RMSE_train:
        print("Model is overfitting")
    else:
        print("Model is not overfitting")
    return y_train_pred, y_pred, RMSE_train, RMSE

def predict(repeat_time, best_params, X_train, y_train, X_test, y_test):
    # Create performance list
    performance = pd.DataFrame(columns=('i', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'importance'))

    # Repeat the model
    for RandomSeed in range(0, repeat_time):
        xgbr = xgb.XGBRegressor(random_state=RandomSeed, **best_params)
        xgbr.fit(X_train, y_train)
        y_pred = xgbr.predict(X_test)

        # Calculating importance of the features
        importance = xgbr.feature_importances_
        feat_labels = x.columns[0:]
        indices = np.argsort(importance)[::-1]

        # If you need to display the results, use the following two lines
        for f in range(X_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))

        # Evaluating model performance
        MAE = metrics.mean_absolute_error(y_test, y_pred)
        MSE = metrics.mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        R2 = metrics.r2_score(y_test, y_pred)
        MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
        performance = performance.append(pd.DataFrame({'i': [RandomSeed], 'MAE': [MAE], 'MSE': [MSE],
                                                       'RMSE': [RMSE], 'R2': [R2], 'MAPE': [MAPE],
                                                       'importance': [importance]}))

        # Figuring y_test and y_pred
        y_test_fig = np.array(y_test)
        plt.plot(y_test_fig, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.show()

    return xgbr, y_pred, performance, y_test_fig

# Data import
df = pd.read_csv('db-energy-water-hdtz19.csv')
df.head()
df.info()
y = df.iloc[:, 25]
x = df.iloc[:, 7:24]

# Splitting train and test sets
RandomState = 1
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RandomState)

# Model results
xgbr, grid_result, best_params, grid_result.best_estimator_ = train_model(X_train, y_train)
y_train_pred, y_pred, RMSE_train, RMSE = evaluate_over_fitting(xgbr, X_train, y_train, X_test, y_test)
repeat_time = 100
xgbr, y_pred, performance, y_test_fig = predict(repeat_time, best_params, X_train, y_train, X_test, y_test)
