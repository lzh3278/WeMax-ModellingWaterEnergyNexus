# Random Forest Regression by Zonghan Li in SEA Group

# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

# Data import
df = pd.read_csv('/Users/zonghanli/Library/CloudStorage/OneDrive-个人/SEA/202302-annual水消费/'
                 'db-energy-water-hdtz19.csv')
'''df = df.fillna(0)'''
df.head()
df.info()
y = df.iloc[:, 25]
x = df.iloc[:, 7:24]

# Create performance list
performance = pd.DataFrame(columns=('i', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE', 'importance'))

# Splitting train and test sets
RandomState = 1
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RandomState)

# Define the grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_features': [0.3, 0.5, 0.7],
    'max_depth': [1, 3, 5, 7, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
rfr = RandomForestRegressor(random_state=RandomState)

# Grid searching
grid_search = GridSearchCV(rfr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Fitting the model using grid search results
rfr = RandomForestRegressor(random_state=RandomState, **best_params)
rfr.fit(X_train, y_train)
train_sizes, train_scores, test_scores = learning_curve(rfr, x, y, cv=10)

# Learning curve
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
fig = plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Score of train set')
plt.plot(train_sizes, test_mean, label='Score of test set')
plt.xlabel('n_Samples')
plt.ylabel('Scores')
plt.legend()
plt.show()

for RandomSeed in range(0, 10000):
    rfr = RandomForestRegressor(random_state=RandomSeed, **best_params)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

    # Calculating importance of the features
    importance = rfr.feature_importances_
    feat_labels = x.columns[0:]
    indices = np.argsort(importance)[::-1]
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))

    # Model Performance
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = metrics.r2_score(y_test, y_pred)
    MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
    performance = performance.append(pd.DataFrame({'i': [RandomSeed], 'MAE': [MAE], 'MSE': [MSE],
                                                   'RMSE': [RMSE], 'R2': [R2], 'MAPE': [MAPE],
                                                   'importance': [importances]}))

# Figuring y_test and y_pred
y_test_fig = np.array(y_test)
plt.plot(y_test_fig, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
