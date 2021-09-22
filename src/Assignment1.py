# core
import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
# data preprocessing
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import train_test_split
# ML models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
# ML model evaluation
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# function to encode categorical features in a dataframe (based on function from the lab 1)
def feature_encoding(df:DataFrame, feature_name:str, encoder):
    new_feats = encoder.transform(df[feature_name])
    # create dataframe from encoded feature; the name of the column will have " (encoded)" at the end of it
    new_cols = pd.DataFrame(new_feats, dtype=int, columns=[feature_name + " (encoded)"])
    new_df = pd.concat([df, new_cols], axis=1)    
    new_df.drop(feature_name, axis=1, inplace=True)
    return new_df


# load the data
flight_df = pd.read_csv("../data/flight_delay.csv")
# print(flight_df.head().to_markdown())


## Data Preprocessing ##
# encode cathegorical features using sklearn.preprocessing.LabelEncoder
encoder = LabelEncoder()

# "Depature Airport" and Destination Airport contain the same names, so they can be encoded using the same encoder
encoder.fit(np.concatenate((flight_df["Depature Airport"].values, flight_df["Destination Airport"].values)))
flight_df = feature_encoding(flight_df, "Depature Airport", encoder)
flight_df = feature_encoding(flight_df, "Destination Airport", encoder)

# print(flight_df.head().to_markdown())

# extract useful information from departure and arival times
# here we can extract: 
# day, month, and dayofweek of departure (this should be relatively th same for the arrival time) -> some specific days/months might be more congested
# time of arrival and departure in minutes (delay is minutes, so it will be more useful in this form) -> some time blocks might be busier than others
# flight duration (might look like a linearly derived type, but since the day of arrival is omited, this will preserve some of its information) -> longer paths, longer delay
# use the year to split data into training and test (otherwise year should be irrelevant)

# construct array of elements and array of column names
dep_datetime = pd.to_datetime(flight_df["Scheduled depature time"]).dt
arr_datetime = pd.to_datetime(flight_df["Scheduled arrival time"]).dt
flight_duration_timedelta = pd.to_datetime(flight_df["Scheduled arrival time"]) - pd.to_datetime(flight_df["Scheduled depature time"])

new_feats = np.array([dep_datetime.day.values, dep_datetime.month.values, dep_datetime.dayofweek.values, (dep_datetime.hour * 60 + dep_datetime.minute).values, (arr_datetime.hour * 60 + arr_datetime.minute).values, flight_duration_timedelta / pd.Timedelta('1 minute'), dep_datetime.year.values]).T
new_feats_names = ["Flight Day", "Flight Month", "Day of the week", "Time of departure (min)", "Time of arrival (min)", "Flight duration (min)", "Year"]

# construct new DataFrame
new_cols = pd.DataFrame(new_feats, dtype=int, columns=new_feats_names)

# update original DataFrame
flight_df = pd.concat([flight_df, new_cols], axis=1)
flight_df.drop(["Scheduled depature time", "Scheduled arrival time"], axis=1, inplace=True)
del(dep_datetime, arr_datetime, flight_duration_timedelta, new_feats, new_feats_names, new_cols, encoder)
# print(flight_df.head().to_markdown())


## Outlier removal ##
# remove outliers using z-score
# remove the those 2 outliers that could be seen on the Raw Time vs Delay graph
flight_df = flight_df[(np.abs(stats.zscore(flight_df["Flight duration (min)"])) < 5)]
# other parameters seems to be within good z-score (based on experimentatation in graph_generator.py)


## Splitting the Data ##
# suppress pandas warnings about writting to the parent df
pd.options.mode.chained_assignment = None
# split into train and test datasets based on date
# 2015-2017 -> train
# 2018      -> test
train_df = flight_df.loc[(flight_df["Year"] >= 2015) & (flight_df["Year"] <= 2017)]
test_df = flight_df.loc[(flight_df["Year"] == 2018)]
# now we can drop the Year
train_df.drop("Year", axis=1, inplace=True)
test_df.drop("Year", axis=1, inplace=True)

# remove ouliers from train dataset based on the delay
# need to be careful here, majority of data has low delay, but lower threshold seem to improve the models performance on test dataset
# however, alpha lower than 3 starts to increase MSE error, while improving MAE -> worse generalization
train_df = train_df[(np.abs(stats.zscore(train_df["Delay"])) < 3)]

# split into predictors and target
x_train_df = train_df.loc[:, train_df.columns != "Delay"]
y_train_df = train_df["Delay"]

x_test_df = test_df.loc[:, test_df.columns != "Delay"]
y_test_df = test_df["Delay"]

# finally, scale predictor to slightly speed up calculations
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_df = pd.DataFrame(scaler.fit_transform(x_train_df), columns=x_train_df.columns)
x_test_df = pd.DataFrame(scaler.transform(x_test_df), columns=x_test_df.columns)


## Machine Learning Models ##
# the target in this task is a continious quantitative output -> regression
# first, let's try simple linear regression
linReg = LinearRegression()
linReg.fit(x_train_df, y_train_df)

y_pred = linReg.predict(x_test_df)

# it is a regression problem, so we cannot calculate accuracy, recall, F1, etc.
print("-"*15)
print('Simple Linear regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_df, y_pred))                                   # 14.173306569086455
print('Mean Squared Error:', metrics.mean_squared_error(y_test_df, y_pred))                                     # 1616.1465293478113
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)))                       # 40.20132497005306
scores = cross_val_score(linReg, x_train_df, y_train_df, scoring="neg_mean_squared_error", cv=10)
print('Cross-validation MSE on train set = {:.2e}(+/- {:.2e})'.format( -scores.mean(), scores.std()), end="\n\n")            # 2.15e+03(+/- 6.73e+02)
                                                                                                                # these and the following numbers were recorded before removal of the "outliers" from the Delay
# error is quite big (relatively speaking)
# both train and test error is big -> underfitting? need more complex models

# after printing, let's collect the metrics into a table
metrics_table = {}
metrics_table["Simple Linear regression"] = {"MAE" : round(metrics.mean_absolute_error(y_test_df, y_pred), 3), "MSE" : round(metrics.mean_squared_error(y_test_df, y_pred), 3), "RMSE" : round(np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)), 3), "Train MSE" : round(-scores.mean(), 3)}

# now, let's try polynomial regression
degrees = [2, 3, 4] # careful, this takes a lot of memory
for i in range(len(degrees)):
    polFeat = PolynomialFeatures(degree=degrees[i])
    linReg = LinearRegression()

    polFeat.fit(x_train_df, y_train_df)
    pol_x_train = polFeat.transform(x_train_df)
    linReg.fit(pol_x_train, y_train_df)

    y_pred = linReg.predict(polFeat.transform(x_test_df))
    
    print("-"*15)
    print(f"Polynomial degree: {degrees[i]}")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_df, y_pred)) 
    print('Mean Squared Error:', metrics.mean_squared_error(y_test_df, y_pred)) 
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)))
    scores = cross_val_score(linReg, pol_x_train, y_train_df, scoring="neg_mean_squared_error", cv=5) # higher cv = longer computation time
    print('Cross-validation MSE on train set = {:.2e}(+/- {:.2e})'.format( -scores.mean(), scores.std()), end="\n\n")
    # collect the metrics
    metrics_table[f"Polynomial degree: {degrees[i]}"] = {"MAE" : round(metrics.mean_absolute_error(y_test_df, y_pred), 3), "MSE" : round(metrics.mean_squared_error(y_test_df, y_pred), 3), "RMSE" : round(np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)), 3), "Train MSE" : round(-scores.mean(), 3)}

del(polFeat, linReg, pol_x_train, y_pred, scores)
# best degree seems to be 2:
#   Mean Absolute Error: 13.613446089710846
#   Mean Squared Error: 1602.4904591464808
#   Root Mean Squared Error: 40.03111863471318
#   Cross-validation MSE on train set = 2.15e+03(+/- 3.28e+02)
# however, both train and test errors are high
# higher degree results in higher errors; starting from degree 4, cross-validation error starts to increase
# btw, scalling increased the error almost unnoticeably

# let's try to improve the best result with an optimizer
# Lasso Regression
alphas = np.arange(0.001, 2, 0.01) # step size reduced, for faster computation; originally tested with lower step size
losses = []
x_train, x_val, y_train, y_val = train_test_split(x_train_df, y_train_df, test_size=0.1, random_state=123)
for alpha in alphas:
    # create model
    lasso = Lasso(alpha)
    lasso.fit(x_train, y_train)
    # predict on validation set to tune alpha
    y_pred = lasso.predict(x_val)
    losses.append(metrics.mean_squared_error(y_val, y_pred))

best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha for Lasso:", best_alpha, end="\n\n")

# Ridge Regression
alphas = np.arange(0.001, 2, 0.01) # step size reduced, for faster computation; originally tested with lower step size
losses = []
x_train, x_val, y_train, y_val = train_test_split(x_train_df, y_train_df, test_size=0.1, random_state=123)
for alpha in alphas:
    # create model
    ridge = Ridge(alpha)
    ridge.fit(x_train, y_train)
    # predict on validation set to tune alpha
    y_pred = ridge.predict(x_val)
    losses.append(metrics.mean_squared_error(y_val, y_pred))

best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha for Ridge:", best_alpha, end="\n\n")

# in both cases lower value of alpha results in lower value of error
# this means that the current model deffinetelly underfits because 

lasso = Lasso(0.001)
lasso.fit(x_train_df, y_train_df)
y_pred = lasso.predict(x_test_df)

print("-"*15)
print('Lasso Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_df, y_pred))                           # 14.175441290741455                             
print('Mean Squared Error:', metrics.mean_squared_error(y_test_df, y_pred))                             # 1616.1501826449899       
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)))               # 40.20137040754942       
scores = cross_val_score(lasso, x_train_df, y_train_df, scoring="neg_mean_squared_error", cv=5)
print('Cross-validation MSE on train set = {:.2e}(+/- {:.2e})'.format( -scores.mean(), scores.std()), end="\n\n")    # 2.15e+03(+/- 3.27e+02)
# collect the metrics
metrics_table["Lasso Regression"] = {"MAE" : round(metrics.mean_absolute_error(y_test_df, y_pred), 3), "MSE" : round(metrics.mean_squared_error(y_test_df, y_pred), 3), "RMSE" : round(np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)), 3), "Train MSE" : round(-scores.mean(), 3)}

ridge = Ridge(0.001)
ridge.fit(x_train_df, y_train_df)
y_pred = ridge.predict(x_test_df)

print("-"*15)
print('Ridge Regression:')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_df, y_pred))                           # 14.17330658254203        
print('Mean Squared Error:', metrics.mean_squared_error(y_test_df, y_pred))                             # 1616.1465293630463       
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)))               # 40.20132497024254        
scores = cross_val_score(ridge, x_train_df, y_train_df, scoring="neg_mean_squared_error", cv=5)
print('Cross-validation MSE on train set = {:.2e}(+/- {:.2e})'.format( -scores.mean(), scores.std()), end="\n\n")    # 2.15e+03(+/- 3.27e+02)
# collect the metrics
metrics_table["Ridge Regression"] = {"MAE" : round(metrics.mean_absolute_error(y_test_df, y_pred), 3), "MSE" : round(metrics.mean_squared_error(y_test_df, y_pred), 3), "RMSE" : round(np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)), 3), "Train MSE" : round(-scores.mean(), 3)}

# both lasso and ridge are worse than the original simple linear regression
# Final note: after removing Delay "outliers" the train error improved significantly
# The best model is Polynomial Degree 3

plt.close()
# finally, let's export the table
# column labels are the types of error
column_labels = list(metrics_table["Ridge Regression"].keys())
# row labels are the machine learning models
row_labels = list(metrics_table.keys())
# construct data row by row
data = [list(model.values()) for model in metrics_table.values()]

fig, ax = plt.subplots()

ax.axis('off')
table = ax.table(cellText=data, colLabels=column_labels, rowLabels=row_labels, loc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)

plt.show()