# core
import pandas as pd 
import numpy as np
import matplotlib.pylab as plt
# data preprocessing
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from scipy import stats
# ML models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
# let's explore the effect of different alphas
train_df2 = train_df[(np.abs(stats.zscore(train_df["Delay"])) < 2)]
train_df3 = train_df[(np.abs(stats.zscore(train_df["Delay"])) < 3)]
train_df4 = train_df[(np.abs(stats.zscore(train_df["Delay"])) < 4)]
train_df5 = train_df[(np.abs(stats.zscore(train_df["Delay"])) < 5)]

# split into predictors and target for all DataFrames (including dataframe with Delay "outliers")
x_train_df0 = train_df.loc[:, train_df.columns != "Delay"]
y_train_df0 = train_df["Delay"]
x_train_df2 = train_df2.loc[:, train_df2.columns != "Delay"]
y_train_df2 = train_df2["Delay"]
x_train_df3 = train_df3.loc[:, train_df3.columns != "Delay"]
y_train_df3 = train_df3["Delay"]
x_train_df4 = train_df4.loc[:, train_df4.columns != "Delay"]
y_train_df4 = train_df4["Delay"]
x_train_df5 = train_df5.loc[:, train_df5.columns != "Delay"]
y_train_df5 = train_df5["Delay"]

x_test_df = test_df.loc[:, test_df.columns != "Delay"]
y_test_df = test_df["Delay"]

# finally, scale predictor to slightly speed up calculations
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_df0 = pd.DataFrame(scaler.fit_transform(x_train_df0), columns=x_train_df0.columns)
x_test_df = pd.DataFrame(scaler.transform(x_test_df), columns=x_test_df.columns)

x_train_df2 = pd.DataFrame(scaler.transform(x_train_df2), columns=x_train_df2.columns)
x_train_df3 = pd.DataFrame(scaler.transform(x_train_df3), columns=x_train_df3.columns)
x_train_df4 = pd.DataFrame(scaler.transform(x_train_df4), columns=x_train_df4.columns)
x_train_df5 = pd.DataFrame(scaler.transform(x_train_df5), columns=x_train_df5.columns)

## Machine Learning Models ##
# the target in this task is a continious quantitative output -> regression
# here we will explore only the best performing solution -> degree 2 polynomial regression
linReg = LinearRegression()
polFeat = PolynomialFeatures(degree=2)

# it is a regression problem, so we cannot calculate accuracy, recall, F1, etc.
for i in [0, 2, 3, 4, 5]:
    # go through each threshold
    exec(f"polFeat.fit(x_train_df{i}, y_train_df{i})")
    pol_x_train = eval(f"polFeat.transform(x_train_df{i})")
    exec(f"linReg.fit(pol_x_train, y_train_df{i})")

    y_pred = linReg.predict(polFeat.transform(x_test_df))

    # observe the effect
    print("-"*15)
    print("Polynomial degree: 2")
    print(f'z-score threshold: {i}')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test_df, y_pred)) 
    print('Mean Squared Error:', metrics.mean_squared_error(y_test_df, y_pred)) 
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test_df, y_pred)))
    scores = eval(f'cross_val_score(linReg, pol_x_train, y_train_df{i}, scoring="neg_mean_squared_error", cv=5)')
    print('Cross-validation MSE = {:.2e}(+/- {:.2e})'.format( -scores.mean(), scores.std()), end="\n\n")

# finally, plot what is left of the data after outlier removal
# prepare plot data
x1 = x_train_df2["Flight duration (min)"]
x2 = x_train_df3["Flight duration (min)"]
x3 = x_train_df4["Flight duration (min)"]
x4 = x_train_df5["Flight duration (min)"]

y1 = y_train_df2
y2 = y_train_df3
y3 = y_train_df4
y4 = y_train_df5

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig.set_figwidth(30)
fig.set_figheight(10)
fig.suptitle("Flight duration vs Delay after outlier removal", fontsize=16)

ax1.scatter(x1, y1, 8)
ax1.set(xlabel='Flight Duration (min)', ylabel='Delay (min)', title='Flight Duration vs Delay with threshold 2')
ax1.grid()

ax2.scatter(x2, y2, 8, color='g')
ax2.set(xlabel='Flight Duration (min)', ylabel='Delay (min)', title='TFlight Duration vs Delay with threshold 3')
ax2.grid()

ax3.scatter(x3, y3, 8, color='orange')
ax3.set(xlabel='Flight Duration (min)', ylabel='Delay (min)', title='Flight Duration vs Delay with threshold 4')
ax3.grid()

ax4.scatter(x4, y4, 8, color='cyan')
ax4.set(xlabel='Flight Duration (min)', ylabel='Delay (min)', title='Flight Duration vs Delay with threshold 5')
ax4.grid()

fig.savefig("../figures/Effect of Delay outlier removal.png")