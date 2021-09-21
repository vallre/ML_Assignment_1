import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from scipy import stats


# function to encode categorical features in a dataframe
def ohe_new_features(df:DataFrame, feature_name:str, encoder):
    new_feats = encoder.transform(df[feature_name])
    # create dataframe from encoded feature; the name of the column will have " (encoded)" at the end of it
    new_cols = pd.DataFrame(new_feats, dtype=int, columns=[feature_name + " (encoded)"])
    new_df = pd.concat([df, new_cols], axis=1)    
    new_df.drop(feature_name, axis=1, inplace=True)
    return new_df


# load the data
flight_df = pd.read_csv("../data/flight_delay.csv")

## Data Preprocessing
# encode cathegorical features using sklearn.preprocessing.LabelEncoder
encoder = LabelEncoder()

# "Depature Airport" and Destination Airport contain the same names, so they can be encoded using the same encoder
encoder.fit(np.concatenate((flight_df["Depature Airport"].values, flight_df["Destination Airport"].values)))
flight_df = ohe_new_features(flight_df, "Depature Airport", encoder)
flight_df = ohe_new_features(flight_df, "Destination Airport", encoder)

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

## Graph generation
# to understand the dependecies, let's plot each parameter against the target (Delay)

# time dependecy
x1 = flight_df["Flight duration (min)"]
x2 = flight_df["Time of departure (min)"]
x3 = flight_df["Time of arrival (min)"]

# date dependency
x4 = flight_df["Day of the week"].replace([0, 1, 2, 3, 4, 5, 6], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
x5 = flight_df["Flight Day"]
x6 = flight_df["Flight Month"].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

# airport station dependecy
x7 = flight_df["Depature Airport (encoded)"]
x8 = flight_df["Destination Airport (encoded)"]

# target
y = flight_df["Delay"]

# first, time depency
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig.set_figwidth(30)
fig.set_figheight(10)
fig.suptitle("Time dependecies of the Delay", fontsize=16)
# remove unused subplot
fig.delaxes(ax4)

ax1.scatter(x1, y, 8)
ax1.set(xlabel='Flight Duration (min)', ylabel='Delay (min)', title='Flight Duration vs Delay')
ax1.grid()

ax2.scatter(x2, y, 8, color='g')
ax2.set(xlabel='Time of departure (min)', ylabel='Delay (min)', title='Time of departure vs Delay')
ax2.grid()

ax3.scatter(x3, y, 8, color='orange')
ax3.set(xlabel='Time of arrival (min)', ylabel='Delay (min)', title='Time of arrival vs Delay')
ax3.grid()

fig.savefig("../figures/Raw Time vs Delay.png")
# plt.show()

# date depency
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig.set_figwidth(30)
fig.set_figheight(10)
fig.suptitle("Date dependecies of the Delay", fontsize=16)
# remove unused subplot
fig.delaxes(ax4)

ax1.scatter(x4, y, 8)
ax1.set(xlabel='Day of the week', ylabel='Delay (min)', title='Day of the week vs Delay')
ax1.grid()

ax2.scatter(x5, y, 8, color='g')
ax2.set(xlabel='Flight Day', ylabel='Delay (min)', title='Flight Day vs Delay')
ax2.grid()

ax3.scatter(x6, y, 8, color='orange')
ax3.set(xlabel='Flight Month', ylabel='Delay (min)', title='Flight Month vs Delay')
ax3.grid()

fig.savefig("../figures/Raw Date vs Delay.png")
# plt.show()

# station depency
fig, (ax1, ax2) = plt.subplots(2, 1)

fig.set_figwidth(30)
fig.set_figheight(10)
fig.suptitle("Airport location dependecies of the Delay", fontsize=16)

ax1.scatter(x7, y, 8)
ax1.set(xlabel='Depature Airport', ylabel='Delay (min)', title='Depature Airport vs Delay')
ax1.grid()

ax2.scatter(x8, y, 8, color='g')
ax2.set(xlabel='Destination Airport', ylabel='Delay (min)', title='Destination Airport vs Delay')
ax2.grid()

fig.savefig("../figures/Raw Place vs Delay.png")
# plt.show()

# amount of data before removal
t = flight_df.shape[0]

# remove outliers using z-score
flight_df = flight_df[(np.abs(stats.zscore(flight_df["Flight duration (min)"])) < 5)]
# other parameters seems to be within good z-score

# Data for plotting: x1, x5, x7, x8
x1 = flight_df["Flight duration (min)"]
x5 = flight_df["Flight Day"]
x7 = flight_df["Depature Airport (encoded)"]
x8 = flight_df["Destination Airport (encoded)"]
y = flight_df["Delay"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

fig.set_figwidth(30)
fig.set_figheight(10)
fig.suptitle("Graphs after outlier removal", fontsize=16)

ax1.scatter(x1, y, 8)
ax1.set(xlabel='Flight Duration (min)', ylabel='Delay (min)', title='Flight Duration vs Delay')
ax1.grid()

ax2.scatter(x5, y, 8, color='g')
ax2.set(xlabel='Flight Day', ylabel='Delay (min)', title='Flight Day vs Delay')
ax2.grid()

ax3.scatter(x7, y, 8, color='orange')
ax3.set(xlabel='Depature Airport', ylabel='Delay (min)', title='Depature Airport vs Delay')
ax3.grid()

ax4.scatter(x8, y, 8, color='cyan')
ax4.set(xlabel='Destination Airport', ylabel='Delay (min)', title='Destination Airport vs Delay')
ax4.grid()

fig.savefig("../figures/Graphs after outlier removal.png")
# plt.show()

# amount of data lost after removal (in %)
print(100 - flight_df.shape[0] / t * 100)