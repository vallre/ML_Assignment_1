import panda as pd
import pandas_profiling

# load data
flight_df = pd.read_csv("../data/flight_delay.csv")
# create the data profile report
pandas_profiling.profile_report.ProfileReport(flight_df).to_file("../docs/original data profile.html")