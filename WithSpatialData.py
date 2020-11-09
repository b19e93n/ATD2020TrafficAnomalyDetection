#version 5.0 trying to add spatial data and multiply by a coefficient
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import itertools
from pathlib import Path
import multiprocessing
import matplotlib
import time

font = {"family": "normal", "weight": "normal", "size": 20}
matplotlib.rc("font", **font)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

pd.options.mode.chained_assignment = None

import atd2020

day_of_the_week = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
}
weekdays = set(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
weekends = set(["Saturday", "Sunday"])
day_reverse = {day_of_the_week[key]: key for key in day_of_the_week}
    
def AddData(df, df_hour, df_station, df_paired_station, rank, day_in, hour_in):
    if rank == 0: #weekend: add the other weekend. weekday: add adjacent weekdays
        if day_in in range(1, 6):
            add_day = [day_in % 5 + 1, (day_in - 2) % 5 + 1]
            add_day_rv = [day_reverse[i] for i in add_day]
            concat_df = df_hour[df_hour["Weekday"].isin(add_day_rv)]
            return (pd.concat([df, concat_df]), rank + 2)
        else:
            df["Rank"] = [0 for i in range(len(df))]
            if day_in == 0:
                add_day = [6]
            else:
                add_day = [0]
            add_day_rv = [day_reverse[i] for i in add_day]
            concat_df = df_hour[df_hour["Weekday"].isin(add_day_rv)]
            return (pd.concat([df, concat_df]), rank + 4)
    if rank == 1: 
        #concat_df = pd.concat([df, df.sample(frac = 0.5)])
        #return pd.concat([df, concat_df]), rank + 3
        return df, rank + 3
    if rank == 2: #weekday add second adjacent days
        add_day = [(day_in + 2) % 5, (day_in - 3) % 5 + 1]
        add_day_rv = [day_reverse[i] for i in add_day]
        concat_df = df_hour[df_hour["Weekday"].isin(add_day_rv)]
        return pd.concat([df, concat_df]), rank + 1
    if rank == 3: #weekday add rest of the weekdays
        concat_df = df_hour[df_hour["Weekday"].isin(weekdays ^ set(df["Weekday"]))]
        return pd.concat([df, concat_df]), rank + 1
    if rank == 4: #weekend + weekday, adding adjacent station
        if day_in in range(1,6):
            df_paired_weekday = df_paired_station[df_paired_station["Weekday"].isin(weekdays)]
            concat_df = df_paired_weekday[df_paired_weekday["Hour"] == hour_in]
            mean1 = np.mean(concat_df["TotalFlow"])
            mean2 = np.mean(df["TotalFlow"])
            concat_df["TotalFlow"] = mean1 / mean2 * concat_df["TotalFlow"]
        else:
            df_paired_weekday = df_paired_station[df_paired_station["Weekday"].isin(weekends)]
            concat_df = df_paired_weekday[df_paired_weekday["Hour"] == hour_in]
            mean1 = np.mean(concat_df["TotalFlow"])
            mean2 = np.mean(df["TotalFlow"])
            concat_df["TotalFlow"] = mean1 / mean2 * concat_df["TotalFlow"]
        return pd.concat([df, concat_df]), rank + 1
            
    if rank == 5: #weekend + weekday, adding adjacent hour
        if day_in in range(1, 6):
            add_hour = [(hour_in + 1) % 24, (hour_in - 1) % 24]
            concat_hour = df_station[(df_station["Hour"].isin(add_hour))]
            concat_df = concat_hour[(concat_hour["Weekday"].isin(weekdays))]
            return pd.concat([df, concat_df]), rank + 1
        else:
            add_hour = [(hour_in + 1) % 24, (hour_in - 1) % 24]
            concat_hour = df_station[(df_station["Hour"].isin(add_hour))]
            concat_df = concat_hour[(concat_hour["Weekday"].isin(weekends))]
            
            #return pd.concat([df, pd.concat([concat_df, concat_df, concat_df.sample(frac=0.5)])]), rank + 1
            return pd.concat([df, concat_df]), rank + 1
        
    if rank == 6: #weekend + weekday, adding next adjacent hour
        if day_in in range(1, 6):
            add_hour = [(hour_in + 2) % 24, (hour_in - 2) % 24]
            concat_hour = df_station[(df_station["Hour"].isin(add_hour))]
            concat_df = concat_hour[(concat_hour["Weekday"].isin(weekdays))]
            return pd.concat([df, concat_df]), rank + 1
        else:
            add_hour = [(hour_in + 2) % 24, (hour_in - 2) % 24]
            concat_hour = df_station[(df_station["Hour"].isin(add_hour))]
            concat_df = concat_hour[(concat_hour["Weekday"].isin(weekends))]
            
            #return pd.concat([df, pd.concat([concat_df, concat_df, concat_df.sample(frac=0.5)])]), rank + 1
            return pd.concat([df, concat_df]), rank + 1
            
    if rank == 7: #weekend: do nothing. Weekday: adding adjacent hour from the rest of the days
        if day_in in range(1, 6):
            add_hour = [(hour_in + 3) % 24, (hour_in - 3) % 24]
            concat_hour = df_station[(df_station["Hour"].isin(add_hour))]
            concat_df = concat_hour[(concat_hour["Weekday"].isin(weekdays))]
            return pd.concat([df, concat_df]), rank + 1
        else:
            add_hour = [(hour_in + 3) % 24, (hour_in - 3) % 24]
            concat_hour = df_station[(df_station["Hour"].isin(add_hour))]
            concat_df = concat_hour[(concat_hour["Weekday"].isin(weekends))]
            
            #return pd.concat([df, pd.concat([concat_df, concat_df, concat_df.sample(frac=0.5)])]), rank + 1
            return pd.concat([df, concat_df]), rank + 1
    
    if rank == 8: 
            hour_ls = [(hour_in + 3) % 24, (hour_in + 2) % 24, (hour_in + 3) % 24, (hour_in + 1) % 24, hour_in, (hour_in - 1) % 24, (hour_in - 2) % 24, (hour_in - 3) % 24]
            
            return df_hour[df_hour['Hour'].isin(hour_ls)], rank + 1
    
        
def FindThreshold(df, df_station, df_paired_station, df_hour, day_in, hour_in, add_data=False, n_th=0, d_th=0, max_rank=0):
    # If we choose to add data
    if add_data:
        curr_rank = 0
        while len(df) < n_th:
            df, curr_rank = AddData(df, df_hour, df_station, df_paired_station, curr_rank, day_in, hour_in)
            if curr_rank > max_rank:
                break
        mean = np.mean(df["TotalFlow"])
        stdev = np.std(df["TotalFlow"], ddof = 0)
        dis = stdev * 3
        return [mean - dis, mean + dis]
    # If we do not add data
    else:
        mean = np.mean(df["TotalFlow"])
        stdev = np.std(df["TotalFlow"], ddof = 0)
        side, gap = CalGap(anomaly, normal)
        if gap < d_th * stdev:
            if side == "up":
                dis = max(normal["TotalFlow"]) - mean + gap / 2
            else:
                dis = mean - min(normal["TotalFlow"]) + gap / 2
        else:
            dis = stdev * 3
        return [mean - dis, mean + dis]
        
def MultiCore(observed_station, paired_station, n_th):
    weekdays = {
        "Sunday":0,
        "Monday":1,
        "Tuesday":2,
        "Wednesday":3,
        "Thursday":4,
        "Friday":5,
        "Saturday":6,
    }
    timestamps, ids, outs = [], [], []
    for hour in range(24):
        observed_hour = observed_station[observed_station["Hour"] == hour]
        for weekday in weekdays.keys():
            observed_weekday = observed_hour[observed_hour["Weekday"] == weekday]
            thresh = FindThreshold(
                observed_weekday,
                observed_station,
                paired_station,
                observed_hour,
                day_in=weekdays[weekday],
                hour_in=hour,
                add_data=True,
                n_th=n_th,
                d_th=0.5,
                max_rank=8
            )
            timestamps_group, ids_group, outs_group = Test_per_set(observed_weekday, thresh)
            timestamps.extend(timestamps_group)
            ids.extend(ids_group)
            outs.extend(outs_group)        
    return timestamps, ids, outs

def CreateStationPairs(df):
    stations = df[["ID", "Latitude", "Longitude"]].drop_duplicates()
    pairs = {}
    for i in stations.index:
        distance = float('Inf')
        for j in stations.index:
            if stations['ID'][i] != stations['ID'][j]:
                d = (stations["Latitude"][i] - stations["Latitude"][j]) ** 2 + (stations["Longitude"][i] - stations["Longitude"][j]) ** 2
                if d < distance:
                    distance = d
                    station = stations["ID"][j]
        pairs[stations["ID"][i]] = station
    return pairs
            
def Test_per_set(df, thresholds):
    timestamps = []
    ids = []
    outs = []
    df = df.reset_index()
    for i in range(len(df)):
        if df["TotalFlow"][i] > thresholds[0] and df["TotalFlow"][i] < thresholds[1]:
            out = 'FALSE'
        else:
            out = 'TRUE'
        timestamps.append(df["Timestamp"][i])
        ids.append(df['ID'][i])
        outs.append(out)
    return timestamps, ids, outs
            
if __name__ == '__main__':
    day_of_the_week = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }
    
    filename = Path("data/City1.parquet.brotli")
    data = atd2020.utilities.read_data(filename)
    data.reset_index(drop=True, inplace=True)
    d_d = atd2020.detrend.detrend(data[data['Observed'] == True])
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    observed_d_d = d_d
    
    for n_th in [10, 25, 50, 100]:
        count = 0
        output_file = open('results/City1_nth%d_new5.csv'%n_th, 'w+')
        print('Timestamp' + ',' + 'ID' + ',' + 'Anomaly', end = '\n', file = output_file)
        for fraction in [0.01, 0.02, 0.05, 0.10, 0.20]:
            stations = list(set(observed_d_d[observed_d_d["Fraction_Observed"] == fraction]["ID"]))
            pairs = CreateStationPairs(observed_d_d[observed_d_d["Fraction_Observed"] == fraction])

            x = [(observed_d_d[observed_d_d['ID'] == station], observed_d_d[observed_d_d['ID'] == pairs[station]], n_th) for station in stations]
            for (timestamps, ids, outs) in pool.starmap(MultiCore, x):
                count+= 1
                print('[%d/%d]'%(count, 500), end = '\r')
                for j in range(len(timestamps)):
                    print(timestamps[j].strftime("%m/%d/%Y %H:%M:%S"), ids[j], outs[j], sep = ',', end = '\n', file = output_file)
            print()
        output_file.close()
    print('finish!')
    exit()
