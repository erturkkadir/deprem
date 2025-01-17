import numpy as np
import pandas as pd
import io
import sys


data = pd.read_csv('data/usgs.txt', header=None, sep='\t', names=['fid', 'id', 'date', 'time', 'lat', 'lon', 'dep', 'mag', 'pla', 'type'])

print(data[0:10])

df2 = pd.DataFrame()

df2['lat'] = data['lat'].values
df2['lon'] = data['lon'].values
df2['dep'] = data['dep'].values
df2['mag'] = data['mag'].values
df2['dt'] =  pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str))
df2['dif'] = -(df2['dt']-df2['dt'].shift(1)).dt.total_seconds().fillna(0)

print(df2)