import requests
import pandas as pd
import mysql.connector
import datetime
import math
import os

# https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php

mydb = mysql.connector.connect(
  host=os.environ.get("DB_HOST"),
  user=os.environ.get("DB_USER"),
  password=os.environ.get("DB_PASS"),
  database=os.environ.get("DB_NAME")
)

mycursor = mydb.cursor()
min_magnitude = 2
hdr = "insert into usgs(us_code, us_date, us_time, us_datetime, us_lat, us_lon, us_dep, us_mag, us_place, us_type, us_magType, us_x, us_y, us_d, us_m) VALUES "
today = datetime.date.today()
# 19862 days since Jan 1 1970
for i in range(2, 0, -1):
    start_time = f"now-{i*1*10}days"
    end_time =  f"now-{ 1*(i-1)*10}days"
    path = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_time}&endtime={end_time}&minmagnitude={min_magnitude}"
    url = requests.get(path)
    dataset = url.json()
    features = dataset['features']
    print(path)
    sql = ""
    for feature in features:
        code = feature['id']
        mag = feature['properties']['mag']
        if mag==None:
            mag = 0
        plc = feature['properties']['place'].replace("'", "").replace('"',"")
        tmp = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%y/%m/%d %H:%M:%S')

        dat = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%y/%m/%d')
        tim = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%H:%M:%S')
        typ = feature['properties']['type']
        magType = feature['properties']['magType']

        lon = feature['geometry']['coordinates'][0]
        lat = feature['geometry']['coordinates'][1]
        dep = feature['geometry']['coordinates'][2]
        x = lat + 90
        y = lon + 180        
        if dep == None:
            dep = 0
        if dep<0:
            dep = 0
        d = dep
        m = math.trunc(mag)
        sql += f"( '{code}', '{dat}', '{tim}', '{tmp}', {lat}, {lon}, {dep}, {mag}, '{plc}', '{typ}', '{magType}', '{x}', '{y}', '{d}', '{m}' ),"
    n_sql = hdr + sql[:-1]
    # print(n_sql)
    try:
        mycursor.execute(n_sql)
    except:
        print("execution error" + n_sql)
        # exit()
    mydb.commit()

# RUN CALL PREPARE() to prepare data 
print("latters table is clearing...")
sql = "DELETE FROM letters"
mycursor.execute(sql)

print("letters table is recreating...")
sql = "insert into letters(lt_x, lt_y, lt_cnt) select us_x, us_y, count(*) from usgs where us_mag>4 group by 1, 2 order by 1, 2, 3 desc"
mycursor.execute(sql)

print("usgs table us_c fiels is  updating (assigning a letter for each event)...")
sql = "update usgs u set us_c = (select t.us_id from tmp2 t where t.us_x = u.us_x and t.us_y = u.us_y) where u.us_c is null limit 100000"
mycursor.execute(sql)
