import requests
import pandas as pd
import mysql.connector
import datetime
import csv
import numpy as np
import torch
import config


class DataC():
    def __init__(self):        
        self.yr_max = 55
        self.mt_max = 12
        self.x_max = 180
        self.y_max = 360
        self.m_max =  91
        self.d_max =  75
        self.t_max = 150
        self.data = []
        self.train = []
        self.valid = []
        self.n = 0.8

        self.mydb = mysql.connector.connect(
            host=config.DB_HOST, 
            user=config.DB_USER, 
            password=config.DB_PASS, 
            database=config.DB_NAME
        )
        
        self.mycursor = self.mydb.cursor()        
        self.fname =  "./data/latest.csv"
        
    def db2File(self, min_mag):
        min_magnitude = 2
        # sql = "SELECT year(us_datetime)-1970 as year, month(us_datetime) as mont, us_x, us_y, cast( (us_mag*10) as signed) as us_m, us_d, case when us_t>9000 then 150 else cast(us_t/60 as SIGNED) end as us_t FROM usgs WHERE	us_mag>1.99 and us_type = 'earthquake' and us_magtype like 'm%' order by  us_datetime asc "
        # sql + sql + " INTO OUTFILE 'data/result0.txt' FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' "
        sql = f"CALL get_data({min_mag})"    # quakes grater than min_mag by experiment (4.9)
        self.mycursor.execute(sql)
        rows = self.mycursor.fetchall()

        cols = [i[0] for i in self.mycursor.description]

        fp = open(self.fname, 'w')
        myFile = csv.writer(fp, lineterminator = '\n')
        myFile.writerow(cols)
        myFile.writerows(rows)
        fp.close()
        self.mydb.close()
        
    def usgs2DB(self):    
        self.mycursor.execute("delete from usgs_tmp")

        min_magnitude = 2
        hdr = "insert into usgs_tmp(us_code, us_date, us_time, us_datetime, us_lat, us_lon, us_dep, us_mag, us_place, us_type, us_magType, us_x, us_y, us_d, us_m) VALUES "

        for i in range(3, 0, -1):
            start_time = f"now-{i}days"
            end_time =  f"now-{(i-1)}days"
            path = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_time}&endtime={end_time}&minmagnitude={min_magnitude}"
            url = requests.get(path)
            dataset = url.json()
            features = dataset['features']

            sql = ""
            for feature in features:
                code = feature['id']
                mag = feature['properties']['mag']
                if mag==None:
                    mag = 0
                plc = feature['properties']['place']
                if plc!= None:
                    plc = plc.replace("'", "").replace('"',"")

                dtm = pd.to_datetime(feature['properties']['time'], unit='ms').strftime('%y/%m/%d %H:%M:%S')
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
                d = dep/10
                m = mag
                n_sql = hdr +  f"( '{code}', '{dat}', '{tim}', '{dtm}', {lat}, {lon}, {dep}, {mag}, '{plc}', '{typ}', '{magType}', '{x}', '{y}', '{d}', '{m*10}' )"
                try:
                    self.mycursor.execute(n_sql)
                    print(f"executed in  {i} {plc} {mag} ")
                except:
                    print(f"execution in error  :  {i} ")
            self.mydb.commit()        
        try:
            res = self.mycursor.execute("call ins_quakes(); ") # new eartguakes are inserting to usgs table
            print(f"executed out {res} ")
        except:
            print(f"error out  ")

        self.mycursor.execute("SELECT us_id, us_mag, us_place FROM usgs WHERE us_t is null")
        rows =  self.mycursor.fetchall()
        for row in rows:
            # print(row)
            id = row[0]
            sql = f"call calc_dt({id})"
            self.mycursor.execute(sql)
        self.mydb.commit();
        
    def getData(self):
        
        df1 = pd.read_csv(self.fname)
        df1.fillna(0, inplace=True)
        self.data = np.array(df1.values)
        n = int(self.n*len(self.data))
        self.train = self.data[:n]
        self.valid = self.data[n:]
        # print(f"&& data size  {len(self.data ):,}")
        # print(f"&& train size {len(self.train):,}")
        # print(f"&& valid size {len(self.valid):,}")
        
        self.yr_max = self.data[:, 0].max() + 1  # year 0 to 54
        self.mt_max = self.data[:, 1].max() + 1 
        self.x_max  = self.data[:, 2].max() + 1 
        self.y_max  = self.data[:, 3].max() + 1 
        self.m_max  = self.data[:, 4].max() + 1 
        self.d_max  = self.data[:, 5].max() + 1 
        self.t_max  = self.data[:, 6].max() + 1 
        
        return self.data
        
    def getSizes(self):
        if (len(self.data)<1):
            print("Array size, please call getData first")
            return "Error " 
        # print(f" Yr min:  {self.data[:,0].min()} max : {self.data[:,0].max()}")
        # print(f" Mt min:  {self.data[:,1].min()} max : {self.data[:,1].max()}")
        # print(f" x  min:  {self.data[:,2].min()} max : {self.data[:,2].max()}")
        # print(f" y  min:  {self.data[:,3].min()} max : {self.data[:,3].max()}")
        # print(f" m  min:  {self.data[:,4].min()} max : {self.data[:,4].max()}")
        # print(f" d  min:  {self.data[:,5].min()} max : {self.data[:,5].max()}")
        # print(f" t  min:  {self.data[:,6].min()} max : {self.data[:,6].max()}")
        sizes = {
            'yr_size' : self.yr_max, 
            'mt_size' : self.mt_max, 
            'x_size' : self.x_max, 
            'y_size' : self.y_max, 
            'm_size' : self.m_max, 
            'd_size' : self.d_max, 
            't_size' : self.t_max
        }
        return sizes    

    def getScaledData(self):
        data = self.getData()
        # I checked data and values up until this point
        #
        # year [0 54] Some years are active then others
        # month [1 12] some months are active than other, July is the most active month for sure
        # lat [6 177] most activity around 130 degree (range is [0 180])
        # lon [0 360] most activity around  70 degree (range is [0 360])
        # mag [0 9] 2,3,4 dominate the data
        # dep [0 70] 1 and 10 dominates
        # dt  [0 150] in minutes, nice curve
        self.yr_max = data[:, 0].max()  # year 0 to 54
        self.mt_max = data[:, 1].max()  
        self.x_max  = data[:, 2].max()  
        self.y_max  = data[:, 3].max()  
        self.m_max  = data[:, 4].max()  
        self.d_max  = data[:, 5].max()  
        self.t_max  = data[:, 6].max()
        return data
    
    def getBatch(self, B, T, split, col):
        
        if (len(self.data)<1):            
            self.getData()

        data_ = self.train if split=='train' else self.valid
        
        dataT = torch.from_numpy(data_[:, col]) # latitude
        data_ = torch.from_numpy(data_)

        ix = torch.randint(len(data_)-T, (B,))
        # print(f"ix {ix}")
        # ix = range(B)

        x = torch.stack([data_[i:i+T    ] for i in ix])
        y = torch.stack([dataT[i+1:i+T+1] for i in ix])

        # print(f"x shape {x.shape} y shape {y.shape}")
        # print(f"x data  {x}")
        # print(f"y data  {y}")
        # print(f"{1/0}")
        # x => B, T, F
        # y => B, T

        return x.long(), y.long()
        

    def getLast(self, B, T, split, col):
        
        if (len(self.data)<1):            
            self.getData()

        data_ = self.train if split=='train' else self.valid
                
        data_ = torch.from_numpy(data_)

        ix = torch.randint(len(data_)-T, (B,))
        
        # ix = range(B)
        # print(f"IX IX {ix}")
    

        x = torch.stack([data_[i:i+T] for i in ix])
        y = torch.stack([data_[i+T:i+T+1] for i in ix])
        # print(f"INFERENCE X {x.shape} {y.shape}")


        y = y[-1, :, col]
        # print(f"INFERENCE X {x.shape} {y.shape}")
        # print(f"INFERENCE X {y}")
        # print(f"{1/0}")
        
        return x.long(), y.long()
    

    def closeAll(self):
        self.mydb.close()
        