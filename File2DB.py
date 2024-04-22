import os
import mysql.connector

files = os.listdir("data/dates")
i = 0
mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  password="yourpassword",
  database="mydatabase"
)

mycursor = mydb.cursor()

for a_file in files:
    f_name = "data/dates/" + a_file
    print(f_name)
    with open(f_name, 'r') as f:
        for line in f:
            c_line = line.split("\t")
            c_no = c_line[0]
            c_kod = c_line[1]
            c_date = c_line[2]
            c_time = c_line[3]
            c_lat = c_line[4]
            c_lon = c_line[5]
            c_der = 0 if c_line[6] == "" else c_line[6]
            c_xm = 0 if c_line[7] == "" else c_line[7]
            c_md = 0 if c_line[8] == "" else c_line[8]
            c_ml = 0 if c_line[9] == "" else c_line[9]
            c_mw = 0 if c_line[10] == "" else c_line[10]
            c_ms = 0 if c_line[11] == "" else c_line[11]
            c_mb = 0 if c_line[12] == "" else c_line[12]
            c_tip = c_line[13]
            c_yer = c_line[14]
            print(f"c_ml : {c_ml}")
            print(f"c_mw : {c_mw}")
            print("Yer : " + c_yer)
            sql = (
                "insert into deprem(dp_kodu, dp_date, dp_time, dp_lat, dp_lon, dp_depth, dp_xm, dp_md, dp_ml, dp_mw, "
                "dp_ms, dp_mb, dp_tip, dp_yer) VALUES ")
            sql += (f"('{c_kod}', '{c_date}','{c_time}', {c_lat}, {c_lon}, {c_der}, {c_xm}, {c_md}, {c_ml}, "
                    f"{c_mw}, {c_ms}, {c_mb}, '{c_tip}', '{c_yer}' )")
            print(sql)
            # mycursor.execute(sql)
