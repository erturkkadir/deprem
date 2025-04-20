# deprem
Transformer mimarisi ile deprem tahmini
1. Collecting Data
    i. get data from usgs server to mysql db --> 8ubcomment line 48 @ app.py
    ii. saves into csv file --> uncomment line 49 @ app.py (set min mag to filter)
    iii. close all open connection


How to start
 
git clone https://github.com/erturkkadir/deprem.git
cd deprem

python -m venv venv

source venv/bin/activate

run the app to start training

python app.py

more to come...