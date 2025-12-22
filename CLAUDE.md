This project it to predict nexct earthquake. 
The idea is 
    pull data from any source either realtime or close to realtime, 
    predict next eartquake by using AI model 
    show result in web page 
    if event occurss/not occurs update succes rat on web page

The idea for AI model is this
the embedding model embeds earthquake overy year, month, x(latitude), y(longitude), m(magnitude), d(depth) and dt(time differnce between 2 earthquakes).
( I saw some pattern over month and year cycle)

    self.yr_embed = nn.Embedding(self.yr_size, n2_embed)    # 16
    self.mt_embed = nn.Embedding(self.mt_size, n2_embed)    # 16 
    self.x_embed  = nn.Embedding(self.x_size, n2_embed)     # 16
    self.y_embed  = nn.Embedding(self.y_size, n2_embed)     # 16
    self.m_embed  = nn.Embedding(self.m_size, n2_embed)     # 16
    self.d_embed  = nn.Embedding(self.d_size, n2_embed)     # 16
    self.t_embed  = nn.Embedding(self.t_size, n2_embed)     # 96
Use complex valued Embedding and attention layers

web folder is to represent all result
data folder is to save trainig data
database is yysql and connection paramters are in config.py file
pulling data from usgs code and save is at database class
