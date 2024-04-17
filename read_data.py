from obspy import read
# import numpy as np
# import matplotlib.pyplot as plt

# st = read("data/20040127_153300-OZALP-(VAN).M=3.6/BNG.SHZ.KO")
# st = read("data/20040128_083800-KEMALIYE-(ERZINCAN).M=3.5/BNN.SHZ.KO")
st = read("data/20040128_083800-KEMALIYE-(ERZINCAN).M=3.5/ERZ.SHZ.KO")
station = st[0].stats.station
sta = st[0].stats.starttime
end = st[0].stats.endtime
chn = st[0].stats.channel
print(sta)
print(end)
print(st[0].stats)

data = st[0].data

# arr = np.array(data, dtype='int')
# plt.plot(arr)
# plt.show()


st.plot(color='gray', tick_format='%I:%M %p',
        starttime=st[0].stats.starttime,
        endtime=st[0].stats.starttime+20)
print(data)
print(len(data))

st[0].spectrogram(log=True)
