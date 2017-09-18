import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

url = "https://www.quandl.com/api/v3/datasets/CHRIS/CME_CL1.csv"
wticl1 = pd.read_csv(url, index_col=0, parse_dates=True)
wticl1.sort_index(inplace=True)
wticl1_last = wticl1['Last']
wticl1['PctCh'] = wticl1.Last.pct_change()

fig = plt.figure(figsize=[7,5])
ax1 = plt.subplot(111)
line = wticl1_last.plot(color='red',linewidth=3)
ax1.set_ylabel('USD per barrel')
ax1.set_xlabel('')
ax1.set_title('WTI Crude Oil Price', fontsize=18)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.tick_params(axis='x', which='major', labelsize=8)
plt.savefig('oil.png', dpi=1000)
plt.show()


