import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime

start = datetime(1990, 1, 1)
end = datetime(2018, 1, 1)
df = web.DataReader(['GOOG', 'AAPL', 'MSFT', 'SNE', 'BAC', 'DAL', 'NVDA', 'TM'], 'yahoo', start, end)

os.makedirs('data')

for key in df.minor_axis:
    df2 = df.minor_xs(key).dropna()
    df2.to_pickle(os.path.join('data', key + '.pickle'))
