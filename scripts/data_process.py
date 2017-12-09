import pandas as pd
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
import pickle


def create_data():
    url = "data/CME_CL1.csv"
    crude_oil = pd.read_csv(url, index_col=0, parse_dates=True)
    crude_oil.sort_index(inplace=True)
    crude_oil_last = crude_oil['Last']

    param = {
            'q': 'XOM',
            'i': 86400,
            'x': "NYSE",
            'p': '40Y'
    }

    try:
        print("Retrieving data")
        df = get_price_data(param)
        df.set_index(df.index.normalize(), inplace=True)
        stock_close = df['Close']
        pickle.dump(stock_close, open("data/stock_close.p", "wb"))
    except:
        print("Error in retrieving data... Loading previous saved stock data.")
        stock_close = pickle.load(open("data/stock_close.p", "rb"))

    oil_price, stock_price = crude_oil_last.align(stock_close, join='inner')

    split_index = int(3*len(oil_price)/4)
    oil_train = oil_price.iloc[:split_index]
    stock_train = oil_price.iloc[:split_index]

    oil_test = oil_price.iloc[split_index:]
    stock_test = oil_price.iloc[split_index:]

    return oil_train, stock_train, oil_test, stock_test, oil_price, stock_price


def add_lag(dataset_1, dataset_2, lag):
    if lag != 0:
        dataset_2 = dataset_2[lag:]
        dataset_1 = dataset_1[:-lag]

    return dataset_1, dataset_2


