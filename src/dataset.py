import pandas as pd
from datetime import datetime

class Dataset(object):
    def __init__(self):
        self.load_data()

    def load_data(self):
        raise NotImplementedError

class TaxiKD(Dataset):
    def taxi_date_to_int(self, strdate):
        return int(strdate.split(' ')[0].replace('-', ''))

    def taxi_time_to_int(self, strdate):
        return int(strdate.split(' ')[1].replace(':', ''))

    def load_data(self, attrs=['pickup_date', 'pickup_time', 'dropoff_date',
                                'dropoff_time', 'trip_distance', 'PULocationID','DOLocationID']):
        path = '../data/taxi6.csv'
        self.df = pd.read_csv(path)
        print(self.df.describe())

class Taxi1D(TaxiKD):
    pass

class Taxi2D(TaxiKD):
    pass

class Taxi3D(TaxiKD):
    pass

class Taxi4D(TaxiKD):
    pass

class Taxi5D(TaxiKD):
    pass

class Taxi6D(TaxiKD):
    pass

class Taxi(Dataset):
    def taxi_date_to_epoch(self, strdate):
        return int(int(datetime.strptime(strdate, '%Y-%m-%d %H:%M:%S').strftime('%s')))

    def load_data(self, attrs=['pickup_datetime', 'trip_distance']):
        taxi = pd.read_csv('../data/yellow_tripdata_2019-01.csv')
        taxi['pickup_datetime'] = taxi['tpep_pickup_datetime'].apply(self.taxi_date_to_epoch)
        taxi['dropoff_datetime'] = taxi['tpep_dropoff_datetime'].apply(self.taxi_date_to_epoch)
        self.df = taxi[attrs]
        print(self.df.describe())

class Instacart(Dataset):
    # https://www.instacart.com/datasets/grocery-shopping-2017
    def load_data(self):
        print("loading dataset...")
        df = pd.read_csv('../data/instacart_2017_05_01/order_products__train.csv')
        df = df[df.isnull().any(axis=1) == False]
        self.df = df
        print(df.describe())

class Data1M(Dataset):
    def load_data(self):
        print("loading dataset...")
        df = pd.read_csv('../data/Data1M.csv')
        df = df[df.isnull().any(axis=1) == False]
        self.df = df
        print(df.describe())

class IntelWireless(Dataset):
    def load_data(self):
        print("\nLoading IntelWireless dataset...")
        intel = pd.read_csv('../data/intel.txt', delimiter=' ')
        intel = intel[intel.isnull().any(axis=1) == False]
        intel['idate'] = 1*(pd.to_numeric(intel['date'].str.replace('-', '')) - 20040000)
        intel['itime'] = 1*(pd.to_numeric(intel['time'].str.replace(':', '')).astype(int))
        intel['voltage'] = intel['voltage']*100
        self.df = intel
        print("Loaded %s tuples" % self.df.shape[0])
