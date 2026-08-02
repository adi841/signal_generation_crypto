
import os
import pandas as pd

class MarketDataReplay:
    """
    Things to Note:
        1. Timestamp is in UTC
        2. Data is resampled to 1 min
        3. Timestamp is not timezone aware
    
    Sample Data:
                        open     high      low    close  volume
    2016-04-01 00:00:00  0.80005  0.80015  0.80005  0.80015    22.0
    2016-04-01 00:01:00  0.80015  0.80035  0.80015  0.80035     6.0
    2016-04-01 00:03:00  0.80035  0.80035  0.80025  0.80035     5.0
    2016-04-01 00:04:00  0.80035  0.80035  0.80005  0.80005     7.0
    2016-04-01 00:05:00  0.80005  0.80005  0.80005  0.80005     2.0    
    """

    def __init__(self):
        self.hist_replay_data = {}

    def load_hist_replay_data(self, hist_replay_dir, symbol_list, ffill):
        """
        ffill: If True, fill missing values with forward fill.
                Set False in case of pairs.

        Symbol resolution: try most-specific match first so multiple naming
        conventions all work:
            "BTC-USDT-1m-data.parquet" + symbol_list=["BTC-USDT"] -> "BTC-USDT"
            "ADAUSDT-1m-data.parquet"  + symbol_list=["ADAUSDT"]  -> "ADAUSDT"
            "ADAUSDT.parquet"          + symbol_list=["ADAUSDT"]  -> "ADAUSDT"
        """
        for file in os.listdir(hist_replay_dir):
            print("Checking: ", file)
            if file.endswith(".parquet"):
                stem = os.path.splitext(file)[0]
                parts = stem.split("-")
                symbol = None
                for candidate in (stem, "-".join(parts[:2]), parts[0]):
                    if candidate in symbol_list:
                        symbol = candidate
                        break
                if symbol is None:
                    continue

                df_parquet = pd.read_parquet(os.path.join(hist_replay_dir, file))
                df_parquet = df_parquet.sort_index()
                print("Parsing: ", symbol, file)
                self.__parse_df(symbol, df_parquet, ffill)

            else:
                raise ValueError(f"File {file} is not a parquet file")
    
    def __parse_df(self, symbol, df, ffill):
        ## Create timestamp
        start_time = df.index.min()
        end_time = df.index.max()

        ## Create timestamp
        timestamp = pd.date_range(start=start_time, end=end_time, freq='1min')

        ## Create dataframe
        print("Reindexing: ", symbol)
        df = df.reindex(timestamp)
        print("Reindexing Done: ", symbol)

        ## Fill missing values open, high, low, close
        if ffill:
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='ffill')
            df['volume'] = df['volume'].fillna(0)

        ## parse the data to ohlcv dict
        if symbol not in self.hist_replay_data:
            self.hist_replay_data[symbol] = {}

        # Fast path: build mapping in bulk using NumPy view (avoids per-row attribute access)
        print("Building mapping: ", symbol)
        cols = ['open', 'high', 'low', 'close', 'volume']
        values_view = df[cols].to_numpy(copy=False)
        self.hist_replay_data[symbol].update(dict(zip(df.index, map(tuple, values_view))))
        print("Building mapping Done: ", symbol)

    ##
    def get_hist_replay_data(self, curr_time, base_coin):
        return self.hist_replay_data[base_coin][curr_time]



