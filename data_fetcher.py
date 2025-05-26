from math import ceil
from time import sleep
from pandas_ta import macd, rsi
from yfinance import Ticker, download
from webull import paper_webull as pwb
from datetime import datetime, date, time
from pandas_market_calendars import get_calendar
from yfinance.exceptions import YFRateLimitError
from pandas import concat, DataFrame, to_datetime, Timedelta

def getDictionary(ticker:str, get_dataframe=True):
    try: return Ticker(ticker).history(period='max') 
    except YFRateLimitError:
        df = download(ticker, multi_level_index = False, auto_adjust=True, period='max', progress=False)
        if len(df): return df
        if 'inceptionDate' in pwb().get_ticker_info(ticker):
            start_date = to_datetime(pwb().get_ticker_info(ticker)['inceptionDate']).date() #when QQQ first entered the market
            today_date = date.today()
            dates = [(start_date+Timedelta(days=365*(i))) for i in range(int(ceil((today_date - start_date)/Timedelta(days=365))))]
            days = (today_date - dates[-1]).days
            if days < 365 and days > 0: dates[-1] = today_date
            dictionary = {'Start Date: ' + dates[i].strftime('%Y-%m-%d') : pwb().get_bars(stock=ticker,
            count=len(get_calendar('NYSE').schedule(dates[i],dates[i+1])), interval='d1',
            timeStamp=int(datetime.combine(dates[i+1]+Timedelta(days=1),
            time(0,0)).timestamp())) for i in range(len(dates)) if i!=(len(dates)-1)}
            if get_dataframe: dictionary = concat(list(dictionary.values())).reset_index().drop_duplicates().set_index('timestamp')
            return dictionary
    print('No data obtained.')
    return DataFrame()

def get_historical_data(ticker:str, macd_params: dict={}, rsi_params: dict={}):
    tradeable_dict = pwb().get_tradable(ticker)
    if 'success' in tradeable_dict and tradeable_dict['success']:
        df = getDictionary(ticker)
        df.rename(columns={i:i.capitalize() for i in df.columns},inplace=True)
        if not macd_params:
            macd_df = macd(df.Close)
            if macd_df.columns[macd_df.columns.str.contains('MACD_')]: 
                df = df.join(macd_df[macd_df.columns[macd_df.columns.str.contains('MACD_')]])
        else: df = df.join(macd(df['Close'], **macd_params))
        if not rsi_params: df['RSI'] = rsi(df.Close)
        else: df['RSI'] = rsi(df['Close'], **rsi_params)
        df.dropna(inplace=True)
        return df
    return DataFrame()