# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:52:43 2017

@author: sahil.231090
"""
import os
import re
import functools
import pandas as pd
from datetime import datetime, timedelta
from sklearn import linear_model
import matplotlib.pyplot as plt
from fpdf import FPDF
import const
import textacy 
import string
import difflib

data_df = pd.DataFrame()

clf = linear_model.LinearRegression()
DATEPARSER = lambda x: datetime.strptime(str(int(x)), '%Y%m%d').date()
DATEPARSER2 = lambda x: datetime.strptime(x, '%Y-%m-%d').date()

def get_headlines():
    DATE_COL = const.HEADLINE_FILE_DATE_COL
    HEADLINE_COL = const.HEADLINE_FILE_HEADLINE_COL
    df = pd.read_csv(const.HEADLINE_FILE,
                     sep=const.HEADLINE_FILE_SEP)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER2)
    return df[[DATE_COL, HEADLINE_COL]]

def headline_to_svo(text):
    text_str = ''.join(filter(lambda x:x in string.printable, text))
    text_lower = text_str.lower()
    d1 = textacy.Doc(text_lower, lang=u"en")
    text_lower_str = str(text_lower)
    vs = textacy.extract.get_main_verbs_of_sent(d1)
    for v in vs:
        v_str = str(v)
        idx = text_lower_str.index(v_str)
        text_str = text_str[:idx] + v_str + text_str[idx+len(v):]
    d = textacy.Doc(text_str, lang=u"en")
    svo = textacy.extract.subject_verb_object_triples(d)
    return next(svo, None)

def get_comp_df():
    DATE_COL = const.COMP_QUAT_FILE_DATE_COL
    COPMANY_COL = const.COMP_QUAT_FILE_COMPANY_COL
    df = pd.read_csv(const.COMP_QUAT_FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    df[COPMANY_COL] = df[COPMANY_COL].str.upper().apply(
                            lambda x: re.sub('[.!,]', '', x))
    common_words = pd.Series(' '.join(df[COPMANY_COL].unique()
                                      ).split()
                            ).value_counts()[:30].index
    common_words = filter(lambda x: len(x) > 2, common_words)
    df[COPMANY_COL] = df[COPMANY_COL].replace(common_words, '',
                                regex=True).apply(str.strip)
    return df
    
def map_ticker(text, headline_date, comp_df):
    COMPANY_COL = const.COMP_QUAT_FILE_COMPANY_COL
    TICKER_COL = const.COMP_QUAT_FILE_TICKER_COL
    DATE_COL = const.COMP_QUAT_FILE_DATE_COL
    comps = comp_df[COMPANY_COL]
    name_match = difflib.get_close_matches(str(text).upper(),
                                                list(comps),
                                                cutoff=0.95)
    if(len(name_match) > 0):
        vals = comp_df.loc[comp_df[COMPANY_COL] == name_match[0],
                         [TICKER_COL,DATE_COL]].values[0]
        ticker = vals[0]
        data_date = vals[1]
        if(headline_date >= data_date):
            return ticker
    return None
    
def get_cached_returns(key):
    file_path = r'C:\Users\Public\Data\Cache\{0}.csv'.format(key)
    df = pd.read_csv(file_path)
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date())
    return df.set_index('date')

def get_risk_free_returns():
    DATE_COL = 'date'
    df = pd.read_csv(const.RISK_FREE_RETURN_FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    return df.set_index(DATE_COL)/100

def get_size_returns():
    DATE_COL = 'date'
    df = pd.read_csv(const.SIZE_FF_RETURN_FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    return df.set_index(DATE_COL)/100

def get_value_returns():
    DATE_COL = 'date'
    df = pd.read_csv(const.VALUE_FF_RETURN_FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    return df.set_index(DATE_COL)/100

def get_momentum_returns():
    DATE_COL = 'date'
    df = pd.read_csv(const.MOMENTUM_FF_RETURN_FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    return df.set_index(DATE_COL)/100

def get_ff_returns(factors=['SIZE', 'VALUE', 'MOM']):
    out_df = pd.DataFrame
    for factor in factors:
        if factor == 'SIZE':
            temp_df = get_size_returns()
        elif factor == 'VALUE':
            temp_df = get_value_returns()
        elif factor == 'MOM':
            temp_df = get_momentum_returns()
        
        
        if out_df.empty:
            out_df = temp_df
        else:
            out_df = out_df.merge(temp_df, how='outer',
                                  right_index=True, left_index=True)
    return out_df
    
def get_us_trading_days():
    return map(lambda x: datetime.strptime(str(x), '%Y-%m-%d').date(),
               pd.read_csv(const.US_TRADING_DAYS_FILE, header=-1)[0].tolist())

def get_all_returns():
    ret_df = pd.read_csv(const.RET_DATA_FILE)
    ret_df = ret_df[['TICKER', 'RET', 'date']]
    ret_df['RET'] = pd.to_numeric(ret_df['RET'], errors='coerce')
    ret_df = pd.pivot_table(ret_df, values='RET', index='date', columns='TICKER')
    ret_df.index = ret_df.index.map(DATEPARSER)
    return ret_df

    
def get_returns(ticker_list, adjust_cusip=True):
    out_df = pd.DataFrame()
    RET_COL = 'RET'
    DATE_COL = 'date'
    CUSIP_COL = 'CUSIP'
    df_list = []
    for ticker in ticker_list:
        csv_file = const.RETURNS_FOLDER + os.sep + ticker + '.csv'
        ticker_df = pd.read_csv(csv_file, dtype={DATE_COL:int}
                                ).sort_values(by=DATE_COL)
        ticker_df[RET_COL] = pd.to_numeric(ticker_df[RET_COL], errors='coerce')
        cusips = ticker_df[CUSIP_COL].unique()
        if (len(cusips) > 1) and adjust_cusip:
            max_cusip = ticker_df.loc[ticker_df[DATE_COL].idxmax(), CUSIP_COL]
            min_cusip = ticker_df.loc[ticker_df[DATE_COL].idxmin(), CUSIP_COL]
            if max_cusip is not min_cusip:
                ticker_df = ticker_df[ticker_df[CUSIP_COL] == max_cusip]
        ticker_df = ticker_df[[RET_COL, DATE_COL]]
        ticker_df = ticker_df.rename(columns={RET_COL:ticker})
        df_list.append(ticker_df)
    out_df = functools.reduce(lambda x,y:  pd.merge(x,y,how='outer',on=DATE_COL), df_list)            
    out_df.loc[:, DATE_COL] = out_df.loc[:, DATE_COL].apply(DATEPARSER)
    out_df = out_df.drop_duplicates(subset=DATE_COL, keep='last')
    out_df = out_df.set_index(DATE_COL).dropna(how='all').sort_index()
    return out_df
        
def get_ticker_returns(ticker):
    return get_returns([ticker])

def _filter_df_around_date(df, around_date,
                           days_before=30, days_after=60):
    idx = pd.np.argmax(df.index >= around_date)
    return df.iloc[idx-days_before:idx+days_after+1, :]

def _scale_by_rolling_std(df, std_window=252):
    if len(df) <= std_window:
        print('Warning! Tickers {0} do not have enough returns\n STD Window: {1} Returns {2}'.format(df.columns, std_window, len(df)))
        df = df.divide(df.std())
    else:     
        df /= df.rolling(center=False,window=std_window).std().shift(1)
    return df.iloc[std_window:,:]

def _run_ff_regression(ret_df, mkt_df, ff_df, beta_window=252):
    out_df = pd.DataFrame(pd.np.zeros(ret_df.iloc[beta_window:,:].shape),
                        index=ret_df.index[beta_window:],
                        columns=ret_df.columns)
    for idx, dt in enumerate(out_df.index):
        for jdx, tic in enumerate(out_df.columns):
            X = pd.np.concatenate((mkt_df.iloc[idx:beta_window+idx],
                                   ff_df.iloc[idx:beta_window+idx],
                                    pd.np.ones((beta_window,1))),
                                    axis=1)
            y = ret_df.ix[idx:beta_window+idx, jdx]
            beta = pd.np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
            X_test = pd.np.concatenate((mkt_df.iloc[beta_window+idx, :],
                                   ff_df.iloc[beta_window+idx, :],
                                    [1]))
            y_test = ret_df.ix[beta_window+idx, jdx]
            y_hat = beta.T.dot(X_test)
            out_df.loc[dt, tic] = y_test - y_hat
    return out_df

def get_scaled_returns(ticker_list, std_window=252):
    ret_df = get_returns(ticker_list)
    return _scale_by_rolling_std(ret_df, std_window=252)

def get_returns_around_date(ticker_list, around_date,
                            days_before=30, days_after=60):
    ret_df = get_returns(ticker_list)
    return _filter_df_around_date(ret_df, around_date,
                           days_before, days_after)

def get_ticker_returns_around_date(ticker, around_date,
                            days_before=30, days_after=60):
    ret_df = get_ticker_returns(ticker)
    return _filter_df_around_date(ret_df, around_date,
                           days_before, days_after)

def get_scaled_returns_around_date(ticker_list, around_date,
                            days_before=30, days_after=60,
                            std_window=252):
    ret_df = get_returns_around_date(ticker_list, around_date,
                            days_before, days_after)
    return _scale_by_rolling_std(ret_df, std_window)

def load_index_returns():
    DATE_COL = 'datadate'
    df = pd.read_csv(const.INDEX_RETURN_FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    df = df.set_index(DATE_COL)
    return df

def _load_gics_file_returns(COL, FILE, is_mcap_weighted):
    DATE_COL = 'date'
    SUBIND_COL = COL
    if is_mcap_weighted:
        return_weighting = 'mret'
    else:
        return_weighting = 'ret'
    df = pd.read_csv(FILE)
    df[DATE_COL] = df[DATE_COL].apply(DATEPARSER)
    df[SUBIND_COL] = df[SUBIND_COL].apply(int)
    return pd.pivot_table(df, index=DATE_COL,
                          columns=SUBIND_COL,
                          values=return_weighting)

def load_gics_returns(level, is_mcap_weighted=True):
    if level in [8, 'subind', 'SUBINDUSTRY']:
        (COL, FILE) = ('subind', const.SUBIND_RETURN_FILE)
    elif level in [6, 'ind', 'INDUSTRY']:
        (COL, FILE) = ('ind', const.IND_RETURN_FILE)
    elif level in [4, 'indgrp', 'INDUSTRYGROUP']:
        (COL, FILE) = ('group', const.GROUPIND_RETURN_FILE)
    elif level in [2, 'sct', 'SECTOR']:
        (COL, FILE) = ('sector', const.SECTOR_RETURN_FILE)
    return _load_gics_file_returns(COL, FILE,
                                   is_mcap_weighted)
        
def get_subind_returns(subinds, is_mcap_weighted=True):
    df = load_gics_returns(8, is_mcap_weighted)
    return df[subinds]

def get_ind_returns(inds, is_mcap_weighted=True):
    df = load_gics_returns(6, is_mcap_weighted)
    return df[inds]

def get_indgrp_returns(indgrps, is_mcap_weighted=True):
    df = load_gics_returns(4, is_mcap_weighted)
    return df[indgrps]

def get_sct_returns(scts, is_mcap_weighted=True):
    df = load_gics_returns(2, is_mcap_weighted)
    return df[scts]

def get_snp_500_returns():
    INDEX_COL = 'index'
    RET_COL = 'ret'
    idx_df = load_index_returns()
    idx_df = idx_df[idx_df[INDEX_COL] == const.SNP_500_KEY]
    idx_df = idx_df.rename(columns={RET_COL:'S&P500'})
    return idx_df[['S&P500']]

def get_market_returns(market='S&P500'):
    assert market in ['S&P500']
    if market == 'S&P500':
        df = get_snp_500_returns()
    return df

def get_beta_adj_returns(ticker_list, market='S&P500',
                         beta_window=252):
    ret_df = get_returns(ticker_list)
    mkt_df = get_market_returns(market)
    df = pd.merge(ret_df, mkt_df, how='inner',
                  left_index=True, right_index=True)
    mkt_df = df[market]
    ret_df = df.drop(market, axis=1)
    df = df.rolling(window=beta_window).cov(df['S&P500']).shift(1)
    df = df.divide(df[market], axis=0)
    df = df.drop(market, axis=1)
    ret_df -= df.multiply(mkt_df, axis=0)
    return ret_df.iloc[beta_window:, :]

def get_beta_adj_std_returns(ticker_list, market='S&P500',
                         beta_window=252, std_window=252):
    ret_df = get_beta_adj_returns(ticker_list, market,
                         beta_window)
    return _scale_by_rolling_std(ret_df, std_window)

def get_beta_adj_returns_around_date(ticker_list, around_date,
                                     days_before=30, days_after=60,
                                     market='S&P500', beta_window=252):
    ret_df = get_beta_adj_returns(ticker_list, market,
                                  beta_window)
    return _filter_df_around_date(ret_df, around_date,
                           days_before, days_after)

def get_beta_adj_std_returns_around_date(ticker_list, around_date,
                                     days_before=30, days_after=60,
                                     market='S&P500', beta_window=252,
                                     std_window=252):
    ret_df = get_beta_adj_std_returns(ticker_list, market,
                                  beta_window, std_window)
    return _filter_df_around_date(ret_df, around_date,
                           days_before, days_after)

def load_global_data_df():
    global data_df
    DATE_COL = 'date'
    print('Warning: Global DataFrame is empty, might take some time')
    data_df = pd.read_csv(const.DATA_FILE)
    data_df[DATE_COL] = data_df[DATE_COL].apply(DATEPARSER)
    print('OKAY!! Good to go')
    return data_df

def get_valid_tickers():
    global data_df
    if data_df.empty:
        load_global_data_df()
    return data_df.TICKER.unique().tolist()
    
def get_industry_classification(ticker_list, level=6):
    global data_df
    DATE_COL = 'date'
    if level in [8, 'subind', 'SUBINDUSTRY']:
        COL = 'gsubind'
    elif level in [6, 'ind', 'INDUSTRY']:
        COL = 'gind'
    elif level in [4, 'indgrp', 'INDUSTRYGROUP']:
        COL = 'ggroup'
    elif level in [2, 'sct', 'SECTOR']:
        COL = 'gsector'
    
    if data_df.empty:
        print('Warning: Global DataFrame is empty, might take some time')
        data_df = pd.read_csv(const.DATA_FILE)
        data_df[DATE_COL] = data_df[DATE_COL].apply(DATEPARSER)
        print('OKAY!! Good to go')
        
    data_df[COL] = data_df[COL].astype(int)
    dts = list(get_us_trading_days())
    df = pd.pivot_table(data_df, index=DATE_COL, columns='TICKER',
                          values=COL, aggfunc='last'
                          )[ticker_list].dropna(how='all')
    df = pd.DataFrame(df, index=dts).bfill().ffill()
    return df

def get_beta_ind_adj_returns(ticker_list, level=6,
                             beta_window=252,
                             is_mcap_weighted=True):
    ret_df = get_returns(ticker_list)
    ind_df = load_gics_returns(level, is_mcap_weighted)
    cls_df = get_industry_classification(ticker_list, level)
    
    df = pd.merge(ret_df, ind_df, how='outer',
                  left_index=True, right_index=True)
    cls_df= cls_df.loc[df.index, :].bfill().ffill()

    out_df = pd.DataFrame(pd.np.zeros((len(df)-beta_window, len(ticker_list))),
                         index=df.index[252:], columns=ticker_list)
    for i, dt in enumerate(out_df.index):
        for j, tic in enumerate(ticker_list):
            ind = int(cls_df.loc[dt, tic])
            y = df[tic][i:i+beta_window]
            x = df[ind][i:i+beta_window]
            r = df[tic].iloc[i+beta_window]
            ri = df[ind].iloc[i+beta_window]
            temp = pd.np.cov(x,y)
            beta = temp[0,1]/temp[0,0]
            out_df.loc[dt, tic] = r - ri*beta 
    return out_df

def get_beta_ind_adj_returns_around_date(ticker_list, around_date,
                                        days_before=30, days_after=60,
                                        level=6, beta_window=252,
                                        is_mcap_weighted=True):
    df = get_beta_ind_adj_returns(ticker_list, level,
                             beta_window, is_mcap_weighted)
    return _filter_df_around_date(df, around_date,
                                  days_before, days_after)
    
def get_beta_ind_adj_std_returns(ticker_list, level=6,
                             beta_window=252, std_window=252,
                             is_mcap_weighted=True):
    try:
        ret_df = get_beta_ind_adj_returns(ticker_list, level,
                         beta_window,
                         is_mcap_weighted)
    except:
        print(ticker_list)
        assert 1 == 0
    return _scale_by_rolling_std(ret_df, std_window)

def get_beta_ind_adj_std_returns_around_date(ticker_list, around_date,
                                 level=6, days_before=30, days_after=60,
                                beta_window=252, std_window=252,
                                is_mcap_weighted=True):
    df = get_beta_ind_adj_std_returns(ticker_list, level,
                             beta_window, std_window,
                             is_mcap_weighted)
    return _filter_df_around_date(df, around_date,
                                  days_before, days_after)
def get_industry_classification_around_date(ticker_list, around_date,
                                            days_before=30, days_after=60,
                                            level=6):
    df = get_industry_classification(ticker_list, level)
    return _filter_df_around_date(df, around_date,
                                  days_before, days_after)
 
def get_ff_adj_returns(ticker_list,  market='S&P500',
                         beta_window=252,
                         ff_factors=['SIZE', 'VALUE', 'MOM']):
    ret_df = get_returns(ticker_list)
    rf_df = get_risk_free_returns().loc[ret_df.index, :]
    mkt_df = get_market_returns(market).loc[ret_df.index, :]
    ff_df = get_ff_returns(ff_factors).loc[ret_df.index, :]
    
    ret_df = ret_df.subtract(rf_df['RF'], axis=0)
    mkt_df = mkt_df.subtract(rf_df['RF'], axis=0)
    return _run_ff_regression(ret_df, mkt_df, ff_df, beta_window)    

def get_ff_adj_returns_around_date(ticker_list, around_date,
                                   days_before=30, days_after=60,
                                   beta_window=252, market='S&P500',
                                   ff_factors=['SIZE', 'VALUE', 'MOM']):

    ret_df = get_returns_around_date(ticker_list, around_date,
                                     days_before+beta_window, days_after)
    rf_df = get_risk_free_returns().loc[ret_df.index, :]
    mkt_df = get_market_returns(market).loc[ret_df.index, :]
    ff_df = get_ff_returns(ff_factors).loc[ret_df.index, :]
    
    ret_df = ret_df.subtract(rf_df['RF'], axis=0)
    mkt_df = mkt_df.subtract(rf_df['RF'], axis=0)
    return _run_ff_regression(ret_df, mkt_df, ff_df, beta_window)    


def get_ff_adj_std_returns(ticker_list,  market='S&P500',
                         beta_window=252,  std_window=252,
                         ff_factors=['SIZE', 'VALUE', 'MOM']):
    df = get_ff_adj_returns(ticker_list,  market,
                         beta_window, ff_factors)
    return _scale_by_rolling_std(df, std_window)

def get_ff_adj_std_returns_around_date(ticker_list, around_date,
                                   days_before=30, days_after=60,
                                   beta_window=252, std_window=252,
                                   market='S&P500',
                                   ff_factors=['SIZE', 'VALUE', 'MOM']):
    df = get_ff_adj_returns_around_date(ticker_list, around_date,
                                   days_before+std_window, 
                                   days_after, beta_window, market,
                                   ff_factors)
    return _scale_by_rolling_std(df, std_window)

    
def get_custom_returns(ticker_list,
                       return_method='get_ff_adj_std_returns_around_date',
                       **kwargs):
    if return_method == 'get_returns':
        return get_returns(ticker_list)
    elif return_method == 'get_scaled_returns':
        return get_scaled_returns(ticker_list, **kwargs)
    elif return_method == 'get_returns_around_date':
        return get_returns_around_date(ticker_list, **kwargs)
    elif return_method == 'get_scaled_returns_around_date':
        return get_scaled_returns_around_date(ticker_list, **kwargs)
    
    elif return_method == 'get_beta_adj_returns':
        return get_beta_adj_returns(ticker_list, **kwargs)
    elif return_method == 'get_beta_adj_std_returns':
        return get_beta_adj_std_returns(ticker_list, **kwargs)
    elif return_method == 'get_beta_adj_returns_around_date':
        return get_beta_adj_returns_around_date(ticker_list, **kwargs)
    elif return_method == 'get_beta_adj_std_returns_around_date':
        return get_beta_adj_std_returns_around_date(ticker_list, **kwargs)
    
    elif return_method == 'get_beta_ind_adj_returns':
        return get_beta_ind_adj_returns(ticker_list, **kwargs)
    elif return_method == 'get_beta_ind_adj_std_returns':
        return get_beta_ind_adj_std_returns(ticker_list, **kwargs)
    elif return_method == 'get_beta_ind_adj_returns_around_date':
        return get_beta_ind_adj_returns_around_date(ticker_list, **kwargs)
    elif return_method == 'get_beta_ind_adj_std_returns_around_date':
        return get_beta_ind_adj_std_returns_around_date(ticker_list, **kwargs)
    
    elif return_method == 'get_ff_adj_returns':
        return get_ff_adj_returns(ticker_list, **kwargs)
    elif return_method == 'get_ff_adj_std_returns':
        return get_ff_adj_std_returns(ticker_list, **kwargs)
    elif return_method == 'get_ff_adj_returns_around_date':
        return get_ff_adj_returns_around_date(ticker_list, **kwargs)
    elif return_method == 'get_ff_adj_std_returns_around_date':
        return get_ff_adj_std_returns_around_date(ticker_list, **kwargs)
    
        
    return None

def _event_df_to_ret_df(event_df, ticker_col='TICKER', date_col='DATE',
                        days_before=30, days_after=60,
                          return_method='get_ff_adj_std_returns_around_date',
                          **kwargs):
    vals = event_df[[ticker_col, date_col]].values.T
    mul_idx = pd.MultiIndex.from_arrays(vals)
    out_df = pd.DataFrame(index=range(-days_before,days_after+1),
                          columns=mul_idx)
    i=0
    for ticker, event_dt in mul_idx:
        try:
            ret_vec = get_custom_returns([ticker], return_method,
                                        around_date=event_dt.date(),
                                        days_before=days_before,
                                        days_after=days_after,
                                         **kwargs).values
            #out_df.loc[:, (ticker, event_dt)] = ret_vec
            out_df.iloc[:, i] = ret_vec[:,0]
            i=i+1
        except:
            continue
        if i % 100 == 0:
            print(i)
    return out_df.iloc[:, :i]

def _event_df_to_ret_df_from_cache(event_df, cache_df,
                                   days_before=30, days_after=60,
                                   ticker_col='TICKER',
                                   date_col='DATE'):
    max_date = max(event_df[date_col])
    event_df = event_df[event_df[date_col] < max_date-timedelta(days=days_after*7.0/5)]
    cache_df = cache_df - pd.np.nanmean(cache_df)
    vals = event_df[[ticker_col, date_col]].drop_duplicates().values.T
    mul_idx = pd.MultiIndex.from_arrays(vals)
    out_df = pd.DataFrame(data=pd.np.zeros((days_before+days_after+1,
                                            len(mul_idx)))*pd.np.nan,
                          index=range(-days_before,days_after+1),
                          columns=mul_idx)
    for ticker, event_dt in mul_idx:
        idx = pd.np.argmax(cache_df.index >= event_dt.date())
        ret_vec = cache_df.ix[idx-days_before:idx+days_after+1, ticker]
            
        out_df.loc[:, (ticker, event_dt)] = ret_vec.values
    return out_df

def _get_event_info_df(num_events=5):
    df = pd.read_csv(const.TEST_EVENT_INFO_FILE)[['Subject', 'date']]
    df.columns = ['TICKER', 'DATE']
    df['DATE'] = df['DATE'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d').date())
    df['EVENT_ID'] = pd.np.random.randint(num_events, size=len(df))
    return df  

def get_event_study_panel_from_cache(event_info_df, cache_df,
                                     days_before=30, days_after=60):
    EVENT_COL = 'EVENT_ID'
    TICKER_COL = 'TICKER'
    DATE_COL = 'DATE'
    df_dict = {}
    for event_id, event_df in event_info_df.groupby(EVENT_COL):
        ret_df = _event_df_to_ret_df_from_cache(event_df, cache_df,
                        days_before, days_after, TICKER_COL, DATE_COL)
        df_dict[event_id] = ret_df
    return df_dict

def get_event_study_panel(event_info_df, days_before=30, days_after=60,
                          return_method='get_ff_adj_std_returns_around_date',
                          **kwargs):
    EVENT_COL = 'EVENT_ID'
    TICKER_COL = 'TICKER'
    DATE_COL = 'DATE'
    df_dict = {}
    valid_tics = get_valid_tickers()
    print('starting filter')
    event_info_df = event_info_df[event_info_df[TICKER_COL].isin(valid_tics)]
    print('filter done')
    print(event_info_df.shape)
    for event_id, event_df in event_info_df.groupby(EVENT_COL):
        ret_df = _event_df_to_ret_df(event_df, TICKER_COL, DATE_COL,
                        days_before, days_after, return_method, **kwargs)
        df_dict[event_id] = ret_df
    return df_dict

def plot_graphs_to_output(event_data_dict,
                          output_folder=const.PLOT_OUTPUT_FOLDER,
                          verbs=None, objects=None,
                          save_png=True, save_pdf=True):
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.8
    left_table = left+width+0.1
    table_width = 0.15
    table_height = width/2.
    
    rect_main = [left, bottom, width, height]
    rect_table1 = [left_table, table_height+bottom , table_width, table_height]
    rect_table2 = [left_table, bottom, table_width, table_height]
    
    if save_pdf:
        pdf = FPDF()
    idx = 0
    for event_id, ret_df in event_data_dict.items():
        #
        #fig, ax = plt.subplots(figsize=(15, 8))
        plt.figure(figsize=(15, 8))
        
        axMain = plt.axes(rect_main)
        axTable1 = plt.axes(rect_table1, frameon =False)
        axTable2 = plt.axes(rect_table2, frameon =False)
        axTable1.axes.get_xaxis().set_visible(False)
        axTable2.axes.get_xaxis().set_visible(False)
        axTable1.axes.get_yaxis().set_visible(False)
        axTable2.axes.get_yaxis().set_visible(False)
        
        series = ret_df.apply(pd.np.nanmean, axis=1)
        
        axMain.bar(series.index, series.values, color='b')
        axMain.plot(series.index, series.cumsum().values, color='r')
        axMain.axvline(0, color='k', linestyle='--')
        axMain.set_xlabel('Days After Event')
        axMain.set_ylabel('Returns')
        
        col_labels=['Hit Rate','IR','t-stat']
        row_labels=[]        
        table_vals=[]
                
        for val in [20, 40, 60]:
            row_labels.append('1-{0}'.format(val))
            vals = event_data_dict[event_id].loc[1:val,:].sum(axis=0)
            hr = len(list(filter(lambda x: x >= 0, series.loc[1:val])))*1.0/len(series.loc[1:val])
            avg_ret = pd.np.nanmean(vals)
            ir = pd.np.sqrt(252)*avg_ret/pd.np.nanstd(vals)
            t_stat = ir*pd.np.sqrt(val/252.0)
            table_vals.append(['{0:.2f}'.format(hr),
                               '{0:.2f}'.format(ir),
                                '{0:.2f}'.format(t_stat)])        
        axTable1.table(cellText=table_vals,
                  #colWidths = [0.08]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='upper center')
        axMain.set_xlim([-35, 65])
        
        if verbs is not None:
            row_labels = list(range(1,verbs.shape[1]+1))
            col_labels = ['Verbs', 'Objects']
            table_vals=[]            
            for j in range(verbs.shape[1]):
                idx2 = event_id % 1000
                table_vals.append([verbs[idx2,j].decode('utf-8'),
                                   objects[idx2,j].decode('utf-8')])

            axTable2.table(cellText=table_vals,
                  #colWidths = [0.1]*3,
                  rowLabels=row_labels,
                  colLabels=col_labels,
                  loc='upper center')
        
        axMain.set_title('EVENT_ID:{0}\t COUNT:{1}'.format(
                     event_id, ret_df.shape[1]).expandtabs())
        axTable1.set_title('Summary Statistics')
        axTable2.set_title('Word Sensibility')
        
        out_file = output_folder + os.sep + 'EVENT_{0}.png'.format(event_id)
        plt.savefig(out_file)
        pdf.add_page()
        pdf.image(out_file,10,50,190,100 )
        idx += 1
        plt.close()
    pdf.output(output_folder + os.sep + "RESULTS.pdf", "F")
