# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 20:03:48 2017

@author: sahil.231090
"""

from utils import get_headlines, headline_to_svo, get_comp_df, map_ticker, get_returns_around_date

headline_df = get_headlines()
comp_df = get_comp_df()

num_headlines, headline_file_cols = headline_df.shape
text = headline_df.loc[1, 'headline']
headline_date = headline_df.loc[1, 'date']
subject, verb, obj =  headline_to_svo(text)
ticker = map_ticker(subject,
                    headline_date, comp_df)
ticker_ret = get_returns_around_date(ticker_list=[ticker],
                                     around_date=headline_date)

# Tests
assert num_headlines == 1869136
assert headline_file_cols == 2
assert text == 'Corning (GLW) Crosses Pivot Point Support at $23.01'
assert subject == 'Corning'
assert verb == 'crosses'
assert obj == 'Pivot Point Support'
assert ticker == 'GLW'