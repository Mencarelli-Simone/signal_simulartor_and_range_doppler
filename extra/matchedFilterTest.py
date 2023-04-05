import numpy as np
from matchedFilter import MatchedFilter
from linearFM import Chirp
from tqdm import tqdm

#%%C:\Users\smen851\PycharmProjects
# create a pulse instance
pulse = Chirp()
# bandwidth of 10, tb of 50
pulse.set_kt_from_tb(10,50)

# create matched filter instance
filter = MatchedFilter(pulse)

#%% use test internal
# a time axis
Fc = 10 * pulse.get_bandwidth()
#
#t = 0
#result,a,b = filter.matched_filter_in_time(t)
t = np.arange(-10,10,.1)
res = 1j*np.zeros_like(t)
for i in tqdm(range(0,len(t))): #due commento
    res[i] = filter.matched_filter_in_time(t[i])
