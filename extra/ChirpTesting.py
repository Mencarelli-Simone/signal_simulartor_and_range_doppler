#  ____________________________Imports_____________________________
from linearFM import Chirp
import numpy as np



# ___________________________Test Code_____________________________
# chirp instance under test
pulse = Chirp()
# assign a 10 kHz bandwidth an a time bandwidth product of 50
pulse.set_kt_from_tb(10E3,50)
# resulting in an impulse duration of:
print("impulse duration: ",pulse.duration, " s")

# set a carrier frequency of 10 Ghz
pulse.set_central_freq(10E9)

# pulse repetition frequency of 1 Hz
PRF = 1

# sampling frequency with an oversampling factor of 100
Fc = pulse.get_bandwidth()*100
print (("-")*24)
print("sampling frequency ", Fc," Hz")

# time axis of +- 10s
t = np.arange(-10,10,1/Fc)
# impulse train
s = pulse.baseband_chirp_train(t,PRF)

#%% Plot cell
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,np.abs(s))
ax[1].plot(t,np.angle(s))