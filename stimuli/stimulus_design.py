import random
import numpy as np

stim_duration = 1 # duration of stimulus in ms
firing_rate = 2 # not really sure what this does, but needed for NEST

#times at which stimuli fire
s5t = [60,62,125,150,249]
s15t = [5,15,55,60,62,70,125,127,145,150,192,203,215,217,240,249]
s30t = [5,15,55,60,62,70,90,105,125,127,145,150,152,192,203,215,217,230,240,242,246,249]

#percentage of fibres which fire at each time
s5p = [90,90,90,90,90]
s15p = [25,25,25,60,60,25,25,25,25,40,25,40,25,25,25,40]
s30p = [25,25,25,60,60,25,25,40,25,25,25,40,25,25,40,25,25,25,25,25,25,40]


s5_fp = []
for i in range(len(s5t)):
    S = random.sample(range(310),int(s5p[i]*310/100))
    for j in S:
        s5_fp.append((j, s5t[i], s5t[i]+stim_duration, firing_rate))


s15_fp = []
for i in range(len(s15t)):
    S = random.sample(range(310),int(s15p[i]*310/100))
    for j in S:
        s15_fp.append((j, s15t[i], s15t[i]+stim_duration, firing_rate))


s30_fp = []
for i in range(len(s30t)):
    S = random.sample(range(310),int(s30p[i]*310/100))
    for j in S:
        s30_fp.append((j, s30t[i], s30t[i]+stim_duration, firing_rate))


np.save('stim5_firing_pattern.npy',s5_fp)
np.save('stim15_firing_pattern.npy',s15_fp)
np.save('stim30_firing_pattern.npy',s30_fp)