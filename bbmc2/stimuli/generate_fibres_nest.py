import numpy as np
import random

number_fibres = 310
exc_address = '/uoa/home/s10js8/data/is_neuron_excitatory_mc2.npy'
save_loc = '/uoa/home/s10js8/data/fibres.npy'

fibres = [[] for i in range(number_fibres)]
exc_array = np.load(exc_address)
exc_loc = set(np.where(exc_array)[0])
inh_loc = set(range(31346)).difference(exc_loc)

for i in range(number_fibres):
    exc = random.randint(718,832)
    inh = random.randint(72,94)
    fibres[i].extend(random.sample(exc_loc,exc))
    fibres[i].extend(random.sample(inh_loc,inh))

np.save(save_loc,fibres)
