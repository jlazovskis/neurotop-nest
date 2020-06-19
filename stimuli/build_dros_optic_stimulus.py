
from neuprint import Client
import numpy as np
import random

start_firing = [10,25,50,75]                      #Four randomly chosen firing times
perc = [100,75,50,25]                             #Percentage of the optic fibres to include at each firing time
k = 3                                             #Maximum number of second a vertex is stimulated for
power = 50                                        #Strength of the stimulus

#Download data from janelia
T='' #Insert your neuprint key here
c=Client('neuprint.janelia.org',dataset='hemibrain:v1.0.1',token=T)
q = """MATCH (n :Neuron )
        WHERE n.cropped = False
        RETURN n.bodyId AS bodyId, n.roiInfo AS regions, n.instance AS instance, n.type AS type
        ORDER BY n.pre + n.post DESC
    """
results=c.fetch_custom(q)


#These are the neuron types connecting the optic lobe and the brain taken from:
# https://docs.google.com/viewer?a=v&pid=forums&srcid=MTAzNzI3NzQ4MjY0MDQyNzUyMTkBMTEyMDg4MjIyMjI0MDUwMzk2MjYBbGlWSjEySTZCQUFKATAuMQEBdjI&authuser=0
optic=['DCH','H2','HSE','HSN','HSS','LC10','LC11','LC12','LC13','LC14','LC15','LC16','LC17','LC18','LC20','LC21','LC22','LC24','LC25','LC26','LC27','LC28a','LC28b','LC29','LC31','LC32','LC33','LC34','LC35','LC36','LC37a','LC37b','LC38','LC39','LC4','LC40a','LC40b','LC41','LC43','LC44','LC45a','LC45b','LC46a','LC46b','LC6','LC9','LLPC1','LLPC2a','LLPC2b','LLPC2c','LLPC2d','LLPC3','LPC1','LPC2','LPLC1','LPLC2','LPLC4','LT1','LT11','LT33','LT34','LT35','LT36','LT37','LT38','LT39','LT40','LT41','LT42','LT43','LT44','LT45','LT46','LT47','LT48a','LT48b','LT49','LT51','LT52','LT53','LT54','LT55','LT56','LT57','LT58','LT59','LT60','LT61','LT62','LT63','LT64','LT65','LT66','LT67','LT68','LT69','LT70','LT71','LT72','LT73','LT74','LT75','LT76','LT77','LT78','LT79','LT80','LT81','LT82a','LT82b','LT83','LT84','LT85','LT86','LT87','MC61','MC62','MC64','MC65','MC66','VCH','VS']

# A dictionary that maps the bodyId to the vertex number (I think need to check how Nicloas built matrix)
D={results.iloc[i]['bodyId']:i for i in range(21663)}

#Create a fibre for each optic neuron type
fibres_full = [[D[i] for i in list(results[results['type']==j]['bodyId'])] for j in optic]
fibres = [i for i in fibres_full if len(i)>0]                                        #remove any empty fibres

#for each firing time select perc% of the fibres
firing_neurons = [random.sample(range(len(fibres)),int(len(fibres)*perc[i]/100)) for i in range(len(start_firing))]

#Create stimulus
stimulus = []
for i in range(len(start_firing)):
    for j in firing_neurons[i]:
        stimulus.append((j,start_firing[i],                                         #Each stimulus consists of the tuple (fibre,start_time,end_time,strength)
                             start_firing[i]+random.random()*k,                     #The end time is a random time between start_time and start_time+k
                             power+random.random()*(power/10)-(power/20)))          #The strength randomly taken from within 5% of power value


#Save the fibres and stimulus as numpy arrays
np.save('drosophila_optic_fibres.npy',fibres)
np.save('drosophila_optic_stimulus.npy',stimulus)
