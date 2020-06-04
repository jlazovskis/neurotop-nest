# Based on BlueBrain neocortical microcircuit
# http://neuprint.janelia.org/
# https://www.biorxiv.org/content/10.1101/2020.05.18.102061v1
# Neuro-Topology Group, Institute of Mathematics, University of Aberdeen
# Authors: Jānis Lazovskis, Jason Smith
# Date: June 2020

# Packages
import sys                                                                  # For current directory to read and write files
import nest                                                                 # For main simulation
import argparse                                                             # For options
from datetime import datetime                                               # For giving messages

# Auxiliary: Formatted printer for messages
def ntnstatus(message):
	print("\n"+datetime.now().strftime("%b %d %H:%M:%S")+" neurotop-nest: "+message, flush=True)

# Read arguments
parser = argparse.ArgumentParser(
	description='Drosophila reconstruction and validations',
	usage='python drosophila.py')
parser.add_argument('arg', type=str, default='def', help='help')
args = parser.parse_args()

# Set up
#nest.set_verbosity("M_ERROR")                                              # Uncomment this to make NEST quiet
nest.ResetKernel()                                                          # Reset nest
nest.SetKernelStatus({"local_num_threads": 8})                              # Run on many threads
root = sys.argv[0][:-11]                                                    # Current working directory
simulation_id = datetime.now().strftime("%s")                               # Custom ID so files are not overwritten