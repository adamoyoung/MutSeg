import sys
import numpy as np

data = np.load(sys.argv[1])

array = data['array']
barray = bytes(array)

with open("mutation_counts.dat", 'wb') as file:
	file.write(barray)