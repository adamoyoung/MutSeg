file = open("mutation_counts.txt", 'r')

line = file.readline()


pos_tups = set()
#densities = set()
typs = set()

line = file.readline()
while line != '':
	line = line.split()
	chrm = line[1]
	pos = line[2]
	#mut_density = line[8]
	typ = line[9]
	pos_tups.add( (chrm,pos) )
	#densities.add(mut_density)
	typs.add(typ)
	line = file.readline()

print(len(list(pos_tups)))
#print(densities)
print(len(typs))