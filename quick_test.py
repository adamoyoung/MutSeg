import sys

file = open(sys.argv[1], 'r')

# skip header
line = file.readline()

chrms = set()
pos_dict = {}
for i in range(1,23):
	pos_dict[str(i)] = set()
pos_dict['X'] = set()
pos_dict['Y'] = set()
#densities = set()
typs = set()
sexes = {}

line_count = 1
line = file.readline()
while line != '':
	if line_count % 1000000 == 0:
		print(line_count, end=",")
	line = line.split(",")
	if len(line) < 2 or line[1] == "chr":
		if len(line) < 2:
			print( "line {}:".format(line_count) )
			print( line )
		line_count += 1
		line = file.readline()
		continue
	chrm = line[1]
	pos = line[2]
	mut_density = line[8]
	typ = line[9]
	sex = line[10].strip()
	chrms.add(chrm)
	pos_dict[chrm].add(pos)
	#densities.add(mut_density)
	typs.add(typ)
	if sex in sexes.keys():
		sexes[sex] += 1
	else:
		sexes[sex] = 1
	line_count += 1
	line = file.readline()

# print(chrms)
# total_num_pos = 0
# for key in pos_dict.keys():
# 	pos_dict_len = len(pos_dict[key])
# 	print( "{}: {}".format(key,pos_dict_len) )
# 	total_num_pos += pos_dict_len
# print( "Total num pos = {}".format(total_num_pos) )
#print(densities)
print(sorted(typs))
print(sexes)