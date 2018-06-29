import sys
import numpy as np

if len(sys.argv) != 2:
	print("Usage")
mc_data_file_name = sys.argv[1]


SEG_SIZE = 1000000
# # got these from cytoBand.txt online
# chrm_lens = [ 249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,59373566 ]
# # got these from UCSC genome browser (hg38)
# chrm_lens = [ 248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,57227415 ]

mc_data = np.load(mc_data_file_name)
array_m = mc_data["arrays"][0] # arbitrarily choose male array
chrm_begs = mc_data["chrm_begs"]
chrm_lens = mc_data["chrm_lens"]
M = array_m.shape[0]
T = array_m.shape[1]

NUM_CHRMS = chrm_begs.shape[0]
naive_K = sum([chrm_lens[i]//SEG_SIZE for i in range(NUM_CHRMS)])

chrm_begs = list(chrm_begs) + [M]
chrm_mut_lens = [chrm_begs[i+1] - chrm_begs[i] for i in range(NUM_CHRMS)]
chrm_Ks = [int(round(naive_K*(chrm_mut_lens[i]/M))) for i in range(NUM_CHRMS)]

print( "M = {}".format(M) )
print( "T = {}".format(T) )
print( "Naive K = {}".format(naive_K) )
print( "# of unique mutations per Chrm = {}".format(chrm_mut_lens) )
print( "Chrm Ks = {}".format(chrm_Ks) )