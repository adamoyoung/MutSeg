import numpy as np
import sys

SEG_SIZE = 1000000 # (1 MB)
NUM_CHRMS = 22 # no X or Y
# got these from online cytoBand.txt
#chrm_lens = [ 249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,59373566 ]
# got these from USCS genome browser (hg38)
chrm_lens = [ 248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,57227415 ]
float_t = np.float64
eps = np.finfo(float_t).eps

def one_seg(mc_data):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	total_Ms = np.zeros([len(chrm_begs)], dtype=float_t)
	for i in range(len(chrm_begs) - 1):
		total_Ms[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ])
	total_Ms[-1] = np.sum(array[ chrm_begs[-1] : ])

	T = array.shape[1]
	seg_scores = np.zeros([len(chrm_begs)], dtype=float_t)

	for i in range(len(chrm_begs) - 1):
		tumour_C = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ], axis=0)
		total_C = np.sum(tumour_C)
		total_M = total_Ms[i]
		print("chrm {}: total_C = {}, total_M = {}".format(i+1, total_C, total_M) )
		# assert(total_C == total_M)
		term_one =  np.sum( (tumour_C/total_M) * np.log( (tumour_C/total_M) + eps ))
		term_two = -( total_C / total_M ) * np.log( ( total_C / total_M) + eps )
		seg_scores[i] = term_one + term_two
	tumour_C = np.sum(array[ chrm_begs[-1] : ], axis=0)
	total_C = np.sum(tumour_C)
	total_M = total_Ms[-1]
	print("chrm {}: total_C = {}, total_M = {}".format(len(chrm_begs), total_C, total_M) )
	# assert(total_C == totals_M)
	term_one =  np.sum( (tumour_C/total_M) * np.log( (tumour_C/total_M) + eps ))
	term_two = -( total_C / total_M ) * np.log( ( total_C / total_M) + eps )
	seg_scores[-1] = term_one + term_two

	for i in range(len(chrm_begs)):
		print( "chromosome {}: {}".format(i+1, seg_scores[i]) )

def naive(mc_data, group_by):

	array = mc_data["array"]
	mut_pos = mc_data["mut_pos"]
	chrm_begs = mc_data["chrm_begs"]

	for i in range(len(mut_pos)):
		assert(len(mut_pos[i].shape) == 2)

	# compute total_M
	assert(len(chrm_begs) == len(mut_pos))
	total_Ms = np.zeros([len(chrm_begs)], dtype=float_t)
	for i in range(len(chrm_begs) - 1):
		total_Ms[i] = np.sum(array[ chrm_begs[i] : chrm_begs[i+1] ] )
	total_Ms[-1] = np.sum(array[ chrm_begs[-1] : ] )

	T = array.shape[1]
	seg_scores = [np.zeros([ chrm_lens[i] // SEG_SIZE ], dtype=float_t) for i in range(len(chrm_begs))]
	total_num_segs = sum([seg_scores[i].shape[0] for i in range(len(seg_scores))])
	print( [ chrm_lens[i] // SEG_SIZE for i in range(len(chrm_begs)) ] )
	print( total_num_segs )

	for i in range(len(chrm_begs)):
		#print( "chrm {}: largest mut_pos = {}, largest segment size = {}".format(i+1, mut_pos[i][-1][1], (chrm_lens[i] // SEG_SIZE)*SEG_SIZE) )
		cur_pos = 0
		chrm_beg = chrm_begs[i]
		chrm_mut_pos = mut_pos[i]
		#assert( (chrm_mut_pos[:,0] == chrm_mut_pos[:,1]).all() )
		chrm_score = seg_scores[i]
		total_M = total_Ms[i]
		for j in range( 1, (chrm_lens[i] // SEG_SIZE) - 1 ):
			beg_pt = (j-1)*SEG_SIZE
			end_pt = j*SEG_SIZE
			tumour_C = np.zeros([T], dtype=float_t)
			while cur_pos < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_pos][0] >= beg_pt and chrm_mut_pos[cur_pos][1] < end_pt:
				tumour_C += array[ chrm_beg+cur_pos ]
				cur_pos += 1
			# the case where a mutation crosses over a segment boundary: put it in the segment where most of it is
			if cur_pos < chrm_mut_pos.shape[0] and chrm_mut_pos[cur_pos][0] >= beg_pt and chrm_mut_pos[cur_pos][1] >= end_pt:
				if chrm_mut_pos[cur_pos][1] - end_pt < end_pt - chrm_mut_pos[cur_pos][0]:
					# put it in this segment
					tumour_C += array[ chrm_beg+cur_pos ]
					cur_pos += 1
				else:
					# put it in the next segment
					chrm_mut_pos[cur_pos][0] = chrm_mut_pos[cur_pos][1]
			total_C = np.sum(tumour_C)
			term_one = np.sum( (tumour_C/total_M) * np.log( (tumour_C/total_M) + eps ))
			term_two = -( total_C / total_M ) * np.log( ( total_C / total_M) + eps )
			chrm_score[j-1] = term_one + term_two
		# last segment gets all of mutations on the chrm that haven't been added yet
		if i == len(chrm_begs)-1:
			tumour_C = np.sum(array[ chrm_beg+cur_pos: ], axis=0)
		else:
			tumour_C = np.sum(array[ chrm_beg+cur_pos: chrm_begs[i+1] ], axis=0)
		total_C = np.sum(tumour_C)
		term_one = np.sum( (tumour_C / total_M) * np.log( (tumour_C / total_M) + eps ))
		term_two = -( total_C / total_M ) * np.log( ( total_C / total_M) + eps )
		chrm_score[-1] = term_one + term_two

	for i in range(len(seg_scores)):
		print( "chromosome {}: {}".format(i+1, np.sum(seg_scores[i])) )

	return seg_scores

if __name__ == "__main__":

	mc_file_name = sys.argv[1]
	output_file_name = sys.argv[2]
	group_by = sys.argv[3]

	mc_data = np.load(mc_file_name)

	#one_seg(mc_data)
	seg_scores = naive(mc_data, group_by)
	np.savez(output_file_name, seg_scores=seg_scores)