import sys
import os.path
import argparse
import pandas as pd
import numpy as np
import gzip
import pickle
import tarfile
import csv 
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Constants
TAR_FILE_15 = r"../../Data/Chromatin Annotations/15_State/all.dense.browserFiles.Tgz"
TAR_FILE_18 = r"../../Data/Chromatin Annotations/18_State/all.dense.browserFiles.Tgz"
CHROMATIN_STATES_DIR_15 = r"../../Data/Chromatin Annotations/15_State/chromatin_data"
CHROMATIN_STATES_DIR_18 = r"C:\Users\Kuba\Desktop\Work\Vector\Deeplift\Data\Chromatin Annotations\18_State\chromatin_data"
CELL_TYPES_RAW = r"../../Data/Chromatin Annotations/cell_types.xlsx"
CELL_TYPES_REFACTORED = r"../../Data/Chromatin Annotations/cell_types_refactored.xlsx"
DISCARD_CHRMS = ['chrX', 'chrY', 'chrM', '']
NUM_CHRMS = 22
RESULTS_UNCOMBINED_DIR_15 = r"C:/Users/Kuba/Desktop/Work/Vector/Deeplift/Data/Chromatin Annotations/15_State/results_uncombined/"
RESULTS_COMBINED_DIR_15 = r"C:/Users/Kuba/Desktop/Work/Vector/Deeplift/Data/Chromatin Annotations/15_State/results/"
RESULTS_UNCOMBINED_DIR_18 = r"C:/Users/Kuba/Desktop/Work/Vector/Deeplift/Data/Chromatin Annotations/18_State/results_uncombined/"
RESULTS_COMBINED_DIR_18 = r"C:/Users/Kuba/Desktop/Work/Vector/Deeplift/Data/Chromatin Annotations/18_State/results/"
OPT_SEG_BOUNDS_CSV = r"data/mutation_counts/opt_seg.csv"
NAIVE_SEG_BOUNDS_CSV = r"data/mutation_counts/naive_seg.csv"


# Maps labels to meaning according to roadmap chromatin state annotations for the 15 State model
class Transcription_factor_mapping_15_State():
    MAPPING={
        "1_TssA": "Active TSS",
        "2_TssAFlnk": "Flanking Active TSS",
        "3_TxFlnk": "Transcr. at gene 5 and 3",
        "4_Tx": "Strong Transcription",
        "5_TxWk": "Weak Transcription",
        "6_EnhG": "Genic Enhancers",
        "7_Enh": "Enhancers",
        "8_ZNF/Rpts": "ZNF genes and repeats",
        "9_Het": "Heterochromatin",
        "10_TssBiv": "Bivalent-Poised TSS",
        "11_BivFlnk": "Flaking Bivalent TSS-Enh",
        "12_EnhBiv": "Bivalent Enhancer",
        "13_ReprPC": "Repressed PolyComb",
        "14_ReprPCWk": "Weak Repressed PolyComb",
        "15_Quies": "Quiescent-Low",
    }

# Maps labels to meaning according to roadmap chromatin state annotations for the 18 State model
class Transcription_factor_mapping_18_State():
    MAPPING = {
        "1_TssA": "Active TSS",
        "2_TssFlnk": "Flanking Active TSS",
        "3_TssFlnkU": "Flanking Active TSS Upstream",
        "4_TssFlnkD": "Flanking Active TSS DownStream",
        "5_Tx": "Strong Transcription",
        "6_TxWk": "Weak Transcription",
        "7_EnhG1": "Genic enhancer1",
        "8_EnhG2": "Genic enhancer2",
        "9_EnhA1": "Active enhancer1",
        "10_EnhA2": "Active enhancer2",
        "11_EnhWk": "Weak Enhancer",
        "12_ZNF/Rpts": "ZNF genes and repeats",
        "13_Het": "Heterochromatin",
        "14_TssBiv": "Bivalent-Poised TSS",
        "15_EnhBiv": "Bivalent Enhancer",
        "11_BivFlnk": "Flaking Bivalent TSS-Enh",
        "16_ReprPC": "Repressed PolyComb",
        "17_ReprPCWk": "Weak Repressed PolyComb",
        "18_Quies": "Quiescent-Low",
    }

parser = argparse.ArgumentParser()
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--15_State', dest='new', action='store_false', help="Use 15 State Model")
feature_parser.add_argument('--18_State', dest='new', action='store_true', help="Use 18 State Model")
parser.set_defaults(new=True)



" ----------------------------- FUNCTIONS FOR CELL TYPE STUFF ----------------------------- "
def rgb_to_hex(rgb):
    """
    Return hex version of an rgb color code
    -> NOT USED FOR GENOME GERRY
    """
    rgb = rgb.split(',')
    rgb = int(rgb[0]), int(rgb[1]), int(rgb[2])

    return '#%02x%02x%02x' % rgb

def refactor_cell_types_data(excel_file_raw, excel_file_refactored):
    """
    Extract the columns we need from the original excel file of cell types
    -> NOT USED FOR GENOME GERRY
    """
    # Get df and refactor it
    df = pd.read_excel(excel_file_raw)
    df = df.iloc[:, [1, 3, 4, 7, 8]]
    df.columns = ["Epigenome ID (EID)", "Cell Group", "Color", "Anatomic Location", "Cell Type"]
    df = df.iloc[2:-8]
    df.to_excel(excel_file_refactored, index=None, header=True)

def extract_cell_types(cell_types):
    """
    Return a mapping from hex color codes to cell types and anatomic location based on the data provided
    -> NOT USED IN GENOME GERRY
    """
    color_to_cell_type = {}
    for i in range(len(cell_types)):
        color = cell_types['Color'][i].upper()

        # We already have cell type info from this color
        if color in color_to_cell_type.keys():
            continue
        else:
            group = cell_types["Cell Group"][i]
            type = cell_types["Cell Type"][i]
            loc = cell_types["Anatomic Location"][i]

            color_to_cell_type[color] = [group, type, loc]

    return color_to_cell_type

def append_cell_data_to_annotations(color_to_cell_type, all_data):
    """
    Append the cell type data onto chromatin annotations in all_data, by matching colors in color_to_cell_type
    -> NOT USED IN GENOME GERRY
    """
    for chr_data in all_data:
        hex_codes = chr_data["Hex Code"]
    
        for i in range(len(hex_codes)):
            hex_code = hex_codes[i]

            # Found matching cell type for this color
            if hex_code in color_to_cell_type.keys():
                results = color_to_cell_type[hex_code]
                chr_data["Cell Type"].append(results[0])
                chr_data["Cell Group"].append(results[1])
                chr_data["Anatomical Location"].append(results[2])

            # No data mapping this hex code to cell location
            else:
                chr_data["Cell Type"].append("Unknown")
                chr_data["Cell Group"].append("Unknown")
                chr_data["Anatomical Location"].append("Unknown")



" ----------------------------- HELPERS ----------------------------- "
def extract_tar_file(src, dest_directory):
    """
    Decompress all the files in the tar file at src to dest_directory
    """
    tf = tarfile.open()
    tf.extractall(path=dest_directory)

def aggregate_results(src_dir, result_dir):
    """
    Aggregate result from all src directory files, remove unneeded fields, and write for each chrom to a pkl file in results directory
    -> Since during the data extraction, I intermittently write to disc, I now collect each of those writes into one pkl object
    """
    for i in range(NUM_CHRMS):
        print("Chr: {}".format( i + 1))
        data_file = src_dir + "chr_{}_functional_annotations.pkl".format(i + 1)
        data = []
        with open(data_file, 'rb') as reader:
            try:
                while True:
                    data.append(pickle.load(reader))
            except EOFError:
                pass

        parsed_data = {"Start": [], "End": [], "Transcription Factor": [], "Cell Type": [], "Cell Group": [], "Anatomical Location": []}

        for j in range(len(data)):
            data_inst = data[j]

            parsed_data['Start'].extend(data_inst['Start'])
            parsed_data['End'].extend(data_inst['End'])
            parsed_data['Transcription Factor'].extend(data_inst['Transcription Factor'])

        # Sort by starting position
        start = parsed_data['Start']
        start = [int(start[i]) for i in range(len(start))]
        end = parsed_data['End']
        end = [int(end[i]) for i in range(len(end))]
        tr = parsed_data['Transcription Factor']

        all_data = list(zip(start, end, tr))
        all_data.sort()
        
        start_sorted = [start for start, end, tr in all_data]
        end_sorted = [end for start, end, tr in all_data]
        tr_sorted = [tr for start, end, tr in all_data]

        parsed_data = {"Start": start_sorted, "End": end_sorted, "Transcription Factor": tr_sorted}

        with open(result_dir + "chr_{}_functional_annotations.pkl".format(i + 1), 'wb') as writer:
            pickle.dump(parsed_data, writer, protocol=pickle.HIGHEST_PROTOCOL)

def interval_intersection_helper(functional_element_interval, segment_bounds, counts, ind):
    """
    Functional element interval, segment interval -> bp bounds to check intersections for
    Bounds -> list of all the segment bounds, used to check multiple segment boundary intersections
    Counts -> list of functional element counts in each segment
    Count -> count of functional element counts in current segment 
    Ind -> index into current element in bounds
    """
    curr_segment_interval = segment_bounds[ind]

    # Case 1: functional element completely out of current segment
    if functional_element_interval[0] >= curr_segment_interval[1]:

        # Since functional element intervals are sorted by base pair, all proceeding functional element can't be in current segment either.
        ind += 1
        curr_segment_interval = segment_bounds[ind]

        # Loop over segments until theres an intersection with functional element leading to case 2 or case 3
        while functional_element_interval[0] >= curr_segment_interval[1]:
            ind += 1
            curr_segment_interval = segment_bounds[ind]
    
    # Case 2: functional element completely contained within segment
    if curr_segment_interval[0] <= functional_element_interval[0] and curr_segment_interval[1] >= functional_element_interval[1]:

        # Add to current count and return, so that calling function moves onto next transcription factor
        counts[ind] += 1
        return segment_bounds, counts, ind

    # Case 3: functional element split between multiple segs

    # Create a list of sizes of overlap in all seg. We will take the largest and consider the functional element to be in that segment
    lengths_in_following_segs = []
    length_in_curr_seg = curr_segment_interval[1] - functional_element_interval[0]
    lengths_in_following_segs.append(length_in_curr_seg)

    i = 1

    # While the functional element overlaps with this segment
    while (ind + i) < len(segment_bounds) and functional_element_interval[1] > segment_bounds[ind + i][1]:
        length_in_next_seg = segment_bounds[ind + i][1] - segment_bounds[ind + i][0]
        lengths_in_following_segs.append(length_in_next_seg)
        i += 1
    length_in_next_seg = functional_element_interval[1] - segment_bounds[ind + i - 1][0]
    lengths_in_following_segs.append(length_in_next_seg)

    # Add count to the maximum segment intersection from the list
    max_seg_ind = lengths_in_following_segs.index(max(lengths_in_following_segs))
    counts[ind + max_seg_ind] += 1
    
    return segment_bounds, counts, ind



" ----------------------------- MAIN FUNCTIONS ----------------------------- "

def extract_chromatin_annotations(chromatin_data, all_data, model_18):
    """
    Mutate all_data dictionary with chromatin annotation data for each autosome based on data files from Roadmap Epigenomics.
    -> chromatin_data contains data from a single file (chromatin start, chromatin end, transcription factor, color_code)
    """
    if model_18:
        tr_map = Transcription_factor_mapping_18_State()
    else:
        tr_map = Transcription_factor_mapping_15_State()
    
    for sample in chromatin_data:
        chr = sample[0]

        # Don't need data for this
        if chr in DISCARD_CHRMS:
            continue

        # Fields that we care about
        start = sample[1]
        end = sample[2]
        transcription_factor = tr_map.MAPPING[sample[3]]
        color_rgb = sample[-1]
        
        color_hex = rgb_to_hex(color_rgb)
        chr_index = int(chr[3:]) - 1

        all_data[chr_index]["Start"].append(start)
        all_data[chr_index]["End"].append(end)
        all_data[chr_index]["Transcription Factor"].append(transcription_factor)
        all_data[chr_index]["Hex Code"].append(color_hex)


def find_functional_groupings_per_segments(chrm, tr, opt_seg_csv, naive_seg_csv):
    """
    Returns the counts of each chromatin function in each segment from naive and opt segmentations
    """
    tr_map = Transcription_factor_mapping_18_State()

    # Get segment boundaries opt
    bounds_opt = []
    with open(opt_seg_csv, 'r+') as f:
        reader = csv.reader(f, delimiter=',', lineterminator='\n')
        for i, row in enumerate(reader):
            if int(row[0]) == chrm:
                bounds_opt.append((int(row[1]), int(row[2])))

    # Get segment boundaries naive
    bounds_naive = []
    with open(naive_seg_csv, 'r+') as f:
        reader = csv.reader(f, delimiter=',', lineterminator='\n')
        for i, row in enumerate(reader):
            if int(row[0]) == chrm:
                bounds_naive.append((int(row[2]), int(row[3])))

    # get chromatin data
    with open(RESULTS_COMBINED_DIR_18 + "chr_{}_functional_annotations.pkl".format(chrm), 'rb') as reader:
        results = pickle.load(reader)

    assert len(bounds_opt) == len(bounds_naive), "Length mismatch in number of segments."

    # Counts opt and counts naive are arrays of size "num segments" where element at index i is how many counts of transcription
    # factor tr in the ith segment of chrm
    counts_opt = np.zeros(len(bounds_opt), dtype=int)
    counts_naive = np.zeros(len(bounds_naive), dtype=int)
    ind_opt = 0
    ind_naive = 0

    # Theres a slight miscount here that needs to be fixed (counts_opt and counts_naive should sum to the same thing)
    test = 0 
    for i in range(len(results['Transcription Factor'])):
        if results['Transcription Factor'][i] == tr:
            test += 1
            
            functional_element_interval = (results['Start'][i], results['End'][i])
            
            # For naive
            bounds_naive, counts_naive, ind_naive = interval_intersection_helper(functional_element_interval, 
            bounds_naive, counts_naive, ind_naive)

            # For opt
            bounds_opt, counts_opt, ind_opt = interval_intersection_helper(functional_element_interval, 
            bounds_opt, counts_opt, ind_opt)

    
    # Total number of tr occurences needs to be the same regardless of which segment they are counted in
    assert test == sum(counts_naive) and test == sum(counts_opt)

    return counts_naive, counts_opt



" ----------------------------- PLOTTING FUNCTIONS ----------------------------- "
"""
Data_opt and data_naive contain a list for each transcription factor (trs)
Each of these lists contain same number of elements as there are segments in chrm, with ith element denoting how many occurences of a transcription factor there are in that segment.
"""
def plot_kernel_estimates(data_opt, data_naive, trs, chrm):
    f, axes = plt.subplots(3, 6, figsize=(35, 35), sharex=False)
    ind = 0
    for i in range(3):
        for j in range(6):
    
            # tr[15] has no data in chrom 1
            if ind != 15:
                sns.kdeplot(data_naive[ind], bw=sum(data_naive[ind]) / 1700, shade=True, ax=axes[i][j], color='lightskyblue')
                sns.kdeplot(data_opt[ind], bw=sum(data_naive[ind]) / 1700, shade=True, ax=axes[i][j], color='lightcoral')

            axes[i][j].set_yticks([])
            axes[i][j].set_xticks([])
            axes[i][j].set_ylim(bottom=0)
            axes[i][j].set_xlim(0, np.percentile(np.array(data_opt[ind]), 90))

            # Shorten some chromatin state names to fit on the plots
            if ind == 2:
                axes[i][j].set_title(
                    'Flnk. Act. TSS Upstream' + " ({})".format(sum(data_naive[ind])), fontsize=9)
            elif ind == 3:
                axes[i][j].set_title(
                    'Flnk. Act. TSS Downstream' + " ({})".format(sum(data_naive[ind])), fontsize=9)
            else:
                axes[i][j].set_title(
                    trs[ind] + " ({})".format(sum(data_naive[ind])), fontsize=9)
            ind += 1

    red_patch = mpatches.Patch(color='lightcoral', label='Opt')
    blue_patch = mpatches.Patch(color='lightskyblue', label='Naive')
    f.legend(handles=[red_patch, blue_patch])
    plt.suptitle('Chromatin States Frequency Distribution Across Segments: Chromosome {}'.format(chrm), fontsize=20)
    plt.show()

"""
Get a list of counts of each transcription factor along the segments of a specific chromosome, and save that data to disc
"""
def assemble_plot_data(tr_map, chrm, filename):
    data_opt = []
    data_naive = []
    trs = []
    for tr in tr_map.MAPPING.values():
        print(tr)
        counts_naive, counts_opt = find_functional_groupings_per_segments(chrm, tr, opt_seg_csv=OPT_SEG_BOUNDS_CSV, naive_seg_csv=NAIVE_SEG_BOUNDS_CSV)
        data_opt.append(counts_opt)
        data_naive.append(counts_naive)
        trs.append(tr)

    with open(filename, 'wb') as fp:
        pickle.dump([data_opt, data_naive, trs], fp)
        print("Serialized")


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    if FLAGS.new:
        print("Using 18 State Model")
        TAR_FILE = TAR_FILE_18
        CHROMATIN_STATES_DIR = CHROMATIN_STATES_DIR_18
        RESULTS_UNCOMBINED_DIR =  RESULTS_UNCOMBINED_DIR_18
        RESULTS_COMBINED_DIR =  RESULTS_COMBINED_DIR_18
        model_18 = True
    else:
        print("Using 15 State Model")
        TAR_FILE = TAR_FILE_15
        CHROMATIN_STATES_DIR = CHROMATIN_STATES_DIR_15
        RESULTS_UNCOMBINED_DIR = RESULTS_UNCOMBINED_DIR_15
        RESULTS_COMBINED_DIR = RESULTS_COMBINED_DIR_15
        model_18 = False

    if len(os.listdir(CHROMATIN_STATES_DIR)) == 0:
        extract_tar_file(TAR_FILE, dest_directory="")

    if False:
        # Get cell type data
        if os.path.isfile(CELL_TYPES_REFACTORED):
            cell_types = pd.read_excel(CELL_TYPES_REFACTORED)
        else:
            refactor_cell_types_data(CELL_TYPES_RAW, CELL_TYPES_REFACTORED)
            cell_types = pd.read_excel(CELL_TYPES_REFACTORED)

        color_to_cell_type = extract_cell_types(cell_types)

    # Get a list of all the data files
    source_paths = []
    entries = sorted(os.listdir(CHROMATIN_STATES_DIR))
    for entry in entries:
        entry_path = os.path.join(CHROMATIN_STATES_DIR, entry)
        source_paths.append(entry_path)

    print("\nThere are {} data files being processed.\n".format(len(source_paths)))

    all_data = [{"Start": [], "End": [], "Transcription Factor": [], "Hex Code": [], "Cell Type": [], "Cell Group": [], "Anatomical Location": []} for i in range(NUM_CHRMS)]
    filenames = ["chr_{}_functional_annotations.pkl".format(i + 1) for i in range(NUM_CHRMS)]

    if False:
        for idx, data_file in enumerate(source_paths):
            with gzip.open(data_file, "rb") as f:
                chromatin_data = f.read().decode("utf-8").split("\n")
                chromatin_data = [chromatin_data[i].split("\t") for i in range(len(chromatin_data))][1:]

                # Extract fields
                extract_chromatin_annotations(chromatin_data, all_data, model_18)
                
                if False:
                    # Append cell type info onto chromatin annotations
                    append_cell_data_to_annotations(color_to_cell_type, all_data)

            for i, filename in enumerate(filenames):
                with open(filename, 'ab') as writer:
                    pickle.dump(all_data[i], writer, protocol=pickle.HIGHEST_PROTOCOL)

            all_data = [{"Start": [], "End": [], "Transcription Factor": [], "Hex Code": [], "Cell Type": [], "Cell Group": [], "Anatomical Location": []} for i in range(NUM_CHRMS)]


            if idx % 2 == 0:
                print("{}%".format(round(100 * idx / len(source_paths), 2)))


        aggregate_results(src_dir=RESULTS_UNCOMBINED_DIR, result_dir=RESULTS_COMBINED_DIR)
    
    
    tr_map = Transcription_factor_mapping_18_State()
    chrm = 1
    fname = "chrm_{}_groupings".format(chrm)

    if False:
        assemble_plot_data(tr_map, chrm, fname)
    
    with open(fname, 'rb') as fp:
        itemlist = pickle.load(fp)
        
    data_opt = itemlist[0]
    data_naive = itemlist[1]
    trs = itemlist[2]

    plot_kernel_estimates(data_opt, data_naive, trs, chrm)
    
