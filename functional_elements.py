import matplotlib.pyplot as plt
import sys
import pandas as pd
import gzip
import operator
import os.path
import pickle
import tarfile
import argparse
import csv 
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import pickle

# Constants
TAR_FILE_15 = r"../../Data/Chromatin Annotations/15_State/all.dense.browserFiles.Tgz"
TAR_FILE_18 = r"../../Data/Chromatin Annotations/18_State/all.dense.browserFiles.Tgz"
CHROMATIN_STATES_DIR_15 = r"../../Data/Chromatin Annotations/15_State/chromatin_data"
CHROMATIN_STATES_DIR_18 = r"../../Data/Chromatin Annotations/18_State/chromatin_data"
CELL_TYPES_RAW = r"../../Data/Chromatin Annotations/cell_types.xlsx"
CELL_TYPES_REFACTORED = r"../../Data/Chromatin Annotations/cell_types_refactored.xlsx"
DISCARD_CHRMS = ['chrX', 'chrY', 'chrM', '']
NUM_CHRMS = 22
RESULTS_UNCOMBINED_DIR_15 = r"../../Data/Chromatin Annotations/15_State/results_uncombined/"
RESULTS_COMBINED_DIR_15 = r"../../Data/Chromatin Annotations/18_State/results/"
RESULTS_UNCOMBINED_DIR_18 = r"../../Data/Chromatin Annotations/18_State/results_uncombined/"
RESULTS_COMBINED_DIR_18 = r"../../Data/Chromatin Annotations/18_State/results/"
OPT_SEG_BOUNDS_CSV = r"../../../Genome GerryMandering/MutSeg_Jacob/data/mutation_counts/opt_seg.csv"
NAIVE_SEG_BOUNDS_CSV = r"../../../Genome GerryMandering/MutSeg_Jacob/data/mutation_counts/naive_seg.csv"


# Maps labels to meaning according to roadmap chromatin state annotations
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


"""
Decompress all the files in the tar file
"""
def extract_tar_file(path):
    tf = tarfile.open(path)
    tf.extractall()

"""
Return hex version of an rgb color code
"""
def rgb_to_hex(rgb):
    rgb = rgb.split(',')
    rgb = int(rgb[0]), int(rgb[1]), int(rgb[2])

    return '#%02x%02x%02x' % rgb

"""
Extract the columns we need from the original excel file of cell types
"""
def refactor_cell_types_data(excel_file_raw, excel_file_refactored):
    # Get df and refactor it
    df = pd.read_excel(excel_file_raw)
    df = df.iloc[:, [1, 3, 4, 7, 8]]
    df.columns = ["Epigenome ID (EID)", "Cell Group", "Color", "Anatomic Location", "Cell Type"]
    df = df.iloc[2:-8]

    df.to_excel(excel_file_refactored, index=None, header=True)

"""
Return a mapping from hex color codes to cell types and anatomic location
"""
def extract_cell_types(cell_types):
    color_to_cell_type = {}

    for i in range(len(cell_types)):
        color = cell_types['Color'][i].upper()

        # We already have this data point
        if color in color_to_cell_type.keys():
            continue
        else:
            group = cell_types["Cell Group"][i]
            type = cell_types["Cell Type"][i]
            loc = cell_types["Anatomic Location"][i]

            color_to_cell_type[color] = [group, type, loc]

    return color_to_cell_type

"""
Return a dictionary with chromatin annotation data for each autosome
"""
def extract_chromatin_annotations(chromatin_data, all_data, model_18):
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

"""
Append the cell type data onto chromatin annotations
"""
def append_cell_data_to_annotations(color_to_cell_type, all_data):
    for chr_to_chromatin in all_data:
        hex_codes = chr_to_chromatin["Hex Code"]
    
        for i in range(len(chr_to_chromatin["Hex Code"])):
            hex_code = hex_codes[i]

            if hex_code in color_to_cell_type.keys():
                results = color_to_cell_type[hex_code]
                print("SUCCESS")
                chr_to_chromatin["Cell Type"].append(results[0])
                chr_to_chromatin["Cell Group"].append(results[1])
                chr_to_chromatin["Anatomical Location"].append(results[2])

            # No data mapping this hex code to cell location
            else:
                chr_to_chromatin["Cell Type"].append("Unknown")
                chr_to_chromatin["Cell Group"].append("Unknown")
                chr_to_chromatin["Anatomical Location"].append("Unknown")

"""
Aggregate result from all src directory files, remove unneeded fields, and write to pkl file in results directory
"""
def aggregate_results(src_dir, result_dir):
    for i in range(NUM_CHRMS):
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
            parsed_data['Cell Type'].extend(data_inst['Cell Type'])
            parsed_data['Cell Group'].extend(data_inst['Cell Group'])
            parsed_data['Anatomical Location'].extend(data_inst['Anatomical Location'])


        with open(result_dir + "chr_{}_functional_annotations.pkl".format(i + 1), 'wb') as writer:
            pickle.dump(parsed_data, writer, protocol=pickle.HIGHEST_PROTOCOL)


"""
Returns the counts of each chromatin function in each segment from naive and opt segmentations
"""
def find_functional_groupings_per_segments(chrm, tr, opt_seg_csv, naive_seg_csv):
    tr_map = Transcription_factor_mapping_18_State()

    bounds_opt = []
    with open(opt_seg_csv, 'r+') as f:
        reader = csv.reader(f, delimiter=',', lineterminator='\n')
        for i, row in enumerate(reader):
            if int(row[0]) == chrm:
                bounds_opt.append(row[2])

    bounds_naive = []
    with open(naive_seg_csv, 'r+') as f:
        reader = csv.reader(f, delimiter=',', lineterminator='\n')
        for i, row in enumerate(reader):
            if int(row[0]) == chrm:
                bounds_naive.append(row[3])

    with open(RESULTS_COMBINED_DIR_18 + "chr_{}_functional_annotations.pkl".format(chrm), 'rb') as reader:
        results = pickle.load(reader)

    counts_opt = []
    counts_naive = []
    count_opt = 0
    count_naive = 0
    ind_opt = 0
    ind_naive = 0
    
    # Theres a slight miscount here that needs to be fixed (counts_opt and counts_naive should sum to the same thing)
    for i in range(len(results['Transcription Factor'])):
        if results['Transcription Factor'][i] == tr:
            count_opt += 1
            count_naive += 1

        if int(results['End'][i]) > int(bounds_opt[ind_opt]):
            counts_opt.append(count_opt)
            count_opt = 0
            ind_opt += 1

        if int(results['End'][i]) > int(bounds_naive[ind_naive]):
            counts_naive.append(count_naive)
            count_naive = 0
            ind_naive += 1

    return counts_naive, counts_opt


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


"""
Data_opt and data_naive contain a list for each transcription factor (trs)
Each of these lists contain same number of elements as there are segments in chrm, with ith element denoting how many occurences of a transcription factor there are in that segment.
"""
def plot_kernel_estimates(data_opt, data_naive, trs, chrm):
    f, axes = plt.subplots(3, 6, figsize=(35, 35), sharex=True)
    ind = 0
    for i in range(3):
        for j in range(6):

            # tr[15] has no data in chrom 1
            if ind != 15:
                sns.kdeplot(data_naive[ind], bw=1, shade=True,
                            ax=axes[i][j], color='lightskyblue')
                sns.kdeplot(data_opt[ind], bw=1, shade=True,
                            ax=axes[i][j], color='lightcoral')
            axes[i][j].set_yticks([])
            axes[i][j].set_xticks([])
            axes[i][j].set_xlim(0, 50)
            axes[i][j].set_ylim(bottom=0)

            # Shorten some chromatin state names to fit on the plots
            if ind == 2:
                axes[i][j].set_title('Flnk. Act. TSS Upstream')
            elif ind == 3:
                axes[i][j].set_title('Flnk. Act. TSS Downstream')
            else:
                axes[i][j].set_title(trs[ind])
            ind += 1

    red_patch = mpatches.Patch(color='lightcoral', label='Opt')
    blue_patch = mpatches.Patch(color='lightskyblue', label='Naive')
    f.legend(handles=[red_patch, blue_patch])
    plt.suptitle('Chromatin States Frequency Distribution Across Segments: Chromosome {}'.format(
        chrm), fontsize=20)
    plt.show()


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
        extract_tar_file(TAR_FILE)

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

        for idx, data_file in enumerate(source_paths):
            with gzip.open(data_file, "rb") as f:
                chromatin_data = f.read().decode("utf-8").split("\n")
                chromatin_data = [chromatin_data[i].split("\t") for i in range(len(chromatin_data))][1:]

                # Extract fields
                extract_chromatin_annotations(chromatin_data, all_data, model_18)
                
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

    plot_kernel_estimates(data_opt, data_naive, trs. chrm)
    
