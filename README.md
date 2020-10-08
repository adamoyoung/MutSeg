# Genome Gerrymandering
Optimal divison of the genome into regions with cancer type specific differences in mutation rates

## Abstract
The activity of mutational processes differs across the genome, and is influenced by chromatin state and spatial genome organization. At the scale of one megabase-pair (Mb), regional mutation density correlates strongly with chromatin features, and at this scale can be used to accurately identify cancer type. Here, we explore the relationship between genomic region and mutation rate by developing an information theory driven, dynamic programming algorithm for dividing the genome into regions with differing relative mutation rates between cancer types. Our algorithm improves mutual information when compared to the naive approach, effectively reducing the average number of mutations required to identify cancer type. This approach provides an efficient method for associating regional mutation density with mutation labels, and has future applications in exploring the role of somatic mutations in a number of diseases.

## Requirements
### Python 3.7
numpy, scipy, matplotlib, seaborn
### C
gcc, openmp

## PSB paper
link: https://psb.stanford.edu/psb-online/proceedings/psb20/Young.pdf

## Final paper
The final version of the paper is the file genome_gerrymandering_thesis_version.pdf and can be found in the repo
