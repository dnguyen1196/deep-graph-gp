




# Experiments

## Smaller data sets

fMRI data set
* Collection of 4465 nodes corresponding to different regions of the brain
* Each comes with 292 features in "signal_set_cerebellum.mat"
* Adjacency matrix {0,1} comes in "A_cerebellum.mat"
* 84974 edges (How they assign these edges = https://miplab.epfl.ch/pub/behjat1601p.pdf, edges are assigned by computing connections between adjacent voxels in a 3d neighborhood)



### Data with temporal element

Weather data set
* Collection of temperatures across 95 days in 45 cities in Sweeden
* Adjacency matrix in "city45data.mat" computed from coordinates of these cities
* Temperature "smhi_temp_17.mat"
* Thought: can one predict temperature trajectory instead, comparing between graph GP and just classical GP

Atmospheric tracer diffusion data
* Concentration of perfluorocarbon tracers over 72 hours at 168 locations 
* Adjacency matrix in "A_etex.mat"
* Two sets of measurements in "etex_1.mat" and "etex_2.mat"


## Bigger data sets

Many other data sets
https://linqs.soe.ucsc.edu/data

Cora
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. https://relational.fit.cvut.cz/dataset/CORA


Cite seer
3.3k nodes with 4.5k edges, each node has a class label (6 classess). Maximum degree 99, min degree 1. Problem, there are no features in this data sets
@inproceedings{nr,
    title={The Network Data Repository with Interactive Graph Analytics and Visualization},
    author={Ryan A. Rossi and Nesreen K. Ahmed},
    booktitle={AAAI},
    url={http://networkrepository.com},
    year={2015}
}

## Very big data sets

Pubmed-diabetes
The Pubmed Diabetes dataset consists of 19717 scientific publications (nodes) from PubMed database pertaining to diabetes classified into one of three classes. 
The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words. 
The README file in the dataset provides more details.

Issue: a lot of missing features (nan)

