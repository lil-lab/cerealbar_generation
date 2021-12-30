# Data

### Downloading human-human interaction data.
Download processed human-human interaction data from [the link](https://drive.google.com/file/d/1L0YIYJJd1jQhY8Fethfg2t1eawG7OTaC/view?usp=sharing) and unzip the folder under `/data`. Our data is based on the human-human game record collected by [Suhr et al. 2019](https://aclanthology.org/D19-1218.pdf). `TRAIN_360_subset_games.txt` specified the subset of human-human data we used for pre-training.

### Downloading human-system interaction data.
We provide the human-system interaction data for 14 rounds of long-term user studies (corresponding to Fig.5 of [our paper](https://arxiv.org/pdf/2108.04812.pdf)). 
Download and unzip each folders file under `/data`:
- [`/human_system.zip`](https://drive.google.com/file/d/1z0vVy5hbGAyAo9dxcE5VzIXw2uLTbO6O/view?usp=sharing) is the folders of interaction data in each round. 
- [`/ips_probs.zip`](https://drive.google.com/file/d/1loUUI6h0pCLbXSNl7yrsQa2NB8SqfxBP/view?usp=sharing) is .pkl files from each interaction round containing the sentence probability of each model-generated instruction during deployment (this is necessary for calculating IPS coefficient of negative instructions during training). 
- [`/data_splits.zip`](https://drive.google.com/file/d/1G0HkXyxzu-kIa2Sl5_-2Byh2DOQlvYOK/view?usp=sharing) contains folders of text files assigning either positive or negative labels to each interaction data and specifying the system's plan and/or user execution to use during training.
Note that `/ips_probs` and `/data_splits` miss the data for round 14 since round14 was conducted for human evaluation purpose-only  (i.e., the data from round 14 was not used for training new models.) 
