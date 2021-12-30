# Continual Learning for Grounded Instruction Generation by Observing Human Following Behavior

This is the code repository for the paper: "Continual Learning for Grounded Instruction Generation by Observing Human Following Behavior", [Noriyuki Kojima](https://kojimano.github.io/), [Alane Suhr](http://alanesuhr.com/), and [Yoav Artzi](https://yoavartzi.com/) (TACL 2021, presented at EMNLP 2021).
 
 
## About
[paper](https://arxiv.org/abs/2108.04812) | [talk](https://www.youtube.com/watch?v=KkgIMPTS7H0&t=1s) | [project page](https://lil.nlp.cornell.edu/cerealbar/)

We study continual learning for natural language instruction generation, by observing human users' instruction execution. We focus on a collaborative scenario, where the system both acts and delegates tasks to human users using natural language. We compare user execution of generated instructions to the original system intent as an indication to the system's success communicating its intent. We show how to use this signal to improve the system's ability to generate instructions via contextual bandit learning. In interaction with real users, our system demonstrates dramatic improvements in its ability to generate language over time.

<p align="center">
 <img src="media/tacl2021.gif" width="500" align=/>
</p>

# Codebase

## Installation

1. Create the conda environment: 
```conda create -n cb_gen python=3.7```
1. Make sure that git-LFS is installed before cloning the repo.
1. Clone the repo.
1. Install the requirements: ```pip install -r requirements.txt```

 
## Subdirectories
- `model/` defines the model architectures for instruction generation in the CerealBar
- `agents/` defines information about the CerealBar agent and environment
- `learning/` is used to train models.
- `data/` contains data and define scripts for processing data. 
- `chekcpoints/` contains model checkpoints. 


## Training models

### Pre-training on human-human interaction data  
1. Download formatted GPT-2 weights from [the link](https://drive.google.com/file/d/1UZRXftmNhUIf8iR3g5BoiWNShHcNvlZR/view?usp=sharing) and put it under `/checkpoints/gpt-2`
1. Download processed human-human interaction data from the link [the link](https://drive.google.com/file/d/1W6KgB6CL-TYOBB3vL9h7u4qmfGq_edCw/view?usp=sharing) and unzip the folder under `/data`.
```
python -m learning.training --train_config_file_name learning/configs/pretraining.yml --experiment_name pretraining --max_epochs 400 --checkpoint_step 40000 --turnoff_wandb
```
### Training a model from scratch on human-human & human-system interaction data
1. Download processed human-system interaction data from the link [the link](https://drive.google.com/file/d/1W6KgB6CL-TYOBB3vL9h7u4qmfGq_edCw/view?usp=sharing) and put them under `/data`.
```
# This example is training a model on the aggregated data after the second round of human-system interactions.
python -m learning.training --train_config_file_name learning/configs/continual_example.yml --experiment_name pretraining --max_epochs 400 --checkpoint_step 40000 --turnoff_wandb
```

## Analyzing models
### Downloading model checkpoints
1. Please refer `/checkpoints/README.md` to download checkpoints for the trained models.
```
```

## Notes
This current version of the repo does not include a pipeline for human-system interactions (e.g., the backend server to communicate with Unity game, a deterministic planner to generate system's path-plan and so on.) 

### License
MIT

## Citing
If you find this our work useful in your research, please consider citing the following paper:
```
@article{Kojima2021:gen-learn,
  author  = {Kojima, Noriyuki and Suhr, Alane and Artzi, Yoav},
  title   = {Continual Learning for Grounded Instruction Generation by Observing Human Following Behavior},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {9},
  pages   = {1303-1319},
  year    = {2021},
  month   = {12},
  issn    = {2307-387X},
  doi     = {10.1162/tacl_a_00428},
  url     = {https://doi.org/10.1162/tacl\_a\_00428},
  eprint  = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00428/1976207/tacl\_a\_00428.pdf}
}
```

## Ackowledegement
This research was supported by ARO W911NF21-1-0106, a Google Focused Award, Masason Foundation, a Facebook Fellowship, and NSF under grants No. 1750499 and DGE-1650441. We thank Jonathan Chang, Sasha Rush, the Cornell NLP Group, Robert Hawkins, Dipendra Misra, and John Langford for discussion and comments; Suyi Diao for Unity development; Anna Effenberger for code to compute syntax complexity; Ge Gao, Koji Shiono, and Takayuki Kojima for feedback on our interaction platform; and the crowdsourcing workers for participating in our data collection. Finally, we thank the action editor and the anonymous reviewers for detailed comments.
