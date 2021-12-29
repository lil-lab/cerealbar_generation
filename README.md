This is the code repository for the paper: "Continual Learning for Grounded Instruction Generation by Observing Human Following Behavior", [Noriyuki Kojima](https://kojimano.github.io/), [Alane Suhr](http://alanesuhr.com/) and [Yoav Artzi](https://yoavartzi.com/) (to be appeared at TACL 2021).
 
 
### About
[paper](https://arxiv.org/abs/2108.04812)| [talk](https://www.youtube.com/watch?v=KkgIMPTS7H0&t=1s) | [project page](https://lil.nlp.cornell.edu/cerealbar/)

e study continual learning for natural language instruction generation, by observing human users' instruction execution. We focus on a collaborative scenario, where the system both acts and delegates tasks to human users using natural language. We compare user execution of generated instructions to the original system intent as an indication to the system's success communicating its intent. We show how to use this signal to improve the system's ability to generate instructions via contextual bandit learning. In interaction with real users, our system demonstrates dramatic improvements in its ability to generate language over time.

![](miscs/parser.gif)


## Codebase

### Contents
1. 
## Model

## Data

## Annotaion UI
### Unity Webapp
### Python Model Server

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
We would like to thank [Freda](https://ttic.uchicago.edu/~freda/) for making their code (the code in this repo is largely borrowed from the original VGNSL implementation) public and responding promptly to our inquiry on [Visually Grounded Neural Syntax Acquisition](https://ttic.uchicago.edu/~freda/project/vgnsl/) (Shi et al., ACL2019). 

