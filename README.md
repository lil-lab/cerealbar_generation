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
This research was supported by ARO W911NF21-1-0106, a Google Focused Award, Masason Foundation, a Facebook Fellowship, and NSF under grants No. 1750499 and DGE-1650441. We thank Jonathan Chang, Sasha Rush, the Cornell NLP Group, Robert Hawkins, Dipendra Misra, and John Langford for discussion and comments; Suyi Diao for Unity development; Anna Effenberger for code to compute syntax complexity; Ge Gao, Koji Shiono, and Takayuki Kojima for feedback on our interaction platform; and the crowdsourcing workers for participating in our data collection. Finally, we thank the action editor and the anonymous reviewers for detailed comments.
