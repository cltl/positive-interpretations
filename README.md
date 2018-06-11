# Scoring and Classifying Positive Interpretations

This repository contains the code for conducting the experiments as reported in the following paper:

> C. van Son, R. Morante, L. Aroyo, and P. Vossen. Scoring and Classifying Implicit Positive Interpretations: A Challenge of Class Imbalance. In *Proceedings of the 27th International Conference on Computational Linguistics (COLING 2018)*, Santa Fe, New Mexico, 2018.

It scores and classifies the positive interpretations generated from verbal negations in OntoNotes.

## Requirements
The Jupyter Notebooks in this repository have already been rendered, so that you can inspect the results. Please note, however, that in order to run the code, one has to first obtain the data:

- [OntoNotes 5.0](https://catalog.ldc.upenn.edu/ldc2013t19) / [OntoNotes 4.0](https://catalog.ldc.upenn.edu/ldc2011t03)
- [CoNLL-2011 Shared Task distribution of OntoNotes](http://conll.cemantix.org/2011)
- Positive Interpretations dataset: please contact [Eduardo Blanco](http://www.cse.unt.edu/~blanco/) or [Zahra Sarabi](http://zahrasarabi.com/) to obtain this data

The code has been tested with Python 3.6 and needs the following packages:
- nltk
- pandas
- scipy
- numpy
- scikit-learn

## Content

The repository contains the following folders:
- `code`: contains helper scripts and 4 notebooks for running the experiments
- `data`: the required data (see above) should be placed here
- `data_analysis`: contains the results of the `Data Analysis` notebook
- `results`: contains the feature files used for training/testing, the predictions and the summarizing tables/figures

The notebooks in the `code` folder can best be run in the following order:
- [1-Data_Preparation.ipynb](https://github.com/ChantalvanSon/positive-interpretations/blob/master/code/1-Data_Preparation.ipynb)
- [2-Data_Analysis.ipynb](https://github.com/ChantalvanSon/positive-interpretations/blob/master/code/2-Data_Analysis.ipynb)
- [3-Replication_Experiment.ipynb](https://github.com/ChantalvanSon/positive-interpretations/blob/master/code/3-Replication_Experiment.ipynb)
- [4-Error_Analysis.ipynb](https://github.com/ChantalvanSon/positive-interpretations/blob/master/code/4-Error_Analysis.ipynb)

## Contact

Chantal van Son (c.m.van.son@vu.nl / c.m.van.son@gmail.com)

Vrije Universiteit Amsterdam

