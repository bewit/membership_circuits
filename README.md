### Membership Circuits

The paper can be found [here](Wittelsbach26-Membership_Circuits_Tractable_Membership_Testing_Probabilistic_Circuits.pdf).

Have questions or comments? Want to discuss the paper or the general topic? Reach out!
Contact: bennet.wittelsbach@tu-darmstadt.de

This repository contains the code for the paper "Membership Circuits: Tractable Membership Testing via Probabilistic Circuits". MCs are a novel type of Probabilistic Circuits that provide formal guarantees on a hypothesis test regarding the membership of multivariate observations in learned distributions.

The required packages can be easily installed using uv.
The model and its algorithms are located in src/models/nodewise/.
The experiments are located in src/experiments/.
An introduction to the topic of membership testing and how to compute respective p-values in univariate and multivariate distributions is given in Hypothesis_Testing.ipynb.

The datasets for the out-of-distribution detection experiments can be downloaded from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF and need to be extracted into data/unsupervised\_outlier\_detection/.

