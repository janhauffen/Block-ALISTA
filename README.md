# Block-ALISTA 

We present deep unfolding using the block iterative shrinkage thresholding algorithm (BISTA). The general idea of deep unfolding is that we interpret an iterative algorithm as a neural network and learn the weights through data-driven optimization. We then apply this to BISTA and thus present learned BISTA (LBISTA). We also present Analytical LBISTA, where we compute the analytical weight matrix through data-free optimization. Furthermore we provide two solution strategies to solve this data-free optimization problem.

# Run Code

0. To test the code, we provided all dependencies inside the dependencies.txt and a conda environment environment.yml. We highly recommend to use a conda virtual environment to install the dependencies. Additionally the cvxpy toolbox is necessary for some scripts. 
1. We provide the main scripts, ```run_***_studies.py```. These train the networks and save the NMSE over iterations, reconstructions and the regularization parameters in the /mat folder.
2. We also provide with ```Example_Optimize_GenBlockCoheherence.ipynb``` a jupyter-notebook, based on the cvxpy-package, to examine the two solution strategies to solve the data-free optimization problem.
