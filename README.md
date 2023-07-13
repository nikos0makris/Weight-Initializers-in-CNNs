# Weight Initialization Algorithms, Implementation on CNN 
This project utilizes 3 convolutional neural networks (1 known, 2 custom) to determine which weight initialization
algorithms are the most efficient, in a series of classifiaction tasks. Famous researchers have proposed these algorithms,
such as LeCun, Glorot and He, and these experiments showcase the evaluation metrics that each of the models have scored, on 4 different
datasets provided by the Keras API (MNIST, Fashion MNIST, Cifar-10, Cifar-100). The purpose is to highlight the most efficient
metrics per algorithm and to further research hyperparameter tuning in general.

# Requirements
This project uses Python 3.5 and the PIP following packages:
-Tensorflow (and Keras API)
-Numpy
-Scikit-Learn
-MatplotLib
-Optuna
All scripts were runned on a mini-conda enviroment, after the above packages were installed. It is suggested to utilize a
high-end graphics card in your system (such as RTX 3090 used here) to shorten the running time of this demanding scripts.

# Struct
-The files contain all Python code needed to conduct the experiments, and are specified as: model_dataset.py 
(e.g lenet_c100.py) to separate and conduct each experiment solely. The models used are the LeNet, as constructed
by the famous Yann LeCun, and 2 custom models of mine, named ModNet1 and ModNet2, which are considered deeper neural networks.
-There are also 3 alternative models proposed, where the Optuna software conducted trials to determine a better 
learning rate and filters numbers for each model, and the 3 models where trained again on the Cifar-100 dataset.

# Results
-Each script showcases (using MatplotLib) graphs of the Accuracy and Loss metrics gathered in the training process, for 5
iterations. Then, it evaluates the model in the training and testing dataset, and prints the average metrics including
Precision, Recall and F1-Score (for each algorithm) after 5 iterations.
-On the optimization scripts, Optuna conducts 10 trials on each of the 3 models to research for better hyperparameters, which
are then replaced on the initial scripts for another training process.
