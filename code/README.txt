Weight Initialization Algorithms, Implementation on CNN 
This project utilizes 3 convolutional neural networks (1 known, 2 custom) to determine which weight initialization
algorithms are the most efficient, in a series of classifiaction tasks. Famous researchers have proposed these algorithms,
such as LeCun, Glorot and He, and these experiments showcase the evaluation metrics that each of the models have scored, on 4 different
datasets provided by the Keras API (MNIST, Fashion MNIST, Cifar-10, Cifar-100). The purpose is to highlight the most efficient
metrics per algorithm and to further research hyperparameter tuning in general.

Requirements
This project uses Python 3.5 and the PIP following packages:
-Tensorflow (and Keras API)
-Numpy
-Scikit-Learn
-MatplotLib
-Optuna
All scripts were runned on a mini-conda enviroment, after the above packages were installed. It is suggested to utilize a
high-end graphics card in your system (such as RTX 3090 used here) to shorten the running time of this demanding scripts.

Struct
-The files contain all Python code needed to conduct the experiments, and are specified as: model_dataset.py 
(e.g lenet_c100.py) to separate and conduct each experiment solely. The models used are the LeNet, as constructed
by the famous Yann LeCun, and 2 custom models of mine, named ModNet1 and ModNet2, which are considered deeper neural networks.
-There are also 3 alternative models proposed, where the Optuna software conducted trials to determine a better 
learning rate and filters numbers for each model, and the 3 models where trained again on the Cifar-100 dataset.

Results
-Each script showcases (using MatplotLib) graphs of the Accuracy and Loss metrics gathered in the training process, for 5
iterations. Then, it evaluates the model in the training and testing dataset, and prints the average metrics including
Precision, Recall and F1-Score (for each algorithm) after 5 iterations.
-On the optimization scripts, Optuna conducts 10 trials on each of the 3 models to research for better hyperparameters, which
are then replaced on the initial scripts for another training process.

--------------------------------------------------------------------------------------------------------------------------------

Αλγόριθμοι αρχικοποίησης βαρών, υλοποίηση σε συνελικτικά νευρωνικά δίκτυα
Αυτό το project χρησιμοποιεί 3 συνελικτικά νευρωνικά δίκτυα (1 γνωστό, 2 προσαρμοσμένα) για να προσδιορίσει ποιοί αλγόριθμοι αρχικοποίησης
βαρών είναι οι πιο αποτελεσματικοί, σε μια σειρά πειραμάτων ταξινόμησης. Διάσημοι ερευνητές έχουν προτείνει αυτούς τους αλγόριθμους,
όπως οι LeCun, Glorot και He, και αυτά τα πειράματα παρουσιάζουν τις μετρικές αξιολόγησης που έχει σημειώσει καθένα από τα μοντέλα, σε 4 διαφορετικά
σύνολα δεδομένων που παρέχονται από το Keras API (MNIST, Fashion MNIST, Cifar-10, Cifar-100). Σκοπός είναι η ανάδειξη των πιο αποτελεσματικών
μετρικών ανά αλγόριθμο και για περαιτέρω έρευνα του συντονισμού υπερπαραμέτρων γενικότερα.

Απαιτήσεις
Αυτό το έργο χρησιμοποιεί Python 3.5 και τα ακόλουθα πακέτα PIP:
-Tensorflow (και Keras API)
-Numpy
-Scikit-Learn
-MatplotLib
-Optuna
Όλα τα σενάρια εκτελέστηκαν σε ένα mini-conda περιβάλλον, μετά την εγκατάσταση των παραπάνω πακέτων. Προτείνεται να αξιοποιηθεί 
κάρτα γραφικών προηγμένης τεχνολογίας στο σύστημά σας (όπως η RTX 3090 που χρησιμοποιείται εδώ) για να συντομεύσετε το χρόνο εκτέλεσης αυτών των απαιτητικών σεναρίων.

Δομή
-Τα αρχεία περιέχουν όλους τους κώδικες Python που απαιτούνται για τη διεξαγωγή των πειραμάτων και προσδιορίζονται ως: model_dataset.py
(π.χ. lenet_c100.py) για τον διαχωρισμό και τη διεξαγωγή κάθε πειράματος αποκλειστικά. Τα μοντέλα που χρησιμοποιούνται είναι το LeNet, όπως έχει κατασκευαστεί
από τον διάσημο Yann LeCun, και 2 δικά μου προσαρμοσμένα μοντέλα, που ονομάζονται ModNet1 και ModNet2, τα οποία θεωρούνται βαθύτερα νευρωνικά δίκτυα.
-Υπάρχουν επίσης 3 εναλλακτικά μοντέλα που προτείνονται, όπου το λογισμικό Optuna διεξήγαγε δοκιμές για να καθορίσει ένα καλύτερο
ρυθμός εκμάθησης και φίλτρα για κάθε μοντέλο και τα 3 μοντέλα εκπαιδεύτηκαν ξανά στο σύνολο δεδομένων Cifar-100.

Αποτελέσματα
-Κάθε σενάριο εμφανίζει (χρησιμοποιώντας MatplotLib) γραφήματα των μετρικών Accuracy και Loss που συγκεντρώθηκαν στη διαδικασία εκπαίδευσης, για 5
επαναλήψεις. Στη συνέχεια, αξιολογείται το μοντέλο στα σύνολα δεδομένων εκπαίδευσης και δοκιμής και εκτυπώνει τον μέσο όρο των μετρικών, συμπεριλαμβάνοντας
και τις Precision, Recall και F1-Score (για κάθε αλγόριθμο) μετά από 5 επαναλήψεις.
-Στα σενάρια βελτιστοποίησης, το Optuna πραγματοποιεί 10 δοκιμές σε καθένα από τα 3 μοντέλα για την έρευνα καλύτερων υπερπαραμέτρων, οι οποίες
στη συνέχεια αντικαθίστανται στα αρχικά σενάρια για μια ακόμα εκπαιδευτική διαδικασία.