# CS590MLG-project


### Comments on proposal 
Feedback for our paper:


Overall, this is a fantastically novel idea. I am sure that it could be very rewarding while also challenging. Here are a few advices:

1. I am sure that this work includes a lot of trial and error before you can get a reasonable solution. Perhaps, you need good computational resources. Please be prepared. If you cannot find other resources, I suggest you start doing the experiments asap.

2. As I can remember, NETTACK could be very slow, especially for the Evasion attack. For this, I suggest you start with some random attacks (you proposed the mechanism to attack), which can give you plenty of training data. If using NETTACK, perhaps first focus on poisoning attack, which is faster.

3. In practice, the number of nodes that are attacked is typically small (say 5%). Moreover, the number of labeled attacked nodes could even be smaller (say 0.5%). Therefore, a more practical model to detect attack may be unsupervised or semi-supervised. Keep this in mind. You may want to check some unsupervised technique, say autoencoder, matrix factorization, network embedding, ....

4. This task could be rather challenging. I suggest that before training models in a very rough way, I suggest you think about doing some light mathematical analysis, like those in NetTack or RobustGCN (you cited). The analysis will significantly help you propose an elegant model rather than some complicated models that entangle different building blocks. Moreover, in general, defense & attack works like a game. A defending model cannot solve handle every type of attack. So it is reasonable to make certain-level assumption of the attack, which can also decrease your model complexity.

5. Check more references:

Bayesian Graph Neural Networks with Adaptive Connection Sampling, icml 2020

GNNGuard: Defending Graph Neural Networks against Adversarial Attacks, neurips 2020
