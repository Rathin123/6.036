# Week 2 - Perceptrons

## Some background on perceptrons

* This falls under "clever human/algorithm"
* The history of this is interesting - the dude came up with the algo first, then people tried to understand why it worked. 
    * Usually, people have an optimization problem and then try to come up with an algo
    * Proposed by a 1943 model of neurons by McCulloch, Pitts, Hebb. Developed by Rosenblatt in the 60s
* There's another theorem that says that if your data can be perfectly classified by a linear classifier, perceptrons will do it


## Algorithm

We have a training dataset $D_n$ with $x \in \mathbb{R}$ and $y \in {-1, +1}$. The Perceptron algo trains a binary classifier $h(x;\theta,\theta_0)$ using the following to find $\theta$ and $\theta_0$ using $\tau$ iterative steps:

$\theta = [0\space0\space0 \dots 0]^T \\ \theta_0 = 0 \\ \textbf{for} \space t=1 \space \textbf{to} \space \tau \\ \hspace{3mm} \textbf{for} \space i=1 \space \textbf{to} \space n \\ \hspace{6mm} \textbf{if} \space y^{(i)}(\theta^Tx^{(i)}+\theta_0)\leq0 \\ \hspace{9mm} \theta = \theta + y^{(i)}x^{(i)} \\ \hspace{9mm} \theta_0 = \theta_0 + y^{(i)}$

Remember that y is your vector of labels. So if $y^{(i)}$ is positive, that means you want $\theta^Tx^{(i)} + \theta_0$ to be positive as well - your algo prediction matches the actual. You ONLY get to the last block if there is a mismatch. If it never gets there, your algo is perfect, so you have no training error $\mathcal{E}_n$.

If there is a mismatch, it moves $\theta, \theta_0$ so it's closer to the proper classification 

If there is a linear classifier that works with 0 training error for the dataset, this will find it.