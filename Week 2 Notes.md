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

## An Aside on Validation

We just trained our algo, got an answer $\theta \theta_0$. Note that our training error $\mathcal{E}_n$ is too low an estimate. If we memorize our training error, we get training error 0 - but that isn't our goal. We want to deal with new examples.

So we want to set aside some data as **validation data**. Randomize our original data and pick a subset to use as a validation set. Use the performance on this set to determine how good it is.

Now, what if we have multiple algorithms? You could run all algos on training data and pick the one that does the best on validation data. **BUT** after you've used the validation data, the data isn't impartial any more. So, if you want to generate a new hypothesis, ***you now need more data***.

## Offsets

Can be easier to implement classifiers of form

$$
h(x; \theta ) = \begin{cases}  +1 &  \text {if } \theta ^ Tx > 0 \\ -1 &  \text {otherwise.} \end{cases}
$$

without a specific separator $\theta_0$. Obv, this must go through the origin. But we can convert any porblem involving a linear separator *with* offset into one ***without*** an offset (but in higher dimension).

The way this works:

Consider the $d$-dimensional linear separator defined by $
\theta = \begin{bmatrix}  \theta _1 &  \theta _2 &  \cdots &  \theta _ d \end{bmatrix}
$ and offset $\theta_0$.

* to each data point $x \in D$ **append** (NOT ADD) a coordinate with value +1, yielding $x_{\rm new} = \begin{bmatrix}  x_1 &  \cdots &  x_ d &  +1 \end{bmatrix}^ T$
* define $\theta _{\rm new} = \begin{bmatrix}  \theta _1 &  \cdots &  \theta _ d &  \theta _0 \end{bmatrix}^ T$
* So: $\displaystyle = \theta _1x_1 + \cdots + \theta _ dx_ d + \theta _0 \cdot 1$

So $\theta_{new}$ is an equivalent $(d+1)$ dimensional separator to our original, but with no offset.

You can also create simplified perceptron algo if we restrict ourselves to separators through the origin: 

$\theta = [0\space0\space0 \dots 0]^T \\ \textbf{for} \space t=1 \space \textbf{to} \space \tau \\ \hspace{3mm} \textbf{for} \space i=1 \space \textbf{to} \space n \\ \hspace{6mm} \textbf{if} \space y^{(i)}(\theta^Tx^{(i)})\leq0 \\ \hspace{9mm} \theta = \theta + y^{(i)}x^{(i)} \\ \hspace{9mm}$

## Proof of the perceptron algorithm

For this, assume that we're constrained to separators through the origin.

**Linear Separability (through the origin)**: A data set D is linearly separable if there is some $\theta$ s.t. $y^{(i)}(\theta^Tx^{(i)})>0$ for all i

**Margin of a labeled data point** (x,y) w.r.t. a separator $\theta$ = 
$$
y*\frac{\theta^T x}{||\theta||}
$$
I.e. the *signed distance* between the hyperplane and the point $x$ multiplied by the desired classification $y$. So, this is a measure of ***how right we are*** for that data point.

**Margin of a data set D w.r.t. $\theta$**

$$
min_i \space \space y^{(i)}\frac{\theta^T x}{||\theta||}$$

You pick the smallest of the *data point* margins for this!!
Obivously, this is positive if $\theta$ correctly classifies everything.