# Week 4 - Margin Maximization

## Trying to turn ML problems into Optimization Problems

If we can turn ML problems into optimization problems, we can use the wealth of maximization knowledge to solve ML problems. The Perceptron algorithm was originally created not via math, but via cleverness - starting from math allows for actual optimization.

Define $J(\Theta)$ as an objective function, where $\Theta$ is our array of parameters. Note that this is capital Theta - i.e. all parameters of our problem. So for linear separators, it contains our Theta vector, and our $\Theta_0$. Note that we may also write $J(\Theta; D)$ to clarify the dependence on the dataset $D$.

We want to find $\Theta^{*} = \stackrel{argmin}{\Theta} J(\Theta)$, so we want to find the $\Theta$ that minimizes $J(\Theta)$. This is the *objective function*

The typical ML problem looks like $J(\Theta) = (\frac{1}{n} \sum_{i=1}^{n}L(h(x^{(i)}, \Theta), y^{i})) + \lambda R(\Theta)$.

$L$ is our loss function. $h$ is our prediction is $\Theta$ is our parameter for question $x$. $y$ is the actual answer. So this first term is how unhappy we are for our guess - i.e. **training error**.

Obviously, we want to minimize training error. But the goal here is to do well on stuff you haven't seen. This is where the extra term comes in.

$R(\Theta)$ is a regularizer - you tell the function not to fit the data super closely. $\lambda$ is just a(positive) constant that you can adjust to set the degree to which you want to fit your data. These constants are known as **hyperparameters**.

## Regularization

We use regularizers because we want to perform well on input values we have not seen before. If we just wanted the hypothesis to minimize loss on the training data, we wouldn't apply the regularizer!

This regularizer is applied because it allows for generalization - this is based on the assumption that the testing/training data has some underlying regularity. One way to get these assumptions is by limiting the choices of hypothesis classes. You can also provide smoother guidance - i.e., within a hypothesis class, we prefer some hypotheses to others.

The regularizer expresses this preference and the constant $\lambda$ says how much we are willing to trade off loss on the training data vs. preference over hypotheses.

You can see this below:

![complex sol vs. simple](https://openlearninglibrary.mit.edu/assets/courseware/v1/2da9456dd7b8b19e0f47b192b76ea32e/asset-v1:MITx+6.036+1T2019+type@asset+block/images_logistic_regression_regularization_tikzpicture_1-crop.png)

$h_1$ has 0 training loss, but $h_2$ only misclassifies two points and is very simple. We may prefer $h_2$ because we believe it will perform better on data from the same distribution. 

You can also think of regularization as the idea that we'd like to prevent our hypothesis from being too dependent on the particular training data we were given - we don't want the hypothesis to change much if the training data changed slightly.

These regularizers are often expressed as $R(\Theta) = ||\Theta-\Theta_{prior}||^2$ - this is done when we have some idea in advance that $\Theta$ ought to be near some value $\Theta_{prior}$. If you don't have this, you regularize towards 0. You often also use $||\Theta||^2$ This says you want to keep the norm of your parameters small - so it'll try less to assign a lot of probablity to far away points.

## Logistic Regression/Linear Logistic Classifiers

We'll consider a loss function first.

With a perceptron, we have 0-1 loss, i.e. $Loss_{01}(g,a) = 1$ if $g \neq a$ or equals $0$ if $g=a$. 
The training error here is $J(\Theta, \Theta_0) = \frac{1}{n}\sum^{n}_{i=1} L(sign(\Theta^T x^{(i)}+\Theta_0),y^{(i)})$.

Perceptron can minimize this if the data is linearly separable. Otherwise, sike. So instead, let's be fancy and say that we just want to minimize this totally - not just get 0.

The problem here is that it's not computationally efficient to find this. Minimizing the 0-1 Loss is NP Hard according to compsci! This is because two hypotheses can make the same number of mistakes even though one is actually closer to the optimal values and because the predictions are all categorical - the classifier can't express degree of certainty.

So we want to rephrase the problem s.t. we don't change the statement of the problem, but gives us computational leverage.

So, instead of our guess being discrete - 0 or 1 - let's make it continuous. To do this, we'll need a new hypothesis class. These are called ***Logistic Linear Classifiers***.

Our LLC is: $LLC(x;\Theta, \Theta_0) = \sigma(\Theta^{T}x+\Theta_0)$. Sigma here is the sigmoid function, $\sigma(z) = \frac{1}{1+e^{-z}} \in (0,1)$. This goes between 0 and 1. At $z=0$, this is 0.5, when $z>0$, it's greater than 0.5. We'll interpret this as a probability - the closer it is to 1, the more sure we are that x is positive, the closer to 0, the more sure we are x is negative.

This loss function gives us the ability to make marginal improvements! Not just a question of - "this is right, this is wrong".

Now, we formally define this as a hypothesis class.

To make a **classifier** predict $+1$ when $\sigma(\Theta^T + \Theta_0) > 0.5$. For this to be true, our input to sigma needs to be positive. This is somewhat easy to visualize in one dimension for your $x$ values - you treat your $\sigma$ as an extra dimension. The below shows LLCs for $\sigma(10x+1), \sigma(-2x+1), \sigma(2x-3)$

![one dimensional LLC](https://openlearninglibrary.mit.edu/assets/courseware/v1/390096256866f64034d67b9133a2967c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_logistic_regression_a_new_hypothesis_class_linear_logistic_classifiers_tikzpicture_2-crop.png)

Remember that the definition of a classifier is a mapping from $\R^d \rightarrow \{-1,+1\}$ or another discrete set. So this LLC isn't a classifier by itself since its outputs are in $(0,1)$. If we're forced to make a prediction in ${+1,-1}$, we predict $+1$ if $\sigma(\Theta^T + \Theta_0) > 0.5$ and $-1$ otherwise. The value $0.5$ is sometimes called a *prediction threshold*.

We might change this threshold as needed by the problem definition. If the consequences of predicting $+1$ when the answer is $-1$ are worse than the opposite, we can raise the threshold.

OK - let's define our loss function. The loss on all our data is inversely related to the probability that $\Theta,\Theta_0$ assign to my dataset. So we want a $\Theta$, $\Theta_0$ with a high probability.

We define $g^{(i)} = \sigma(\Theta^Tx^{(i)} + \Theta_0)$. Then our probability is the product of all of our guesses, where we use $g^{(i)}$ if $y^{(i)} = 1$. Otherwise, we use $1-g^{(i)}$. We can rewrite this with some exponent math: 

$$
\prod^{n}_{i=1} g^{(i)^{y^{(i)}}}(1-g^{(i)})^{1-y^{(i)}}
$$

Products can be annoying to deal with, and since the log function is monotonic, we can maximize this log quantity and also maximize the above. So this becomes

$$
\sum^{n}_{i=1} (y^{(i)}\log{g^{(i)}}+(1-y^{(i)})log(1-g^{(i)}))
$$

We can turn this into a minimization problem by taking the negative above and aiming to minimize the loss 

$$
\sum^n_{i=1}L_{nll}(g^{(i)}, y^{(i)})
$$

$L_{nll}$ is the *negative log-likelihood* loss function:

$$
L_{nll} (guess,actual) = -(actual*log(guess) + (1-actual)*log(1-guess)). This is sometimes referred to as *log loss* or *cross entropy*.
$$
## ML as optimization - Gradient Descent

### In one dimension

So now we have this loss function. Let's find the Thetas that minimize our objective function. So let's present another optimization algorithm, that's better than our old approach of "guess at random".

**Gradient Descent** is an iterative algorithm that guesses an $\Theta$ to start - not necessarily in a smart way. Then, it looks at the derivative, and moves in the negative direction along your loss function.

For one dimensional gradient descent: $IDGD(\Theta_{init}, f, f', \epsilon, \eta) \\ \Theta^{(0)}=\Theta_{init}; t=0 \\ loop \\ \space \space t=t+1 \\ \space \space \Theta^{(t)} = \Theta^{(t-1)} - \eta f'(\Theta^{(t-1)}) \\
until \space \space |f(\Theta^{(t)}) - f(\Theta^{(t-1)})| < \epsilon \\ return \space \space \Theta^{(t)}$

You could also terminate when $|f'(\Theta^{(t)}| < \epsilon$, or when $|\Theta^{(t)} - \Theta^{(t-1)}| < \epsilon$ - both are reasonable.

Note that $\eta$ is a factor that tells the algo how sensitive it should be. So you stop iterating until the difference between steps is below your threshold.

This doesn't always work if you have a lot of local minima! This method promises that you'll get to at least one local 1 minima. So our theorem is:

If $f$ if convex, for any desired accuracy $\epsilon$, there is some $\eta$ such that $GD$ will converge to $\Theta$ within $\epsilon$ of optimum.

The takeaway here is that a small eta takes too long, but a large eta can miss stuff.

### General Gradient Descent


$GD(\Theta_{init},f, \nabla_{\Theta} f, \epsilon, \eta)$

Here, we'll use a function $f$ like $f(\Theta_1. ..., \Theta_m)$. M is not necessarily the dimension of our data, but just takes $m$ args. We also have $\Theta$, which is of dimension $m$ by $1$. 

Our gradient, $\nabla_{\Theta}f$. This has dimensions $m$ by $1$. It looks like:

$$
[\frac{\partial f}{ \partial \Theta_1}...\frac{\partial f}{\partial \Theta_m}]^T
$$

$\Theta_{init}$ is also $m$ by $1$ and is our starting point.

So, the only thing different from our one-dimensional version is the formula for the next step. 

$$
\Theta^{(t)} = \Theta^{(t-1)} - \eta \nabla_{\Theta}f(\Theta^{(t-1)})
$$

Termination is the same if we use the original case above!!

## Logistic Classification as Optimization

With all these pieces in place, here's our objective function for optimizing regularized negative log-likelihood for a linear logistic classifier. This is also known as "logistic regression". We'll call it $J_{lr}$ and define it as

$$

J_{lr}(\Theta,\Theta_0,D) = (\frac{1}{n}\sum{L_{nll}(\sigma(\Theta^T x^{(i)} + \Theta_0),y^{(i)})} + \lambda||\Theta||^2

$$

## More on Gradient Descent Optimization

This seems like an error in the course - a repeat of info? BUt going thru just in case.

0-1 Loss isn't super efficient, and we want come up with another loss function that's easier for computers to work with. This is where gradient descent comes in.

**Gradient Descent** in one dimension, we have $(f, f', \Theta_{init}, \nu , \eta)$

Given a function $f(\Theta)$, we want a $\Theta^* = \stackrel{argmin}{\Theta} f(\Theta)$

$f'$ is our derivative! And we don't need a computer for this - so for this assume we have f' on hand.

$\eta$ is our step size, and is generally small and positive, indicating a step in the downhill direction. 

$\Theta_{init}$ is our initial Theta. Then,

$\Theta = \Theta_{init} \\loop
\\\space \space \Theta = \Theta - \eta f'(\Theta) \\
until \space \space f'(\Theta) < \epsilon \\
return \space \space \Theta$

See the definition in the prior section for the more general case.

Note that in the above, for multiple dimensions, we used uppercase Theta, $\Theta$ and we wanted to find an optimal $\Theta^* = arg min_{\Theta}J(\Theta)$. Just think of this as defining a surface over $\Theta$ in one or two dimensions and finding the lowest point. The same intuition applies to higher dimensions - you want to find a minima, so you take small steps in the direction that is steepest down.

### A few challenges

If $\eta$ is large, you could step over the optimal Theta and end up in NaN territory.

Another issue you could run into getting stuck in local optima. With gradient descent, you can say the following about finding a local optimum.

**Theorem**: *If $J$ is convex, for any desired accuracy $\epsilon$, there is some step size $\eta$ such that gradient descent will converge to within $\epsilon$ of the optimal $\Theta$.* However, we must be careful when we choose the step size to prevent slow convergence, oscillation aroung the minimum, or divergence.

**This is why picking $\eta$ is important.** Too large and you could end up in infinity territory, too small and you could get stuck somewhere that isn't optimal. Ther

There are some methods that start with a large $\eta$ and progressively make it smaller, but this is an example of why it's worth learning the theory instead of just jumping into TensorFlow - **if it doesn't work, you have to know why it's not working to fix it.**

### In multiple dimensions

We define the general form above. Some extensions on the definition of the gradient:

$\nabla_{\Theta}J = \frac{1}{n}\sum y^{(i)}x^{(i)}L'(y^{(i)}(\Theta^T x^{(i)} + \Theta_0)) + \lambda \Theta$

This gets you a column vector, so you're good! Remember that $y^{(i)}, L'$ are scalars (the loss function returns a scalar so the derivative will too, $y^{(i)}$ is a scalar by definition as that's what the function maps to. The rightmost term is your regularizer, and that's just a column vector as well.

We also need to take the derivative of $J$ w.r.t. $\theta_0$, but I think her definition here was wrong.

More formally (and generally) defining this:

Assume $\Theta \in \R^{m}$, so $J : \R^m \rightarrow \R$. The gradient of $J$ with respect to $\Theta$ is: 

$$
\nabla _\Theta J = \begin{bmatrix}  \partial J / \partial \Theta _1 \\ \vdots \\ \partial J / \partial \Theta _ m \end{bmatrix}
$$

You can find the termination criterion in the section above.

### Quick overview of the parameters in Gradient Descent

There are three kinds of parameters that we worry about.

1. Thetas - $\Theta, \Theta_0$ -  are parameters of the answer - these are the answer we give out. This describes the hypothesis we give out to the world.
2. Lambda - $\lambda$ - is a hyperparameter. It describes the trade-off we're making. This affect whether the problem we're solving gives us good performance on data we haven't seen.
3. Eta - $\eta$ - a parameter of the optimization algorithm. We don't use cross validation for this. This affects the ability to optimize on the set we have. Step size!


## Application of Gradient Descent to the Logistic Regression Objective Function.