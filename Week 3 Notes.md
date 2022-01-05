# Week 3 - Feature Representation

## Polynomial Basis

She starts out with the explanation of the offsets from last class. This was a transform that allowed you achange a separator that did not go through the origing into one that did, in $d+1$ dimensions. She defines a transform from $R^d$ to $R^D$. It assumes that you have a linear separator in the lower dimensional space.

They provide an interesting example of this. Consider this: ![xor dataset](https://openlearninglibrary.mit.edu/assets/courseware/v1/7113a168fd1cb279b0a1548c7e16c08c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_1-crop.png).

There's no linear separator for this in two dimensions. So let's take a low dimensional dataset and transform into a higher-dimensional one, and look for a separator there. **Note that a separator in 1-D is just a point!**:

![1-d ex](https://openlearninglibrary.mit.edu/assets/courseware/v1/083f20fe82de4d0b45c86e695e4d142c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_2-crop.png)

Clearly, this has no separator. But let's try a transformation $\phi(x) = [x,x^2]$. This gives us: ![in the phi space](https://openlearninglibrary.mit.edu/assets/courseware/v1/bbc807b76c87fd03a606af741140257c/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_3-crop.png)
This clearly has a ton of separators! Note what happens here - the y value is the original x squared, so of course the +s would be positive and *higher* as they have larger magnitude.

But a linear separator in $\phi$ space is nonlinear in the original space. Here's a good example. Let's say we have a separator $x^2-1=0$ (note that this is what is depicted above - x^2 is the y axis, and anything above 1 is labeled as positive). This labels anything $x^2-1>0$ as positive. What does this correspond to in 1-D space? We need to ask "what x values have the property s.t. $x^2-1=0$?" Clearly, $+1,-1$. This defines our 1-D separator: ![1-d version of sep](https://openlearninglibrary.mit.edu/assets/courseware/v1/2df7785d97c451a36a9ff9b7683f1e2a/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_top_tikzpicture_4-crop.png)

This is the basis of *kernel methods*, and is the motivation for multi-layer neural nets. Moving up in dimensions can make your set separable.

Obv, there are many ways to construct $\phi$. Some are systemic, domain independent (*polynomial basis*), others depend on the semantics (meaning) of the original features, so we design with the domain in mind.

One of these approaches is using a *polynomial basis*. You use a *k*-th order basis (obv k is pos) - you then have a feature of every possible product of *k* different dimensions in your original input. Let's try a different approach to our *xor* problem with $k=2$: our $\phi((x_1x_2)) = (1,x_1,x_2,x_1^2,x_1x_2,x_2^2)$. After 4 iterations, the perceptron finds $\theta = (0,0,0,0,4,0)$ with $\theta_0 = 0$. Our polynomial is:

$$
0+0x_1+0x_2+0x_1^2+4x_1x_2+0x_2^2+0 = 0
$$

The classifier does the following, with the gray region being negative, and the white being positive. ![k=2, iteration 4](https://openlearninglibrary.mit.edu/assets/courseware/v1/4288137b7bd7f6b7a431b5b6c9f90b85/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_1.png).

Why does this work? The polynomial you're working with is $4x_1x_2 = 0$. Think of the case $x_2 > 0, x_1 > 0$. This is the first quadrant, just think of $x_2$ as $y$!

Note that this is not a linear separator, tho. In a slightly different setup, we have this setup: $\theta = (1,-1,-1,-5,11,-5),\theta_0 = 1$, and it makes:

![after 65 iterations...](https://openlearninglibrary.mit.edu/assets/courseware/v1/4b61c604452f9f0d62c39ec28345ce8e/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_2.png). 

Why does this take so many iterations? Remember $(R/\gamma)^2$ from the convergence theorem. The radius of the circle it would fit in is *further from 0$ so R is larger. I think your gamma - the measure of how "separable" the data is - gets smaller too since you're in a higher dimensional space? But R grows faster than $\gamma$.

And here's a harder dataset. They are after 200 iterations, with bases of order 2, 3, 4, and 5.

![order 2](https://openlearninglibrary.mit.edu/assets/courseware/v1/5ecc31b1e67647cd012007c78c303be7/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_3.png)
![order 3](https://openlearninglibrary.mit.edu/assets/courseware/v1/2fa58ddc29ad247b6937521bd7d0cce1/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_4.png)
![order 4](https://openlearninglibrary.mit.edu/assets/courseware/v1/722b0abd100d03c738617979ddce8a62/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_5.png)
![order 5](https://openlearninglibrary.mit.edu/assets/courseware/v1/ee0ef00782fd3da6f009fee4baf10dd5/asset-v1:MITx+6.036+1T2019+type@asset+block/images_feature_representation_6.png)

## Features

### Discrete

Discrete features are important because you want the learner to find underlying regularities. We'll assume that these input vectors x are in $\R^d$, so we want to convert these discrete vectors into vectors of real numbers.

Some common strategies:

* **Numeric** - assign these values a number, like $1.0/k, 2.0/k,...1.0$. We could also do further processing. This is only sensible when these numbers signify some sort of numeric quantity and value.
* **Thermometer code** - useful if your discrete values have a natural ordering (well-ordered), but not a natural mapping into real numbers. The strategy here is to use a vector of length $k$ binary variables, s.t. discrete input value $0 < j \leq k$ into a vector s.t. the first j values are $1.0$, and the rest are $0.0$. Does not mean anything for spacing or numerical quantities, but does convey ordering.
* **Factored code** - if your discretes can be broken down into two parts, like "make" and "model", then it's best to treat those separately. 
* **One-hot code** - if there's no obvious numeric, ordering, or factorial structure, then the best strategy is to use a vector of length $k$, where we convert discrete input value $0 < j \leq k$ into a vector in which all values are $0.0$, except for the $jth$, which is $1.0$.
* **Binary code** - representing as a binary number, which lets us represent k values using a vector of length $\log{k}$. Not great though - your system then has to learn the decoding algorithm too.

As an example, how would we encode blood types? The set of blood types is ${A+,A-,B+,B-,AB+,AB-,O+,O-}$. No linear relationship, ordering here. We do have a factoring: ${A, B, AB, O}$ and ${+,-}$. We could even make the first group ${A, notA}$, ${B, notB}$. So, some possible encodings:

* Use a 6-D vector, with two dimensions to encode each factor with a one-hot encoding. $AB+$ here would be $(-1.0,-1.0,1.0,-1.0,1.0,-1.0)$. $A+$ would be $(1.0,-1.0,-1.0,-1.0,1.0,-1.0)$. Could replace $1.0$ with $0.0$
* Use a 3-D vector, with one dimension for each factor, encoding its presence as $1.0$ and absence as $-1.0$ (sometimes better than 0.0). In this case, $AB+$ would be $(1.0,1.0,1.0)$ and $O-$ would be $(-1.0,-1.0,-1.0)$. A+ would be $(1.0,-1.0,1.0)$.

### Text

Text is harder, as you might guess. We cover sequential input models later on - in these, you feed a hypothesis in word-by-word, or even character-by-character! Some simpler encodings include *bag-of-word* (bow) models. Here, you let $d$ be the number of words in our vocab (computed from training set or some other body of text/dictionary). Then, you make a binary vector (with values of $1.0$ and $0.0$) of length $d$, where element $j$ has value $1.0$ if word $j$ occurs in the document, and $0.0$ otherwise.

### Numeric values

If a feature is already numeric, you should probably leave it as such. However, if there are natural breakpoints, like in encoding age, where being over 18/21 in the US marks adulthood, you can encode differently depending on what your goal is. In the prior case, you could bucket values into bins. You could also use a one-hot encoding for medical situations where we don't expect a linear/monotonic relationship between age and physiological features.

If you leave it as numeric, it's useful to scale it so that would put it in the range $[-1.0, +1.0]$. This is useful so the algo doesn't have to work to put parameters that are very different from one another on an equal basis. To standardize, we could apply the transformation $\phi(x) = \frac{x-\bar{x}}{\sigma}$. This normalizes your data so we have mean $0$ and stddev $1$. Note that this isn't great when distance/units matter!

You could also apply a polynomial basis to transform one or more groups of numeric features.