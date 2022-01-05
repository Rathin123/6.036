# Week 3 Lab and Homework Notes

* tsv - tab separated value file

Data in the auto-mpg.tsv:

  1. mpg:           continuous (modified by us to be -1 for mpg < 23, 1 for others)
  2. cylinders:     multi-valued discrete
  3. displacement:  continuous
     1. This is the combined swept volume of the pistons inside the cylinders of an engine. Calculated from the bore (diameter of the cylinders), strok (distance the piston travels), and number of cylinders. Engine displacement = π/4 * bore² * stroke * number of cylinders.
  4. horsepower:    continuous
  5. weight:        continuous
  6. acceleration:  continuous
  7. model year:    multi-valued discrete
  8. origin:        multi-valued discrete
  9.  car name:      string (many values)

Feature selection/coding matter a lot for the perceptron due to the perceptron Convergence Theorem. If you can adjust a feature by a simple scalar - do so. It'll lower the number of needed iterations by a ton.

## Homework

Note that our perceptron classifer h classifies as $+1$ iff h(x) > 0.

One-hot encoding leads to a dimension for reach data point. If we assign different weights to each dimension, we can always find a separator for the data. This data in particular is a system of linear equations each with its own unique variable, so there is always a solution.

data = ([[1, 1, 2, 2],
         [1, 2, 1, 2]])
labels = [[-1, 1, 1, -1]]

This is the XOR/exclusive-or problem is not linearly separable.

The question you got wrong was becacuse you only used 100 trials for your perceptron - should have increased the threshold.

For the algorithms (perceptron, avg perceptron) we know so far, we have a few parameters:

* The number of iterations, T
* Which features to use

We use cross-validation to determine the best combinations of these. Don't be afraid to drop some parameters.

### 4.2b 
I think the answer is `cylinders, weight`.

Averaged perceptron seems to be better, interestingly. I think this is because it doesn't incorporate the last "mistake" as much as normal perceptron - each mistake is averaged out.

Usually, for image classification, we use an $m$ by $n$ array, not a $(mn,1)$ vector.

Your implementation to modify $T$ was hacky - didn't work well with the last question, so if you rerun and are confused, that's why.

Also had to add paths manually.

For some reason, from 3dims to 2dims does not preserve the same order as looping + flattening. I think this is how the algo works. Instead, do it this way to get $(a,b,c)$

