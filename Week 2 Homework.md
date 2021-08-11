# Week 2 Homework Notes

## Exercises

### Problem 2
Consider vectors $\theta = (1, -1, 2, -3)$ and  $\theta' = (-1, 1, -2, 3)$. Are they equivalent hyperplanes?

You said no. Remember that hyperplane H defined by  norm vector $\theta$ is the set of points $x$ s.t. $x \cdot 0 = 0$, or the set of points $x$ perpendicular to $\theta$. Obv $\theta \cdot x = 0 = -\theta \cdot x$

## Homework

### Problem 5

A pretty interesting transformation that allows you to reduce linear classifiers with offsets into ones that don't:

Add a dimension with the same value for each point to your data. This is because the dot product in this higher dimension is equivalent to a dot product in the lower dimensional space with the **addition of a bias term**. Most other texts assume a 1 is apprended to each point instead of explicitly writing the bias term. Thus, the final element of our theta vector = theta_0.