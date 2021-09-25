# Week 2 Homework Notes

## Exercises

### Problem 2
Consider vectors $\theta = (1, -1, 2, -3)$ and  $\theta' = (-1, 1, -2, 3)$. Are they equivalent hyperplanes?

You said no. Remember that hyperplane H defined by  norm vector $\theta$ is the set of points $x$ s.t. $x \cdot 0 = 0$, or the set of points $x$ perpendicular to $\theta$. Obv $\theta \cdot x = 0 = -\theta \cdot x$

## Homework

### Problem 5

A pretty interesting transformation that allows you to reduce linear classifiers with offsets into ones that don't:

Add a dimension with the same value for each point to your data. This is because the dot product in this higher dimension is equivalent to a dot product in the lower dimensional space with the **addition of a bias term**. Most other texts assume a 1 is apprended to each point instead of explicitly writing the bias term. Thus, the final element of our theta vector = theta_0.

## Some Numpy Functions to Know

**np.argmax(array)** - returns the indices of the maximum values along an axis

**np.array_split(array, number of sub-arrays, axis)** - splits an array into multiple sub-arrays. Takes an axis argument.

**np.concatenate(list-like of arrays, axis)** - concatenates sequence of arrays along an existing axis

**np.zeros(shape)** - returns array of zeros of shape (x,y) or length n

**np.ndarray.shape** - method that returns shape of array

**np.array.dot/@** - matrix multiplication. Seems like @ is best practice

To get a single column out of a numpy array, do: array[:, i:i+1], don't try to index it. Just saves you some headache.

## Cross Validation

Cross validation is a strategy for evaluating a learning algorithm that uses a single training set of size $n$. Cross-validation takes in a learning algorithm $L$, a fixed data set $D$, and parameter k. It will run the algorithm $k$ different times, then evaluate the accuracy of the resulting classifier, and finally return the average accuracies over each of the $k$ runs of $L$. **Psuedocode**

    divide D into k parts, as equally as possible;  call them D_i for i == 0 .. k-1
    #be sure the data is shuffled in case someone put all the positive examples first in the data!

    for j from 0 to k-1:
        D_minus_j = union of all the datasets D_i, except for D_j
        h_j = L(D_minus_j)
        score_j = accuracy of h_j measured on D_j
    return average(score0, ..., score(k-1))

Each time, we train on $k-1$ of the pieces of the data set and test the resulting hypothesis on the piece that was not used for training. When $k=n$, this is called *leave-one-out-cross validation*. Note that the loop in python is just:

    for i in range(k)

since range excludes k by default.