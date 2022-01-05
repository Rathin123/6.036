# Numpy Reference

**np.argmax(array)** - returns the indices of the maximum values along an axis

**np.array_split(array, number of sub-arrays, axis)** - splits an array into multiple sub-arrays. Takes an axis argument.

**np.concatenate(list-like of arrays, axis)** - concatenates sequence of arrays along an existing axis

**np.zeros(shape)** - returns array of zeros of shape (x,y) or length n

**np.ndarray.shape** - method that returns shape of array. If you want to reshape a one-d array into a column, you can do ar.reshape(-1,1) (see next bullet too)

**the keepdims parameter** - useful for when you apply an aggregator like mean and want to maintain the shape of the original array without using reshape. This is probably best practice - also lets you use .T and get a col vector from it.

**np.array.dot/@** - matrix multiplication. Seems like @ is best practice

**np.argmin(values)/np.argmax(values)** - find the index of the min, max of an iterable

To get a single column out of a numpy array, do: array[:, i:i+1], don't try to index it. Just saves you some headache.

**Can access a single element with array[row,col], don't have to do array[row][col]**

