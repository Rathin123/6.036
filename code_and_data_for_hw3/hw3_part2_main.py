import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('/Volumes/GoogleDrive/My Drive/MOOCs and Books/6.036/code_and_data_for_hw3/auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

# def auto_xval( T, feature_set):
#     #returns list of accuracies. The first for perceptron, the second for averaged
#     #perceptron

#     auto_data_features, auto_labels_features = hw3.auto_data_and_labels(auto_data_all, feature_set)

#     return [hw3.xval_learning_alg(hw3.perceptron, auto_data_features, auto_labels_features, 10, T), hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data_features, auto_labels_features, 10, T)]

# feature_set_1 = [('cylinders', hw3.raw),
#             ('displacement', hw3.raw),
#             ('horsepower', hw3.raw),
#             ('weight', hw3.raw),
#             ('acceleration', hw3.raw),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.raw)]

# feature_set_2 = [('cylinders', hw3.one_hot),
#             ('displacement', hw3.standard),
#             ('horsepower', hw3.standard),
#             ('weight', hw3.standard),
#             ('acceleration', hw3.standard),
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             ('origin', hw3.one_hot)]

# #4.1c i
# print("4.1c i")
# print(auto_xval(1, feature_set_1))

# #4.1c ii
# print("4.1c ii")
# print(auto_xval(1,feature_set_2))

# #4.1c iii
# print("4.1c iii")
# print(auto_xval(10,feature_set_1))

# #4.1c iv
# print("4.1c iv")
# print(auto_xval(10,feature_set_2))

# #4.1c v
# print("4.1c v")
# print(auto_xval(50,feature_set_1))

# #4.1c vi
# print("4.1c vi")
# print(auto_xval(50,feature_set_2))

# #4.2a 
# print("Parameters of best classifier")
# print(hw3.averaged_perceptron(auto_data, auto_labels, params = {'T':1}))


# #4.2b
# feature_set_limited = [#('cylinders', hw3.one_hot),
#             #('displacement', hw3.standard),
#             ('horsepower', hw3.standard),
#             ('weight', hw3.standard)]
#             # ('acceleration', hw3.standard)
#             ## Drop model_year by default
#             ## ('model_year', hw3.raw),
#             #('origin', hw3.one_hot)]

# print("4.2b")
# print(auto_xval(50,feature_set_limited))


#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# # Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# # The train data has 10,000 examples
# review_data = hw3.load_review_data('reviews.tsv')

# # Lists texts of reviews and list of labels (1 or -1)
# review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# # The dictionary of all the words for "bag of words"
# dictionary = hw3.bag_of_words(review_texts)

# # The standard data arrays for the bag of words
# review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
# review_labels = hw3.rv(review_label_list)
# print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

def review_xval(T):
    
    return [hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10, T), hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, T)]
    
T_vals = [1,10,50]

#5.1 - uncomment since it takes a while to run
# print("5.1")
# print([review_xval(T_val) for T_val in T_vals])

#5.2

# best_classifier = hw3.averaged_perceptron(review_bow_data, review_labels, params = {'T':10})

# reversed_dict = hw3.reverse_dict(dictionary)

# fixed_dict = {reversed_dict[index] : best_classifier[0][index][0] for index in range(len(best_classifier[0]))}

# #best 10 words
# print(sorted(fixed_dict, key=fixed_dict.get, reverse = True)[:10])

# #worst 10 words
# print(sorted(fixed_dict, key=fixed_dict.get)[:10])

# #best and worst review
# classifier_scores = best_classifier[0].T@review_bow_data+best_classifier[1]

# print("best review")
# print(review_data[np.argmax(classifier_scores)])

# print("worst review")
# print(review_data[np.argmin(classifier_scores)])


#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[2]["images"]
d1 = mnist_data_all[4]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    
    to_return = []

    for index in range(x.shape[0]):
      to_return.append(x[index,:,:].reshape(x.shape[1]*x.shape[2]))

    return np.array(to_return).T


def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """

    to_return = []

    for index in range(x.shape[0]):
      to_return.append(np.mean(x[index,:,:], axis = 1).reshape(-1,1))

    return np.array(to_return).T[0]

print(row_average_features(data).shape)

def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    to_return = []

    for index in range(x.shape[0]):
      to_return.append(np.mean(x[index,:,:], axis = 0).reshape(-1,1))

    return np.array(to_return).T[0]

print(col_average_features(data).shape)

def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """

    mid = int(np.floor(x.shape[1]/2))

    to_return = []
    
    for index in range(x.shape[0]):

        data = x[index,:,:]

        top = data[:mid,:]
        bot = data[mid:,:]

        top_avg = np.mean(top)
        bot_avg = np.mean(bot)

        to_return.append([top_avg, bot_avg])
        
    return np.array(to_return).T

#fixed = top_bottom_features(data).reshape(2,160)

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

print(acc)

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

def acc_versus(int1, int2, feature = 'raw'):

    d0 = mnist_data_all[int1]["images"]
    d1 = mnist_data_all[int2]["images"]
    y0 = np.repeat(-1, len(d0)).reshape(1,-1)
    y1 = np.repeat(1, len(d1)).reshape(1,-1)

    # data goes into the feature computation functions
    data = np.vstack((d0, d1))
    # labels can directly go into the perceptron algorithm
    labels = np.vstack((y0.T, y1.T)).T

    if feature == 'raw':
        acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
    elif feature == 'row':
        acc = hw3.get_classification_accuracy(row_average_features(data), labels)
    elif feature == 'col':
        acc = hw3.get_classification_accuracy(col_average_features(data), labels)
    elif feature == 'top/bot':
        acc = hw3.get_classification_accuracy(top_bottom_features(data), labels)

    return acc

#general accuracies
print([acc_versus(0,1), acc_versus(2,4), acc_versus(6,8), acc_versus(9,0)])

#0 v 1, all encodings
print("0 v 1, all encodings")
print([acc_versus(0,1, feature = 'row'), acc_versus(0,1, feature = 'col'), acc_versus(0,1, feature = 'top/bot')])

#2 v 4, all encodings
print("2 v 4, all encodings")
print([acc_versus(2,4, feature = 'row'), acc_versus(2,4, feature = 'col'), acc_versus(2,4, feature = 'top/bot')])

#6 v 8, all encodings
print("6 v 8, all encodings")
print([acc_versus(6,8, feature = 'row'), acc_versus(6,8, feature = 'col'), acc_versus(6,8, feature = 'top/bot')])

#0 v 9, all encodings
print("0 v 9, all encodings")
print([acc_versus(0,9, feature = 'row'), acc_versus(0,9, feature = 'col'), acc_versus(0,9, feature = 'top/bot')])