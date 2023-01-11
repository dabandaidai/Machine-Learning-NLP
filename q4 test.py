"""
Question 2 Skeleton Code
Here you should implement and evaluate the Conditional Gaussian classifier.
"""

import data
from scipy.special import logsumexp
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class
    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    means = np.zeros((10, 64))
    for i in range(10):
        subclass = data.get_digits_by_label(train_data, train_labels, i)
        means[i] = np.mean(subclass, axis=0)
    return means


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class
    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for i in range(10):
        subclass = data.get_digits_by_label(train_data, train_labels, i)
        diff = subclass - means[i]
        covariances[i] = np.dot(diff.T, diff)/subclass.shape[0]
    return covariances


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)
    Should return an n x 10 numpy array
    """
    n = digits.shape[0]
    likelihood = np.zeros((n, 10))
    # now compute log p(x|y,mu,sigma)
    for i in range(10):
        sigma = covariances[i] + 0.01*np.identity(64)
        mu = means[i]
        for j in range(n):
            diff = digits[j] - mu
            exp = -1 / 2 * (diff.T.dot(np.linalg.inv(sigma)).dot(diff))
            likelihood[j][i] = -64 / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(np.linalg.det(sigma)) + exp
    return likelihood


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:
        log p(y|x, mu, Sigma)
    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    """
    N = digits.shape[0]
    M = covariances.shape[0]
    result = np.zeros((N, M))
    # Get the generative likelihood
    LLH = generative_likelihood(digits, means, covariances)
    for j in range(N):
        # Notice 2 ln(0.1) terms cancel out
        log_xy = LLH[j]
        log_x = LLH[j][0]
        for x in range(M - 1):
            log_x = np.logaddexp(log_x, LLH[j][x+1])
        log_x = np.full((M,), log_x)
        result[j] = log_xy - log_x
    return result


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels
        AVG( log p(y_i|x_i, mu, Sigma) )
    i.e. the average log likelihood that the model assigns to the correct class label
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    total_probability = 0
    n = digits.shape[0]
    for j in range(digits.shape[0]):
        label = int(labels[j])
        total_probability += cond_likelihood[j, label]
    avg = total_probability/n
    return avg


def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    # find the average conditional log likelihood on train and test datasets
    train_avg_log = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_log = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    # find the accuracy on train and test datasets
    train_acc = np.mean(classify_data(train_data, means, covariances) == train_labels)
    test_acc = np.mean(classify_data(test_data, means, covariances) == test_labels)

    print(f"The average conditional log-likelihood on train set is {train_avg_log}.")
    print(f"The average conditional log-likelihood on test set is {test_avg_log}.")
    print(f"The accuracy on train set is {train_acc}.")
    print(f"The accuracy on test set is {test_acc}.")

    # plot the leading eigenvectors for each class covariance matrix side by side as 8x8 images
    l_eigenvectors = []
    for i in range(10):
        eigenvalues, eigenvectors = np.linalg.eig(covariances[i])
        l_eigenvectors.append(eigenvectors[:, np.argmax(eigenvalues)].reshape(8, 8))
    image = np.concatenate(l_eigenvectors, axis=1)
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()