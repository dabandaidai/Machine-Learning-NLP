"""
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
"""

import data
import numpy as np


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    means = np.zeros((10, 64))
    for i in range(means.shape[0]):
        # Find subset of digits belongs to class i
        class_i = data.get_digits_by_label(train_data, train_labels, i)
        # The mean estimates
        means[i, ] = np.mean(class_i, axis=0)
    return means


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three-dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    # Get the mean estimates
    means = compute_mean_mles(train_data, train_labels)
    # For each digit class
    for i in range(covariances.shape[0]):
        # Find subset of digits belongs to class i
        class_i = data.get_digits_by_label(train_data, train_labels, i)
        N_i = class_i.shape[0]
        # Get x-mu_k
        x_mu = class_i - means[i]
        x_muT = np.transpose(x_mu)
        # Compute by formula
        square = np.matmul(x_muT, x_mu)
        covariances[i] = square / N_i
    return covariances


def compute_sigma_mles2(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three-dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    for i in range(covariances.shape[0]):
        class_i = data.get_digits_by_label(train_data, train_labels, i)
        N_i = class_i.shape[0]
        x_mu = class_i - means[i]
        x_muT = np.transpose(x_mu)
        square = np.matmul(x_muT, x_mu)
        temp = square / N_i
        # Same code above, but this time only take the diagonals
        covariances[i] = np.diag(np.diag(temp))
    return covariances


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    """
    # Number of images
    N = digits.shape[0]
    # Number of classes
    M = covariances.shape[0]
    # Number of pixels
    D = covariances.shape[1]
    # Result
    result = np.zeros((N, M))
    # now compute log p(x|y,mu,sigma)
    for i in range(M):
        # added identity to increase stability
        sigma = covariances[i] + 0.01 * np.identity(D)
        # Find mu
        mu = means[i]
        for j in range(N):
            # Compute by given formula
            first_part = -D / 2 * np.log(2 * np.pi)
            second_part = -0.5 * np.log(np.linalg.det(sigma))
            X_mu = digits[j] - mu
            X_muT = np.transpose(X_mu)
            sigma_inv = np.linalg.inv(sigma)
            last_part = -0.5 * (np.matmul(np.matmul(X_muT, sigma_inv), X_mu))
            result[j][i] = first_part + second_part + last_part
    return result


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    """
    # Similarly, get the dimensions
    N = digits.shape[0]
    M = covariances.shape[0]
    result = np.zeros((N, M))
    # Get the generative likelihood
    LLH = generative_likelihood(digits, means, covariances)
    for j in range(N):
        # Notice 2 ln(0.1) terms cancel out
        log_xy = LLH[j]
        log_x = LLH[j][0]
        # Add the exponential together
        for x in range(M - 1):
            log_x = np.logaddexp(log_x, LLH[j][x + 1])
        log_x = np.full((M,), log_x)
        result[j] = log_xy - log_x
    return result


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    """
    # First get conditional
    conditional = conditional_likelihood(digits, means, covariances)
    N = digits.shape[0]
    collect = 0
    for i in range(N):
        # Label for this image
        j = int(labels[i].item())
        # Get the conditional for this label of the image
        collect += conditional[i, j]
    result = collect / N
    return result


def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    # Get the conditional likelihood
    conditional = conditional_likelihood(digits, means, covariances)
    # Find the most likely posterior class by argmax
    return np.argmax(conditional, axis=1)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    covariances2 = compute_sigma_mles2(train_data, train_labels)

    # Evaluation
    train_llh = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_llh = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    train_accuracy = np.sum(classify_data(train_data, means, covariances) == train_labels) / train_labels.shape[0]
    test_accuracy = np.sum(classify_data(test_data, means, covariances) == test_labels) / test_labels.shape[0]

    # Evaluation2
    train_llh2 = avg_conditional_likelihood(train_data, train_labels, means, covariances2)
    test_llh2 = avg_conditional_likelihood(test_data, test_labels, means, covariances2)
    train_accuracy2 = np.sum(classify_data(train_data, means, covariances2) == train_labels) / train_labels.shape[0]
    test_accuracy2 = np.sum(classify_data(test_data, means, covariances2) == test_labels) / test_labels.shape[0]

    print("The average conditional log-likelihood on train set is {train_llh}.".format(train_llh=train_llh))
    print("The average conditional log-likelihood on test set is {test_llh}.".format(test_llh=test_llh))
    print("The accuracy on train set is {train_accuracy}.".format(train_accuracy=train_accuracy))
    print("The accuracy on test set is {test_accuracy}.".format(test_accuracy=test_accuracy))
    print("The average conditional log-likelihood on train set is {train_llh}.".format(train_llh=train_llh2))
    print("The average conditional log-likelihood on test set is {test_llh}.".format(test_llh=test_llh2))
    print("The accuracy on train set is {train_accuracy}.".format(train_accuracy=train_accuracy2))
    print("The accuracy on test set is {test_accuracy}.".format(test_accuracy=test_accuracy2))


if __name__ == '__main__':
    main()
