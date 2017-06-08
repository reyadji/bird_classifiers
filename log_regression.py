import os
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pickle
import sklearn
from sklearn import linear_model
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import random
import re


# Initialize classes

classes_file='classes.txt'
classes = {}
with open(classes_file, 'rb') as file:
    for line in file:
        line = line.decode('utf-8')
        classes[re.split('\W+', line)[1]] = re.split('\W+', line)[2]

# SHARED FUNCTION

def load_data(
        features_file='train_features.p',
        labels_file='train_labels.p'):
    """Loading data from train and label pickle
    files to features, labels, and unique labels numpy array
    Keyword arguments:
    features_file
    labels_file
    """
    with open(features_file, 'rb') as file:
        features = pickle.load(file)
    with open(labels_file, 'rb') as file:
        labels = pickle.load(file)
    unique_labels = sorted(list(set(labels)))

    return features, labels, unique_labels


def get_bird_feat(labels, features, bird):
    """getting features for specific bird"""
    ind = [i for i in range(len(labels)) if labels[i] == bird]
    return features[ind,:]


def split_data(x, y, test_size=0.5):
    """split and standardize features and labels into training and test set"""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=64)
    scaler = sklearn.preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    return x_train, x_test, y_train, y_test


def store_classifier(classifiers, classifiers_file):
    """split and standardize features and labels into training and test set"""
    pickle.dump(classifiers, open(classifiers_file, 'wb'))


def store_prediction(prediction, csv_file):
    """Store prediction and its corresponding ID into a csv file"""
    with open(csv_file, 'w+') as file:
        headers = ['Id', 'Prediction']
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(prediction)



# LOG REGRESSION FUNCTIONS

def computegrad(beta, x, y, l):
    """Calculate gradient of logistic regression"""
    n = len(x)
    xy = y[:, np.newaxis]*x
    nume = np.exp(-xy.dot(beta[:, np.newaxis]))
    denom = np.exp(-xy.dot(beta[:, np.newaxis])) + 1
    series = -xy * np.divide(nume, denom)
    delta = 1/n * np.sum(series, axis=0) + 2 * l * beta
    return delta


def objective(beta, x, y, l):
    """Log regression objective function"""
    n = len(y)
    exp_part = np.exp(-y*x.dot(beta))
    log_part = np.log(1 + exp_part)
    obj = 1/n *np.sum(log_part) + l * np.square(np.linalg.norm(beta))

    return obj


def backtracking(beta, x, y, l, t, alpha=0.5, betaparam=0.8, max_iter=100):
    """ Backtracking Function to find the right step size
    Args:
        x: predictor variables
        y: response variable
        l: lambda
        beta: Current point
        t: Starting step size
        alpha: A constant, used to define decrease condition
        delta: Fraction to decrease t if the previous t doesn't work
        max_iter: Maximum iteration before it quits if condition isn't met
    Return:
        t: step size to use
    """
    grad_beta = computegrad(beta, x, y, l)
    found_t=False
    iter=0
    while not found_t and iter < max_iter:
        if objective(beta - t*grad_beta, x, y, l) < \
                (objective(beta, x, y, l) - alpha*t*np.linalg.norm(grad_beta)**2):
            found_t = True
        else:
            t = t * betaparam
            iter += 1
    # if iter == max_iter:
    #     print('Reach maximum iterations of backtracking')

    return t


def fastgradalgo(x, y, beta, theta, l=1, t_init=1, max_iter=1000):
    """Fast Gradient Algorithm with backtracking rule
    Args:
        x: predictor variables
        y: response variable
        beta: starting coefficients
        theta: starting helper coefficients
        l: lambda
        t_init: Initial step size
        max_iter: The limit of iteration
    Return:
        beta_result: The list of coefficients estimate per iteration
    """
    i = 0
    theta_results = theta
    beta_results = beta
    grad = computegrad(theta, x, y, l)
    while i < max_iter:
        t = backtracking(beta, x, y, l=l, t=t_init)
        previous_beta = copy.copy(beta)
        beta = theta - t * grad
        theta = beta + i / (i + 3) * (beta - previous_beta)
        beta_results = np.vstack((beta_results, beta))
        theta_results = np.vstack((theta_results, theta))
        grad = computegrad(theta, x, y, l)
        i += 1
        # if i % 100 == 0:
        #     print('Fastgradalgo iteration: {}'.format(i))

    return theta_results


def get_predict(x, beta):
    """Labels prediction given features"""
    return (x.dot(beta) > 0)*2 - 1


def misclassification_error(x, y, betas):
    """Calculate misclassification error"""
    me_vals = []
    n = len(betas)
    for i in range(n):
        y_predict = get_predict(x, betas[i, :])
        me = np.mean(y_predict != y)
        me_vals.append(me)
    return me_vals


def plot_misclassification_error(mis_errors_train_1, mis_errors_test_1, mis_errors_train, mis_errors_test, opt_lambda, birds):
    """Misclassification error plotting"""

    fig = plt.figure(figsize=(15, 10), dpi=100)

    plt.subplot(211)
    plt.plot(mis_errors_train_1, label='Train')
    plt.plot(mis_errors_test_1, label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error on {0} vs {1} using Lambda 1'.format(birds[0], birds[1]))

    plt.subplot(212)
    plt.plot(mis_errors_train, label='Train')
    plt.plot(mis_errors_test, label='Test')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Error')
    plt.title('Misclassification Error on {0} vs {1} using Lambda {2}'.format(birds[0], birds[1], opt_lambda))

    # plt.figure(figsize=(20, 10))
    plt.savefig('{0}-{1}_miserror.png'.format(birds[0], birds[1]), bbox_inches='tight')
    # plt.show()



# RUNNER FUNCTION

def oneone_trial(max_iter=1000):
    """Trial one to one classification using 2 random classes"""
    features, labels, unique_labels = load_data(
        features_file='train_features.p',
        labels_file='train_labels.p')
    d = features.shape[1]
    beta_init = np.zeros(d)
    theta_init = np.zeros(d)

    # birds = ['001', '002']
    birds = random.sample(set(labels), k=2)
    print('Birds {0} - {1}'.format(birds[0], birds[1]))

    x_a = get_bird_feat(labels, features, birds[0])
    x_b = get_bird_feat(labels, features, birds[1])

    x = np.vstack((x_a, x_b))
    y = np.append(np.ones(len(x_a)), -np.ones(len(x_b)))

    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.5)

    betas = fastgradalgo(x_train, y_train, beta_init, theta_init, l=1, t_init=1, max_iter=max_iter)


    mis_errors_train_1 = misclassification_error(x_train, y_train, betas)
    mis_errors_test_1 = misclassification_error(x_test, y_test, betas)

    # Using Scikit-Learn to find ideal lambda
    logistic_cv = linear_model.LogisticRegressionCV(
        penalty='l2',
        fit_intercept=False,
        max_iter=max_iter)
    logistic_cv.fit(x_train, y_train)
    opt_lambda = logistic_cv.C_[0]
    print('Optimal lambda:{}'.format(opt_lambda))

    # Using the same training set to get the convergence faster using optimal lambda
    betas = fastgradalgo(x_train, y_train, beta_init, theta_init, l=opt_lambda, t_init=1, max_iter=max_iter)

    mis_errors_train = misclassification_error(x_train, y_train, betas)
    mis_errors_test = misclassification_error(x_test, y_test, betas)

    plot_misclassification_error(

        mis_errors_train_1,
        mis_errors_test_1,
        mis_errors_train,
        mis_errors_test,
        opt_lambda,
        birds)

    print('The lowest misclassification error is {} on training dataset'.format(min(mis_errors_train_1)))
    print('The lowest misclassification error is {} on test dataset'.format(min(mis_errors_test_1)))

    print('The lowest misclassification error is {} on training dataset'.format(min(mis_errors_train)))
    print('The lowest misclassification error is {} on test dataset'.format(min(mis_errors_test)))


# CREATE PREDICTION

# create 1-on-1 classifiers
def create_classifiers(
        feat_file='train_features.p',
        label_file='train_labels.p',
        classifiers_file='1v1_classifiers.p',
        max_iter=100):
    """Create and store 1-on-1 classifiers for each pair of classes"""

    train_features, train_labels, train_unique_labels = load_data(
        features_file=feat_file,
        labels_file=label_file)

    d = train_features.shape[1]
    beta_init = np.zeros(d)
    theta_init = np.zeros(d)

    print('Classifying starts.')
    classifiers = {}
    for i in range(len(train_unique_labels)):
        bird_a = train_unique_labels[i]
        print('Classifying bird: {}'.format(bird_a))
        classifiers[bird_a] = {}
        for j in range(i+1, len(train_unique_labels)):
            bird_b = train_unique_labels[j]

            x_a = get_bird_feat(train_labels, train_features, bird_a)
            x_b = get_bird_feat(train_labels, train_features, bird_b)
            x = np.vstack((x_a, x_b))
            y = np.append(np.ones(len(x_a)), -np.ones(len(x_b)))
            x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.5)
            logistic_cv = linear_model.LogisticRegressionCV(
                penalty='l2',
                fit_intercept=False,
                max_iter=100)
            logistic_cv.fit(x_train, y_train)
            opt_lambda = logistic_cv.C_[0]

            beta = fastgradalgo(
                x_train,
                y_train,
                beta_init,
                theta_init,
                l=opt_lambda,
                t_init=1,
                max_iter=max_iter)[-1,:]

            classifiers[bird_a][bird_b] = beta
    print('Number of classifiers: {}'.format(sum(len(v) for v in classifiers.values())))
    print('Classifying ends.')

    # Save the classifiers to pickle
    store_classifier(classifiers, classifiers_file)

    return classifiers


def main(test_features=None, test_labels=None, classifiers_file='1v1_classifiers.p', csv_file='1v1_prediction.csv'):
    """Main function to return prediction"""

    with open(classifiers_file, 'rb') as f:
        classifiers = pickle.load(f)

    # Default dataset if user does not load their own dataset.
    if not test_features and not test_labels:
        test_features, test_id, test_unique_labels = load_data(
            features_file='test_features.p',
            labels_file='test_id.p')

    print('Prediction starts.')
    prediction = []

    for i in range(len(test_features)):
        vote = dict.fromkeys(list(classifiers.keys()), 0)
        for a, bird_a in classifiers.items():
            for b, bird_b in bird_a.items():
                if get_predict(test_features[i], bird_b) > 0:
                    vote[a] += 1
                else:
                    vote[b] += 1

        print('vote for row {0}: {1}'.format(i, vote))
        pred = max(vote.items(), key=itemgetter(1))[0]
        prediction.append({'Id': test_id[i], 'Prediction': pred})
    print('Prediction ends.')

    # Save one-vs-one prediction to csv
    store_prediction(prediction, csv_file)

    return prediction