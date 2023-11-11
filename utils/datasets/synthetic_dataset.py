import collections
import numpy as np

import tensorflow as tf
import tensorflow_federated as tff
import math
import os
import sys
import random
from tqdm import trange




def get_logistic_regression_dataset(dimension, num_samples_per_client, num_clients):
    """Creates logistic regression datset.

    Returns:
    A `(train, test)` tuple where `train` and `test` are `tf.data.Dataset` representing 
    the test data of all clients.
    """
    beta = tf.random.normal(shape = (dimension,))
    # include test clients
    total_clients = num_clients//10 + num_clients

    X=tf.random.normal(shape=(
    num_samples_per_client*total_clients,dimension))
    y = 1/(1+np.exp(-tf.linalg.matvec(X,beta)))
    y_labels = tf.cast(tf.round(y),tf.int32)

    return X,y_labels, beta




def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta, iid, num_users):

    dimension = 60
    NUM_CLASS = 10
    
    samples_per_user = np.random.lognormal(4, 2, (num_users)).astype(int) + 50
    #print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_train_split = [[] for _ in range(num_users)]
    y_train_split = [[] for _ in range(num_users)]

    X_test_split = [[] for _ in range(num_users)]
    y_test_split = [[] for _ in range(num_users)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, num_users)
    mean_b = mean_W
    B = np.random.normal(0, beta, num_users)
    mean_x = np.zeros((num_users, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(num_users):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        #print(mean_x[i])

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1,  NUM_CLASS)

    for i in range(num_users):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])


        train_len = int(0.9 * samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        xx_train = xx[:train_len]
        yy_train = yy[:train_len]

        X_train_split[i] = xx_train.tolist()
        y_train_split[i] = yy_train.tolist()

        xx_test = xx[train_len:]
        yy_test = yy[train_len:]

        X_test_split[i] = xx_test.tolist()
        y_test_split[i] = yy_test.tolist()

        #print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_train_split, y_train_split, X_test_split, y_test_split


def generate_federated_softmax_data(num_users,batch_size,
    client_epochs_per_round, 
    test_batch_size,
    alpha,
    beta,
    iid):


    X_train, y_train, X_test,y_test = generate_synthetic(alpha=alpha, beta=beta, iid=iid, num_users=num_users)     # synthetic (1,1)
    #X, y = generate_synthetic(alpha=0, beta=0, iid=1)      # synthetic_IID


    def get_client_train_data(client_id):
        return tf.data.Dataset.from_tensor_slices(
            collections.OrderedDict(x= X_train[client_id],
                            y= y_train[client_id],
                           ))
    def get_client_test_data(client_id):
        return tf.data.Dataset.from_tensor_slices(
            collections.OrderedDict(x= X_test[client_id],
                            y= y_test[client_id],
                           ))

    clients_ids = np.arange(num_users).tolist()
    federated_data = tff.simulation.ClientData.from_clients_and_fn(clients_ids, get_client_train_data)
    test_data  = tff.simulation.ClientData.from_clients_and_fn(clients_ids, get_client_test_data)

    def preprocess_train_dataset(dataset):
        return dataset.shuffle(buffer_size=418).repeat(
            count=client_epochs_per_round).batch(
                batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.batch(test_batch_size, drop_remainder=False)

    preprocessed_fed_train_data = federated_data.preprocess(preprocess_train_dataset)
    preprocessed_test_data = preprocess_test_dataset( test_data.create_tf_dataset_from_all_clients())
    federated_test = test_data.preprocess(preprocess_test_dataset)

    return preprocessed_fed_train_data, preprocessed_test_data, federated_test


def create_lr_federatedClientData(dimension, 
    num_samples_per_client, 
    num_clients, 
    batch_size,
    client_epochs_per_round, 
    test_batch_size):
    num_test_clients = num_clients//10
    X, y , beta = get_logistic_regression_dataset(dimension, num_samples_per_client, num_clients+num_test_clients)
    def get_client_data(client_id):
        return tf.data.Dataset.from_tensor_slices(
            collections.OrderedDict(x= X[client_id*num_samples_per_client:(client_id+1)*num_samples_per_client],
                            y= y[client_id*num_samples_per_client:(client_id+1)*num_samples_per_client],
                           ))
    clients_ids = np.arange(num_clients).tolist()
    federated_data = tff.simulation.ClientData.from_clients_and_fn(clients_ids, get_client_data)
    train, test = federated_data.train_test_client_split(federated_data, num_test_clients)

    def element_fn(element):
        return collections.OrderedDict(
            x=element['x'], y=element['y'][..., tf.newaxis])

    def preprocess_train_dataset(dataset):
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=client_epochs_per_round).batch(
                batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(test_batch_size, drop_remainder=False)

    train = train.preprocess(preprocess_train_dataset)
    test = preprocess_test_dataset( test.create_tf_dataset_from_all_clients())

    return train, test, beta