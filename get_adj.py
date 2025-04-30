import numpy as np
import pandas as pd


def get_adj_pems03():
    A = np.load('./data/pems03/pems03adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD

def get_adj_pems04():
    A = np.load('./data/pems04/pems04adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD

def get_adj_pems07():
    A = np.load('./data/pems07/pems07adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD

def get_adj_pems08():
    A = np.load('./data/pems08/pems08adj.npy')
    A[A != 0] = 1
    N = A.shape[0] 
    I = np.eye(N) 
    A_tilde = A + I 
    D_tilde = np.sum(A_tilde, axis=1)
    D_tilde_inv_sqrt = np.power(D_tilde, -0.5) 
    D_tilde_inv_sqrt[np.isinf(D_tilde_inv_sqrt)] = 0  
    D_tilde_inv_sqrt_matrix = np.diag(D_tilde_inv_sqrt)
    DAD = np.dot(D_tilde_inv_sqrt_matrix, np.dot(A_tilde, D_tilde_inv_sqrt_matrix))
    return DAD


def get_adj_metrla():
    df1 = pd.read_pickle("data/metrla/adj_mx.pkl")
    return df1[-1]

def get_adj_bay():
    df1 = pd.read_pickle("data/bay/adj_mx_bay.pkl")
    return df1[-1]

