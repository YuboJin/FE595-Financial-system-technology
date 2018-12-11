import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from bdateutil import isbday
from sklearn.linear_model import LinearRegression
from cvxopt import matrix, solvers
import io
import base64

def process_data(data):
    """
    F: the data of Fama French 3 factors model, including (mkt-rf), SMB, HML.
    R: the daily return for each ETF.
    Ex_R: the excess return that is equal to Return on ETF minus ris free rate.
    """
    F = data.iloc[:,:3]
    R = data.iloc[:,4:]
    Ex_R = pd.DataFrame(R.values - data.RF.values.reshape(-1,1),
                        index=R.index, columns=R.columns)
    return F,R,Ex_R

def str_to_num(x):
    """
    Convert a sting of number to number and an empty string would be set to 0
    """
    if x == '':
        x = 0
    else:
        x = float(x)
    return x


def Max_Return(beta_T, R, Ex_R, F, Lambda):
    """
    beta_T: the target risk exposure of the portfolio to the market.
    R: the historical return of the selected assets in the portfolio.
    Ex_R: the excess return of the selected assets in the portfolio.
    F: Fama-French Three Factors.
    Lambda: the risk tolerance.
    """

    n = Ex_R.shape[1]
    wp = np.ones((n, 1)) / n

    # run regression to get the beta
    lm = LinearRegression()
    lm.fit(F, Ex_R)
    beta = lm.coef_[:, 0]
    error = Ex_R - lm.predict(F)
    rho = (np.prod((R - error).values + 1, axis=0) - 1).reshape(-1, 1)

    # calculate the Indentity matrix
    Q = np.eye(n)

    # preparation for the optimization
    P = matrix(2 * Lambda * Q, tc='d')
    q = matrix(-2 * Lambda * (Q.T).dot(wp) - rho, tc='d')
    A = matrix(np.vstack((beta, [1] * n)), tc='d')
    G = matrix(np.vstack((np.eye(n), np.eye(n) * (-1))), tc='d')
    h = matrix([2] * 2 * n, tc='d')
    b = matrix([beta_T, 1], tc='d')
    # do the optimization using QP solver
    opt = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    w = opt['x']
    r = R.dot(np.array(w).reshape(-1, 1))

    return w, r


def Min_Var(R_T, R, Ex_R, F, Lambda):
    """
    R_T: the target return of the portfolio
    R: the historical return of the selected assets in the portfolio
    Ex_R: the excess return of the selected assets in the portfolio
    F: Fama-French Three Factors
    Lambda: the risk tolerance
    """

    n = Ex_R.shape[1]
    cov_f = np.cov(F, rowvar=False)
    wp = np.ones((n, 1)) / n

    # run regression to get the beta
    lm = LinearRegression()
    lm.fit(F, Ex_R)
    coeff3 = lm.coef_
    beta = coeff3[:, 0]
    error = Ex_R - lm.predict(F)
    rho = np.prod((R - error).values + 1, axis=0) - 1

    # calculate the Indentity matrix
    Q_ = np.eye(n)

    # calculate the covariance matrix
    Q = coeff3.dot(cov_f).dot(coeff3.T) + np.diag(error.var(axis=0))

    # preparation for the optimization
    P = matrix(2 * (Q + Lambda * Q_), tc='d')
    q = matrix(-2 * Lambda * (Q_.T).dot(wp), tc='d')
    G = matrix(np.vstack((np.diag([1] * n), np.diag([-1] * n))), tc='d')
    h = matrix([2] * 2 * n, tc='d')
    A = matrix(np.vstack((rho, [1] * n)), tc='d')
    b = matrix([R_T, 1], tc='d')
    # do the optimization using QP solver
    opt = solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    w = opt['x']
    r = R.dot(np.array(w).reshape(-1, 1))

    return w, r

def dealWith_business_day(Date):
    """
    Deal with the date not in business.
    If the day is weekend, add 2 days to the date for getting a business day.
    """
    date = dt.datetime.strptime(Date, '%Y-%m-%d')
    if not isbday(date):
        date += dt.timedelta(2)
    else:
        pass
    return date


def split_train_test(train_start, train_end, test_start, test_end, data):
    """
    Split the data in train set and test set
    """
    train_start = dealWith_business_day(train_start).strftime('%Y-%m-%d')
    train_end = dealWith_business_day(train_end).strftime('%Y-%m-%d')
    test_start = dealWith_business_day(test_start).strftime('%Y-%m-%d')
    test_end = dealWith_business_day(test_end).strftime('%Y-%m-%d')

    train = data.loc[train_start:train_end]
    test = data.loc[test_start:test_end]
    return train, test


def Model(data, strategy, train_start, train_end, test_start, test_end, Lambda=0.01, R_T=1, beta_T=1,):
    """
    Choose strategy.
    Substitute the calculated weights for that in train set and test set
    Get the return in both train set and test set
    """
    F, R, Ex_R = process_data(data)

    F_train, F_test = split_train_test(train_start, train_end, test_start, test_end, F)
    R_train, R_test = split_train_test(train_start, train_end, test_start, test_end, R)
    ER_train, ER_test = split_train_test(train_start, train_end, test_start, test_end, Ex_R)

    if strategy == 'Maximum Return':
        w, r_train = Max_Return(beta_T, R_train, ER_train, F_train, Lambda)
        stategy = 'MaxRet (Target_beta={})'.format(beta_T)
        r_test = R_test.dot(np.array(w)).rename(columns={0: stategy})
        r_train = r_train.rename(columns={0: stategy})

    elif strategy == 'Minimum Variance':
        w, r_train = Min_Var(R_T, R_train, ER_train, F_train, Lambda)
        stategy = 'MinVar (Target_return={})'.format(R_T)
        r_test = R_test.dot(np.array(w)).rename(columns={0: stategy})
        r_train = r_train.rename(columns={0: stategy})
    else:
        w, r_train = Max_Return(beta_T, R_train, ER_train, F_train, Lambda)
        stategy = 'MaxRet (Target_beta={})'.format(beta_T)
        r_test = R_test.dot(np.array(w)).rename(columns={0: stategy})
        r_train = r_train.rename(columns={0: stategy})
    r_train.index = R_train.index
    r_test.index = R_test.index

    return w, r_train, r_test


def show_weights(weights_list, columns_name, index):
    """
    Show the weights in dataframe
    """
    weights = [np.array(w).flatten() for w in weights_list]
    df = pd.DataFrame(data=weights, columns=columns_name, index=index)

    return df.round(4)

# plot the cumulated PnLs
def plot_PnLs(Ret_list, benchmark):
    """
    Assume we have $100 at start.
    Calculate the cumulative product for return plus 1 to get the moving track of price.
    Plot the PnL for the stategies, benchmark, and horizontal line at $100.
    """


    name = benchmark.name
    if len(Ret_list) == 2:
        data = pd.concat(Ret_list, axis=1)
    else:
        data = Ret_list
    data[name] = benchmark
    PnL = 100*np.cumprod(data+1)
    PnL.iloc[0] = 100 * np.ones(len(data.columns))
    ax = PnL.iloc[:,:-1].plot(figsize=(15,8),linewidth=2)
    PnL[name].plot(x=PnL[name].index,y=PnL[name].values,c='r',linewidth=2,sharex=ax)
    ax.plot([0,len(data)],[100,100],'k--',label='100')
    tick = [int(len(data)/6)*i for i in range(6)]+[len(data)-1]
    ax.set_xticks(tick)
    ax.set_xticklabels([data.index.values[i] for i in tick])
    ax.set_xlabel(r'$Days$', labelpad=20, fontsize=16)
    ax.set_ylabel(r'$Price$', labelpad=20, fontsize=16)
    ax.set_title(r'$The\ cumulative\ PnL$', pad=20, fontsize=18)
    ax.legend()
    ax.grid()

    figfile = io.BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    return figdata_png

