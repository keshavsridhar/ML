from __future__ import division
import pandas as pd
import numpy as np
import random as rd
import math as math
import matplotlib.pyplot as plt
# import scipy.stats as stat


class loglike:
    def __init__(self):
        self.logs = []
        self.old = 0
        self.new = 1

    def checklogs(self, l):
        self.logs.append(l)
        self.old = self.new
        self.new = l


class Gaussian:
    def __init__(self):
        self.mean = 0
        self.sd = 0
        self.weight = 0
        self.distribution = []
        self.n = 0


def dnorm(x, mu, sigma):
    dn = math.exp(-0.5*(pow(((x - mu)/sigma), 2)))/(math.sqrt(2*math.pi)*sigma)
    return dn


def em(df, k, llkhd):
    gs = [Gaussian() for i in range(0, k)]
    for i in range(0, k):
        gs[i].weight = 1/k
    iter1 = 0
    # Initialization step:
    for i in gs:
        i.mean = float(rd.choice(df[0]))
        # i.sd = rd.random()
        i.sd = float(rd.randint(1, 5))

    while llkhd.new != llkhd.old:
        iter1 += 1
        assert (1 - sum([i.weight for i in gs])) <= 0.011

        # Expectation step:
        for j in range(0, k):
            # Numerator = weight * prob(x|c):
            df["weight*P(x|c"+str(j)+")"] = [dnorm(i, mu=gs[j].mean, sigma=gs[j].sd) * gs[j].weight for i in df[0]]

        # Denominator = Normalizing
        df["Sums_denom"] = 0
        for i in range(0, k):
            df["Sums_denom"] += df["weight*P(x|c"+str(i)+")"]
        df["Log_Sums_denom"] = np.log(df["Sums_denom"])

        # Responsibilities/Posteriors:
        for i in range(0, k):
            df["Resp"+str(i)] = df["weight*P(x|c"+str(i)+")"] / df["Sums_denom"]

        llkhd.checklogs(round(sum(df["Log_Sums_denom"]), 8))
        # print(df.head(3))
        # print(llkhd.logs)
        print(llkhd.new)

        # Maximization step:
        for j in range(len(gs)):
            gs[j].n = sum(df["Resp"+str(j)])
            # New mean:
            gs[j].mean = (1/gs[j].n)*(sum(df["Resp"+str(j)]*df[0]))
            # New sd:
            gs[j].sd = math.sqrt(sum(df["Resp" + str(j)] * (df[0] - gs[j].mean)**2)/gs[j].n)
            # New priors:
            # print(gs[j].n)
            gs[j].weight = gs[j].n/len(df)
            # print(gs[j].n)
            # print(gs[j].mean)
            # print(gs[j].sd)
            # print(gs[j].weight)
    return gs, iter1


def fileread(f1):
    input_df = pd.read_csv(f1, header=None)
    return input_df

if __name__ == "__main__":
    f = fileread("data1.txt")
    log_likelihood = loglike()
    # print(f.describe())
    # f = np.array(f)
    print("Enter number of gaussians: ")
    k = int(input())
    params, itr = em(f, k, log_likelihood)
    plt.style.use("ggplot")
    plt.title("Log-likelihood vs Number of Iterations:")
    plt.scatter(range(0, len(log_likelihood.logs)), log_likelihood.logs, marker="x", c="Blue")
    plt.ylabel("Log likelihood")
    plt.xlabel("Iteration #")
    plt.show()
    print("Total number of iterations : %d" % itr)
    for i in range(len(params)):
        print("Gaussian %d parameters:" % i)
        print("Mean : %f" % params[i].mean)
        print("Standard deviation : %f" % params[i].sd)
        print("Weight : %f" % params[i].weight)
