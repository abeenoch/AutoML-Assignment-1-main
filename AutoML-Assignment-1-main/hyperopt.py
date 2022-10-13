#########################################################################################
# The assignment file. While it is allowed to change this entire file, we highly
# recommend using the provided template. YOU MUST USE THE RANGES AND HYPERPARAMETERS SPECIFIED
# IN GET_RANGES AND GET_CONFIG_PERFORMAMCE (IN SHORT: USE OUR SURROGATE PROBLEMS)
#########################################################################################
from __future__ import division
import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, truncnorm

from utils import GET_CONFIG_PERFORMANCE, GET_RANGES, SampleType, ParamType # make sure to make use of ParamType and SampleType in your code


parser = argparse.ArgumentParser()
parser.add_argument('--problem', choices=['good_range', 'bad_range', 'interactive'], required=True)
parser.add_argument('--algorithm', choices=['rs','tpe'], required=True)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--random_warmup', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)

"""
Function that performs random search on the given problem. It uses the
ranges and sampling types defined in GET_RANGES(problem) (see utils.py).

Arguments:
  - problem (str): the prolbem identifier
  - function_evaluations (int): the number of configurations to evaluate
  - **kwargs: any other keyword arguments

Returns:
  - history (list): A list of the observed losses
  - configs (list): A list of the tried configurations. Every configuration is a dictionary
                        mapping hyperparameter names to the chosen values
"""


def random_search(problem, function_evaluations=150, **kwargs): #function_evaluations should be 150

    history = []
    
    configs = []

    # get all information about the hyperparameters we want to tune for this problem
    # (see utils.py) for the form of this.
    RANGES = GET_RANGES(problem)
    for n in range(function_evaluations):
        config = {}
        for name in RANGES: # for each choice / hyperparameter
            hyperparameter = RANGES[name]
            if "condition" not in hyperparameter.keys() or hyperparameter["condition"](config) == True:
                ParamRange = hyperparameter["range"]
                if hyperparameter["type"] == 0:
                    p = [1/len(ParamRange) for _ in ParamRange]
                    choice = np.random.choice(ParamRange, p = p)
                elif hyperparameter["sample"] == 0:
                    choice = np.random.uniform(ParamRange[0],ParamRange[1])
                elif hyperparameter["sample"] == 1:
                    choice = loguniform.rvs(ParamRange[0],ParamRange[1])
                if hyperparameter["type"] == 2:
                    choice = np.round(choice).astype(int)

                config[name] = choice
        loss = GET_CONFIG_PERFORMANCE(config)
        history.append(loss)
        configs.append(config)

    return history, configs


"""
Arguments:
    -mix : contains means, standard deviations and range
Returns:
    - sample form mixture of distributions (float)

Samples from each distribution; Final sample is random choice from all samples
"""
def sample_mix(mix):
    means, sds, a,b = mix
    mean = (b-a)/2.0
    sd = b-mean
    samples = [truncnorm.rvs((a-mean)/sd,(b-mean)/sd,loc=mean, scale=sd)] #prior
    try:         
       result = samples
    except ZeroDivisionError:
        result = 0
    for mean, sd in zip(means,sds):
        samples.append(truncnorm.rvs((a-mean)/sd,(b-mean)/sd,loc=mean, scale=sd))
    return np.random.choice(samples)

"""
Arguments:
    -samples : list of candidate samples, sampled from lx
    -mix : contains means, standard deviations and range
Returns:
    - sample form mixture of distributions (float)

returns list of probability densities for the given samples and mix of distributions
"""
def mix_pdf(samples, mix):
    means, sds, a,b = mix
    mean = (b-a)/2
    sd = b-mean
    pdf = truncnorm.pdf(samples, (a-mean)/sd,(b-mean)/sd,loc=mean, scale=sd) #prior
    for mean, sd in zip(means,sds):
        pdf += truncnorm.pdf(samples, (a-mean)/sd,(b-mean)/sd,loc=mean, scale=sd)
    return pdf

"""
Arguments:
    - observations: list of observations for constructing l(x) or g(x)
    - a,b hyperparameter range
    - log : (default: false) if true,
        changes means, a and b the log(means), log(a) and log(b)
Returns:
    - means : list of means for the normal distributions
    - sds : list of standard deviations for the bormal distributions
    - a,b : range
"""
def construct_mix(obs,a,b, log = False):
    means = []
    sds = []
    obs = sorted(obs)
    x = np.append(np.insert(obs,0,a),b)
    if log == True:
        x = np.log(x + abs(a) + 1)
        a,b = [np.log(a + (abs(a) + 1)), np.log(b + (abs(a) + 1))]
    for i in range(1,len(x)-1):
        means.append(x[i])
        sds.append(max((x[i]- x[i-1]), (x[i+1] - x[i])))
    return means,sds, a, b


def tpe(problem, function_evaluations=150, random_warmup=10, gamma=0.2, **kwargs): #function_evaluations = 150

    history = []
    configs = []

    history,configs = random_search(problem, random_warmup)
    RANGES = GET_RANGES(problem)
    # n_c number of candidates
    n_candidates = 1 # not sure wether this is needed
    for i in range(function_evaluations):

        _,D = zip(*sorted(zip(history, configs),key = lambda y: y[0])) #revers true?

        y_star = int(gamma * len(D))
        config = {}
        for name in RANGES:
            hyperparameter = RANGES[name]
            if "condition" not in hyperparameter.keys() or hyperparameter["condition"](config) == True:
                ParamRange = hyperparameter["range"]
                #get all config_i for hyperparameter i for Configs
                observations = []
                for d in D:
                    if name in d.keys():
                        observations.append(d[name])
                D_l = np.unique(observations[:y_star]) # observations to use for constructing l(x)
                D_g = np.unique(observations[y_star:]) # observations to use for constructing g(x)

                if hyperparameter["type"] == 0:
                    lx = [(1 + (np.count_nonzero(D_l == p)))/(len(ParamRange) + len(D_l)) for p in ParamRange]
                    gx = [(1 + (np.count_nonzero(D_g == p)))/(len(ParamRange) + len(D_g)) for p in ParamRange]
                    samples = ParamRange
                    prom = np.array(lx) / np.array(gx)

                elif hyperparameter["sample"] == 0:
                    lx = construct_mix(D_l,ParamRange[0],ParamRange[1])
                    gx = construct_mix(D_g,ParamRange[0],ParamRange[1])
                    samples = [sample_mix(lx) for _ in range(n_candidates)]
                    prom = np.array(mix_pdf(samples,lx)) / np.array(mix_pdf(samples,gx))

                elif hyperparameter["sample"] == 1:
                    lx = construct_mix(D_l,ParamRange[0],ParamRange[1], log=True)
                    gx = construct_mix(D_g,ParamRange[0],ParamRange[1], log=True)
                    samples = [sample_mix(lx) for _ in range(n_candidates)]
                    prom = np.array(mix_pdf(samples,lx)) / np.array(mix_pdf(samples,gx))
                    samples = [np.exp(s) - abs(lx[2]) -1 for s in samples]

                choice = samples[np.argmax(prom)]

                if hyperparameter["type"] == 2:
                    choice = np.round(choice).astype(int)

                config[name] = choice

        loss = GET_CONFIG_PERFORMANCE(config)

        history.append(loss)
        configs.append(config)
    # TODO: implement the rest of the function
    return history, configs


###############################################################################################
# Code that parses command line arguments and saves the results
# code can be run by calling
# python hyperopt.py --algorithm ALG_SPECIFIER --problem PROBLEM_SPECIFIER --more_arguments ...
# you do not need to change the code below
###############################################################################################
alg_fn = {'rs': random_search, 'tpe':tpe}

args = parser.parse_args()
np.random.seed(args.seed)

conf = vars(args)
tried_configs, performances = alg_fn[args.algorithm](**conf)
if not os.path.isdir('./results'):
    os.mkdir('./results')
savename = f"./results/{args.algorithm}-{args.problem}-{args.gamma}-{args.random_warmup}-{args.seed}-perfs.csv"

df = pd.DataFrame(tried_configs)
df["val_loss"] = performances
df.to_csv(savename)
