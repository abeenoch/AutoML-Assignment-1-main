
##################################################################
# DO NOT CHANGE THE CODE IN THIS FILE
##################################################################

import numpy as np
from scipy.stats import multivariate_normal

class ParamType:
    Categorical = 0
    Real = 1
    Int = 2

class SampleType:
    Uniform = 0
    LogUniform = 1

def GET_RANGES(problem):
    """
    DO NOT CHANGE THIS FUNCTION

    Function that returns the hyperparameters and the type, range, and sampling type of them. 
    Every hyperparameter has the following fields:
      - type: the type of the parameter (ParamType.Categorical, ParamType.Int, or ParamType.Real)
      - range: the range of values from which we want to sample 
      - sample: the type of sampling that we want to use to sample candidates (SampleType.Uniform or SampleType.LogUniform)
    
    Some hyperparameters are conditional. These are marked by the extra field 
      - condition: a function that takes as input the hyperparameter configuration (in the form of a dictionary mapping hyperparameters to the chosen values)
                   and returns whether the condition for the hyperparameter to be active/well-defined is true. 
    """
    if problem == "good_range": 
        # Reasonably good range around optimum
        return {
            "nlayers": {"type": ParamType.Categorical, "range": [1,2,3,4], "sample": SampleType.Uniform}, 
            "nodes_in_layer1": {"type": ParamType.Int, "range": (32, 1028), "sample": SampleType.LogUniform, "condition": lambda x: x["nlayers"] >= 1},
            "nodes_in_layer2": {"type": ParamType.Int, "range": (32, 1028), "sample": SampleType.LogUniform, "condition": lambda x: x["nlayers"] >= 2},
            "nodes_in_layer3": {"type": ParamType.Int, "range": (32, 1028), "sample": SampleType.LogUniform, "condition": lambda x: x["nlayers"] >= 3},
            "nodes_in_layer4": {"type": ParamType.Int, "range": (32, 1028), "sample": SampleType.LogUniform, "condition": lambda x: x["nlayers"] >= 4},
            "act_fn": {"type": ParamType.Categorical, "range": ["tanh", "sigmoid", "relu"], "sample": SampleType.Uniform},
            "learning_rate": {"type": ParamType.Real, "range": (8e-5, 4e-1), "sample": SampleType.LogUniform},
        }
    elif problem == "bad_range":
        # Wide range far from optimum
        return {
            "nlayers": {"type": ParamType.Categorical, "range": [1,2,3,4], "sample": SampleType.Uniform}, 
            "nodes_in_layer1": {"type": ParamType.Int, "range": (1, 8000), "sample": SampleType.Uniform, "condition": lambda x: x["nlayers"] >= 1},
            "nodes_in_layer2": {"type": ParamType.Int, "range": (1, 8000), "sample": SampleType.Uniform, "condition": lambda x: x["nlayers"] >= 2},
            "nodes_in_layer3": {"type": ParamType.Int, "range": (1, 8000), "sample": SampleType.Uniform, "condition": lambda x: x["nlayers"] >= 3},
            "nodes_in_layer4": {"type": ParamType.Int, "range": (1, 8000), "sample": SampleType.Uniform, "condition": lambda x: x["nlayers"] >= 4},
            "act_fn": {"type": ParamType.Categorical, "range": ["tanh", "sigmoid", "relu"], "sample": SampleType.Uniform},
            "learning_rate": {"type": ParamType.Real, "range": (1e-9, 1), "sample": SampleType.Uniform},
        }
    elif problem == "interactive":
        # Interaction between hyperparameters
        return {
            "hyper1": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper2": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
        }
    else:
        return {
            "hyper1": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper2": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper3": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper4": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper5": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper6": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper7": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
            "hyper8": {"type": ParamType.Real, "range": (-40, +40), "sample": SampleType.Uniform},
        }

#####################################################################
# Code below specifies surrogate problems
#####################################################################

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def GET_PERFORMANCE_RANGE(value, hyper):
    if hyper == "nlayers":
        # best performance for 3 layers
        if value == 1:
            return 1.5
        elif value == 2:
            return 0.8
        elif value == 3:
            return 0.5
        elif value == 4:
            return 1.2
    
    elif hyper == "act_fn":
        # relu is best
        if value == "relu":
            return 0.2
        elif value == "tanh":
            return 2
        elif value == "sigmoid":
            return 2.1
        
    elif hyper == "learning_rate":
        # Gaussian in linear space on 0.01 
        val_loss = -(normal_dist(value, mean=0.01, sd=0.003) + 0.00942477796076938)*106.1032953945969
        return val_loss
    elif hyper == "nodes_per_layer":
        value = np.array(value)
        if len(value) == 1:
            val_loss = np.sum((-normal_dist(value, mean=800, sd=200)+628.305132314888)*3.6*0.0015915833701941327)/len(value)
        elif len(value) == 2:
            val_loss = np.sum((-normal_dist(value, mean=512, sd=200)+628.305132314888)*2.5*0.0015915833701941327)/len(value)
        elif len(value) == 3:
            val_loss = np.sum((-normal_dist(value, mean=256, sd=200)+628.305132314888)*1.2*0.0015915833701941327)/len(value)
        elif len(value) == 4:
            val_loss = np.sum((-normal_dist(value, mean=128, sd=200)+628.305132314888)*1.8*0.0015915833701941327)/len(value)
        return val_loss
    return 1


def GET_PERFORMANCE_INTERACTIONS(config):
    vector = list(config.values())
    var = multivariate_normal(mean=[0,5], cov=[[10,3],[3,15]])
    return -var.pdf(vector)*1e6

def GET_CONFIG_PERFORMANCE(config, problem="good_range"):
    if problem == "good_range":
        fn = GET_PERFORMANCE_RANGE
    elif problem == "bad_range":
        fn = GET_PERFORMANCE_RANGE
    elif problem == "interactive":
        return GET_PERFORMANCE_INTERACTIONS(config)
    else:
        print("problem not recognized")
        import sys; sys.exit()
    
    val_loss_total = 0
    for hyper in config:
        val_loss = fn(config[hyper], hyper)
        val_loss_total += val_loss
    return val_loss_total
        
