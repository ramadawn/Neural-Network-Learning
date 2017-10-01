# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 01:14:28 2017

@author: Ramadawn
"""

import numpy as np

def softmax(z):
    """Compute softmax values for each sets of scores in z."""
    return np.exp(z) / np.sum(np.exp(z), axis=0)

test = [2,1,0]

print softmax(test)