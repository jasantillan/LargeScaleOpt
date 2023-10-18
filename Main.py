# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""

import Problem
from ALNS import ALNS


testI = "Ca1-3,5,100.txt"
#testI = "Ca1-2,3,15.txt"
problem = Problem.TWO_E_CVRP.readInstance(testI)
costumers=problem.get_n_customers
nDestroyOps = 2
nRepairOps = 2
alns = ALNS(problem, nDestroyOps, nRepairOps)
alns.execute()
print(alns.bestSolution)
