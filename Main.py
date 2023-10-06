# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""

import Problem
from ALNS import ALNS

testI = "Ca1-3,5,100.txt"
problem = Problem.TWO_E_CVRP.readInstance(testI)
nDestroyOps = 1
nRepairOps = 1
alns = ALNS(problem, nDestroyOps, nRepairOps)
alns.execute()
print(alns.bestSolution)
