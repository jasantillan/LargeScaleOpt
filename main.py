
import UFL
import numpy as np

inst = "MO1.txt"
iterations = 1000
run = UFL.LagrangianHeuristic(inst)
LR = UFL.LagrangianHeuristic.runHeuristic(run,inst,iterations)


# problem = UFL.UFL_Problem.readInstance(inst)
# solution = UFL.UFL_Solution(y, x, problem)
# print(solution.y.shape)
