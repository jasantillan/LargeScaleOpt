# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np


class UFL_Problem:
    """
    Class that represent a problem instance of the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    f : numpy array
        the yearly fixed operational costs of all facilities
    c : numpy 2-D array (matrix)
        the yearly transportation cost of delivering all demand from markets to facilities
    n_markets : int
        number of all markets.
    n_facilities : int
        number of all available locations.
    """

    def __init__(self, f, c, n_markets, n_facilities):

        self.f = f
        self.c = c
        self.n_markets = n_markets
        self.n_facilities = n_facilities

    def __str__(self):
        return f" Uncapacitated Facility Location Problem: {self.n_markets} markets, {self.n_facilities} facilities"

    def readInstance(fileName):
        """
        Read the instance fileName

        Parameters
        ----------
        fileName : str
            instance name in the folder Instance.

        Returns
        -------
        UFL Object

        """
        # Read filename
        f = open(f"Instances/{fileName}")
        n_line = 0
        n_markets = 0
        n_facilities = 0
        n_row = 0
        for line in f.readlines():
            asList = line.replace(" ", "_").split("_")
            if line:
                if n_line == 0:
                    n_markets = int(asList[0])
                    n_facilities = int(asList[1])
                    f_j = np.empty(n_markets)
                    c_ij = np.empty((n_markets, n_facilities))
                elif n_line <= n_markets:  # For customers
                    index = n_line - 1
                    f_j[index] = asList[1]
                else:
                    if len(asList) == 1:
                        n_row += 1
                        demand_i = float(asList[0])
                        n_column = 0
                    else:
                        for i in range(len(asList)-1):
                            c_ij[n_row-1, n_column] = demand_i * \
                                float(asList[i])
                            n_column += 1
            n_line += 1
        return UFL_Problem(f_j, c_ij, n_markets, n_facilities)

class UFL_Solution:
    """
    Class that represent a solution to the Uncapcitated Facility Location Problem
        
    Attributes
    ----------
    y : numpy array
        binary array indicating whether facilities are open
    x : numpy 2-D array (matrix)
        fraction of demand from markets sourced from facilities
    instance: UFL_Problem
        the problem instance
    """

    def __init__(self, y, x, instance):
        self.y = y
        self.x = x
        self.instance = instance

    def isFeasible(self):
        """
        Method that checks whether the solution is feasible
        
        Returns true if feasible, false otherwise
        """
        feasible = True

        #Binario
        if not np.all(np.logical_or(self.y == 0, self.y == 1)):
            feasible = False
            return feasible


        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i][j] > self.y[j]:
                    feasible = False
                    return feasible

            if np.sum(self.x[i, :]) != 1:
                feasible = False
                return feasible

        return feasible

    def getCosts(self, problem,x,y):
        """
        Method that computes and returns the costs of the solution
        """
        facility_costs = np.sum(y * problem.f)
        transportation_costs = np.sum(np.sum(x * problem.c, axis=1))

        #possible penalization costs ???
        total_cost = facility_costs + transportation_costs


        return total_cost

class LagrangianHeuristic:
    """
    Class used for the Lagrangian Heuristic
        
    Attributes
    ----------
    instance : UFL_Problem
        the problem instance
    """

    def __init__(self,instance):
        self.instance = instance

    def computeTheta(self,labda,problem,solution):
        """
        Method that, given an array of Lagrangian multipliers computes and returns
        the optimal value of the Lagrangian problem
        """
        # Initialize the Lagrangian lower bound
        theta = 0

        # Compute the contribution of λ_i terms
        theta += np.sum(labda)

        # Compute the contribution of (C_ij - λ_i) * x_ij terms
        for j in range(problem.n_facilities):
            for i in range(problem.n_facilities):
                theta += (problem.c[i][j] - labda[i]) * solution.x[i][j]

        # Compute the contribution of f_j * y_j terms
        for j in range(problem.n_facilities):
            theta += problem.f[j] * solution.y[j]

        return theta


    def computeLagrangianSolution(self,labda,problem,solution):
        """
        Method that, given an array of Lagrangian multipliers computes and returns 
        the Lagrangian solution (as a UFL_Solution)
        """
        for j in range(problem.n_facilities):
            for i in range(problem.n_markets):
                if problem.c[i][j] - labda[i] < 0:
                    solution.x[i][j] = 1  # Build a plant at location j and assign customer i
                    solution.y[j] = 1  # Open the facility at location j

        return solution.x

    def convertToFeasibleSolution(self,problem,solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """
        # Create arrays to store the feasible solution
        x_feasible = np.zeros((problem.n_markets, problem.n_facilities))
        y_feasible = np.zeros(problem.n_facilities)

        # Find out if no facility is open
        # if np.sum(y) == 0:
        #     cheapest_open_cost = float('inf')
        #     chosen_facility = None
        #
        #     # Iterate through facilities (j) to find the cheapest facility to open
        #     for j in range(self.n):
        #         if self.f[j] < cheapest_open_cost:
        #             cheapest_open_cost = self.f[j]
        #             chosen_facility = j
        #
        #     if chosen_facility is not None:
        #         y_feasible[chosen_facility] = 1  # Open the chosen facility

        # Iterate through customers (i)
        for i in range(problem.n_markets):
            cheapest_facility_cost = float('inf')
            chosen_facility = None

            # Iterate through facilities (j) to find the cheapest open facility
            for j in range(problem.n_facilities):
                if solution.y[j] == 1 and problem.c[i][j] < cheapest_facility_cost:
                    cheapest_facility_cost = problem.c[i][j]
                    chosen_facility = j

            #Condition for scenarios where there are not open facilities
            if chosen_facility is not None:
                x_feasible[i][chosen_facility] = 1
                y_feasible[chosen_facility] = 1

        return x_feasible,y_feasible


    def updateMultipliers(self,labda,problem,x_initial,delta):
        """
        Method that, given the previous Lagrangian multipliers and Lagrangian Solution 
        updates and returns a new array of Lagrangian multipliers
        """
        for i in range(problem.n_facilities):
            sum_x = np.sum(x_initial[i, :])
            if sum_x == 1:
                # Keep λ_i unchanged
                continue
            elif sum_x > 1:
                # Decrease λ_i
                labda[i] -= delta  # Adjust the step size as needed
            else:
                # Increase λ_i
                labda[i] += delta  # Adjust the step size as needed

        return labda

    def runHeuristic(self,inst,iterations):
        """
        Method that performs the Lagrangian Heuristic. 
        """
        lower = []
        upper = []
        bestLow = []
        bestUp = []
        bestUB = float('inf')
        bestLB = -float('inf')
        delta = 1

        #Problem initialization
        problem = UFL_Problem.readInstance(inst)
        y = np.zeros(problem.n_facilities,dtype=float)
        x = np.zeros((problem.n_markets, problem.n_facilities),dtype=float)
        solution = UFL_Solution(y, x, problem)
        labda = np.full((problem.n_markets), 15,dtype=float)


        #iterations
        for i in range(iterations):
            solution.y = np.zeros(problem.n_facilities ,dtype=float)
            solution.x = np.zeros((problem.n_markets, problem.n_facilities),dtype=float)
            LB = self.computeTheta(labda, problem, solution)
            lower.append(LB)
            x_initial = self.computeLagrangianSolution(labda,problem,solution)
            #feasible = UFL_Solution.isFeasible(solution)
            #print("This iteration is feasible? ",feasible)

            solution.x,solution.y = self.convertToFeasibleSolution(problem,solution)


            #print("After feasibility, this iteration is feasible? ",UFL_Solution.isFeasible(UFL_Solution))
            UB = UFL_Solution.getCosts(UFL_Solution, problem, solution.x,solution.y)
            upper.append(UB)

            if LB > bestLB:
                bestLB = LB

            if UB < bestUB:
                bestUB = UB

            bestLow.append(bestLB)
            bestUp.append(bestUB)


            print("Iteration cost/UB: ",UB," And theta was/LB: ",LB)

            labda = self.updateMultipliers(labda,problem,x_initial,delta)


        #solution.x, solution.y = x_feasible, y_feasible
        final_cost = UFL_Solution.getCosts(UFL_Solution, problem, solution.x, solution.y)

        print("Final Cost is ",final_cost)
        print(bestUB,bestLB)
