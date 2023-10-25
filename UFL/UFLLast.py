# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

        #demanda <= binario
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i][j] > self.y[j]:
                    feasible = False
                    return feasible
            #sum of rowwise x (demanda) = 1
            if not np.sum(self.x[i, :]) == 1:
                feasible = False
                return feasible

        return feasible

    def getCosts(self, problem):
        """
        Method that computes and returns the costs of the solution
        """
        # facility = 0
        # transport = 0
        #
        # for j in range(problem.n_facilities):
        #     facility += self.y[j]*problem.f[j]
        #     for i in range(problem.n_markets):
        #         transport += self.x[i][j]*problem.c[i][j]
        #
        # total_cost = facility + transport




        facility_costs = np.sum(self.y * problem.f)
        transportation_costs = np.sum(self.x * problem.c)
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
        # If no facility is open

        for j in range(problem.n_facilities):
            for i in range(problem.n_markets):
                if problem.c[i][j] - labda[i] < 0:
                    solution.x[i][j] = 1  # Build a plant at location j and assign customer i
                    solution.y[j] = 1  # Open the facility at location j

        return solution.x,solution.y

    def convertToFeasibleSolution(self,problem,solution):
        """
        Method that, given the Lagrangian Solution computes and returns 
        a feasible solution (as a UFL_Solution)
        """

        # Create arrays to store the feasible solution
        x_feasible = np.zeros((problem.n_markets, problem.n_facilities))
        y_feasible = np.zeros(problem.n_facilities)

        if np.sum(solution.y) == 0:
            columns_sum = np.sum(problem.c,axis=0)
            columns_sum = columns_sum + problem.f
            columns_sum_idx = np.argmin(columns_sum)

            y_feasible[columns_sum_idx] = 1

            for i in range(problem.n_markets):
                x_feasible[i][columns_sum_idx] = 1

        else:
            for i in range(problem.n_markets):
                cheapest_facility_cost = float('inf')
                chosen_facility = None

                # Iterate through facilities (j) to find the cheapest open facility
                for j in range(problem.n_facilities):
                    if solution.y[j] == 1 and problem.c[i][j] < cheapest_facility_cost:
                        cheapest_facility_cost = problem.c[i][j]
                        chosen_facility = j


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
                # Keep labda unchanged
                continue
            elif sum_x > 1:
                # Decrease labda
                labda[i] -= delta
            else:
                # Increase labda
                labda[i] += delta

        return labda

    def runHeuristic(self,inst,iterations):
        """
        Method that performs the Lagrangian Heuristic. 
        """
        #initialize lists for saving variables, delta and bestUB/LB
        lower = []
        upper = []
        bestLow = []
        bestUp = []
        bestUB = float('inf')
        bestLB = -float('inf')
        delta = .5

        #Problem initialization
        problem = UFL_Problem.readInstance(inst)
        y = np.zeros(problem.n_facilities,dtype=float)
        x = np.zeros((problem.n_markets, problem.n_facilities),dtype=float)
        solution = UFL_Solution(y, x, problem)
        labda = np.full((problem.n_markets), 10,dtype=float)


        #iterations
        for i in range(iterations):
            #initialize again solution matrix with zeros for the next iteration
            solution.y = np.zeros(problem.n_facilities ,dtype=float)
            solution.x = np.zeros((problem.n_markets, problem.n_facilities),dtype=float)

            #Calculate theta (lower bound)
            LB = self.computeTheta(labda, problem, solution)
            lower.append(LB)

            #Compute the the Lagrange where condition is cij - lambdai < 0
            solution.x,solution.y = self.computeLagrangianSolution(labda,problem,solution)


            #Update weights of lambda and delta
            labda = self.updateMultipliers(labda,problem,solution.x,delta)
            delta = delta

            print(np.sum(solution.y))

            #Convert to feasible the solution
            # if solution.isFeasible() == False:
            solution.x,solution.y = self.convertToFeasibleSolution(problem,solution)

            print(np.sum(solution.y))

            #Get cost/Upper bound of the constructed feasible solution
            UB = solution.getCosts(problem)
            upper.append(UB)



            #Check for better UB and LB
            if LB > bestLB:
                bestLB = LB

            if UB < bestUB:
                bestUB = UB

            bestLow.append(bestLB)
            bestUp.append(bestUB)


            print("Iteration", i, " cost/UB: ",UB," And theta was/LB: ",LB)


        print("The best Upper bound and Lower bounds found were: ",bestUB, " and ",bestLB)


        #Printing graph
        plt.figure()  # Adjust the figure size as needed
        plt.plot(upper, label='Upper bound')
        plt.plot(bestLow, label='Best lower bound')
        plt.plot(bestUp, label='Best upper bound')
        plt.legend()  # Show legend if labels are provided
        plt.title("Lagrange Heuristic")  # Set the title if provided
        plt.grid(True)  # Add a grid to the plot
        plt.xlabel("iterations")
        plt.ylabel("cost")
        plt.show()


        return solution.x,solution.y



