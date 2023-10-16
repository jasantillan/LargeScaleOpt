# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
from Solution import Solution
import random
import copy
import time

#import math
import matplotlib.pyplot as plt
from mpmath import mp


class Parameters:
    """
    Class that holds all the parameters for ALNS
    """
    nIterations = 1000  # number of iterations of the ALNS
    minSizeNBH = 1  # minimum neighborhood size
    maxSizeNBH = 45  # maximum neighborhood size
    randomSeed = 1  # value of the random seed
    # can add parameters such as cooling rate etc.
    
    #decay, cooling, starting temperature and initializer parameters
    decay = .95
    cooling_rate = .99
    temperature = None
    starting_t = .05
    
    #initializing repair methods weight
    rd1 = 1
    rd2 = 1
    rd3= 1
    
    #initializing repair methods weight
    rr1 = 1
    rr2 = 1
    rr3= 1 
    
    #rewards per scenario
    w1 = 10 #new global solutions
    w2 = 5 #better than current
    w3 = 1 #accepeted with annealing
    w4 = 0 #rejected
    
    weights = {1:w1,2:w2,3:w3,4:w4}
    
    ############ NEW #############################
    rr_list1 = []
    rd_list1 = []
    
    rr_list2 = []
    rd_list2 = []
    
    rr_list3 = []
    rd_list3 = []
    
    i = 0
    j = 0
    k = 0
    
    best_cost = []
    current = []
    temp = []


class ALNS:
    """
    Class that models the ALNS algorithm. 

    Parameters
    ----------
    problem : TWO_E_CVRP
        The problem instance that we want to solve.
    nDestroyOps : int
        number of destroy operators.
    nRepairOps : int
        number of repair operators.
    randomGen : Random
        random number generator
    currentSolution : Solution
        The current solution in the ALNS algorithm
    bestSolution : Solution
        The best solution currently found
    bestCost : int
        Cost of the best solution

    """
    def __init__(self,problem,nDestroyOps,nRepairOps):
        self.problem = problem
        self.nDestroyOps = nDestroyOps
        self.nRepairOps = nRepairOps
        self.randomGen = random.Random(Parameters.randomSeed) #used for reproducibility
        
    
    def constructInitialSolution(self):
        """
        Method that constructs an initial solution using random insertion
        """
        self.currentSolution = Solution(self.problem,list(),list(),list(self.problem.customers.copy()))
        # Generate the second-echelon and first echelon routes by random insertion
        self.currentSolution.executeRandomInsertion(self.randomGen)
        # Calculate the cost
        self.currentSolution.computeCost()
        self.bestSolution = copy.deepcopy(self.currentSolution)
        self.bestCost = self.currentSolution.cost
        print("Created initial solution with cost: "+str(self.bestCost))
        
        ############## START SIMULATED ANNEALING TEMPEARATURE #############
        if Parameters.temperature is None:
            Parameters.temperature = (self.bestCost)*(Parameters.starting_t)
        
        
    def execute(self):
        """
        Method that executes the ALNS
        """
        starttime = time.time() # get the start time
        self.constructInitialSolution()
        for i in range(Parameters.nIterations):
            #copy the current solution
            self.tempSolution = copy.deepcopy(self.currentSolution)            
            #decide on the size of the neighbourhood
            sizeNBH = self.randomGen.randint(Parameters.minSizeNBH,Parameters.maxSizeNBH)
            #decide on the destroy and repair operator numbers
            destroyOpNr = self.determineDestroyOpNr()
            repairOpNr = self.determineRepairOpNr()
            #execute the destroy and the repair and evaluate the result
            self.destroyAndRepair(destroyOpNr, repairOpNr, sizeNBH);
            # Determine the first echelon route using the Greedy insertion
            self.tempSolution.computeCost()
            print("Iteration "+str(i)+": Found solution with cost: "+str(self.tempSolution.cost))
            #determine if the new solution is accepted
            
            ############## Get scenario for weights#################
            scenario = self.checkIfAcceptNewSol()
            
            ############### ADD NECESARRY INFO FOR WIEGHTS##############
            #update the ALNS weights
            self.updateWeights(scenario,repairOpNr,destroyOpNr)
            
            Parameters.best_cost.append(self.bestSolution.cost)
            Parameters.temp.append(self.tempSolution.cost)
            
            
            
        endtime = time.time() # get the end time
        cpuTime = round(endtime-starttime)

        print("Terminated. Final cost: "+str(self.bestSolution.cost)+", cpuTime: "+str(cpuTime)+" seconds")
        
        ################# NEWWWWWWWWW ######################################
        
        
        plt.figure(1)
        # Create a list of indices for the x-axis (time points)
        iterations = list(range(len(Parameters.rr_list1)))
        
        # Create a plot for the first list
        plt.plot(iterations, Parameters.rr_list1, label="Method Repair 1")
        
        if Parameters.rr2!= 1:
            # Create a plot for the second list
            plt.plot(iterations, Parameters.rr_list2, label="Method Repair 2")
        
        if Parameters.rr3!= 0:
             # Create a plot for the third list
            plt.plot(iterations, Parameters.rr_list3, label="Method Repair 3")
        
        # Add labels and a legend
        plt.xlabel("Iterations")
        plt.ylabel("Weight")
        plt.legend()
        
        # Display the plot
        plt.show()
        
        plt.figure(3)
        # Create a plot for the first list
        plt.plot(iterations, Parameters.rd_list1, label="Method Destroy 1")
        
        if Parameters.rd2!= 1:
            # Create a plot for the second list
            plt.plot(iterations, Parameters.rd_list2, label="Method Destroy 2")
        
        if Parameters.rd3!= 0:
             # Create a plot for the third list
            plt.plot(iterations, Parameters.rd_list3, label="Method Destroy 3")

        # Add labels and a legend
        plt.xlabel("Iterations")
        plt.ylabel("Weight")
        plt.legend()
        
        # Display the plot
        plt.show()
        
        print ("Scenario 2 ",Parameters.i)
        print ("Scenario 3 ",Parameters.j)
        print ("Scenario 4 ",Parameters.k)
        
        
        plt.figure(2)
        plt.plot(iterations, Parameters.best_cost, label="Best Cost")
        #plt.plot(iterations, Parameters.current, label="Current")
        plt.plot(iterations, Parameters.temp, label="temp")
        
        # Add labels and a legend
        plt.xlabel("Iterations")
        plt.ylabel("Costs")
        plt.legend()
        
        # Display the plot
        plt.show()
    
    def checkIfAcceptNewSol(self):
        """
        Method that checks if we accept the newly found solution
        """
        
        rand_t = random.random()
        cost_difference = self.tempSolution.cost - self.currentSolution.cost
        
        print("Best cost till now ", self.bestCost)
        
        # if we found a global best solution, we always accept
        if self.tempSolution.cost < self.bestCost:
            scenario = 1
            self.bestCost = self.tempSolution.cost
            self.bestSolution = copy.deepcopy(self.tempSolution)
            self.currentSolution = copy.deepcopy(self.tempSolution)
            print("Found new global best solution.")
            
        
        # currently, we only accept better solutions, no simulated annealing
        elif self.tempSolution.cost < self.currentSolution.cost:
            scenario = 2
            self.currentSolution = copy.deepcopy(self.tempSolution)
            print("Found new solution vs current")
            Parameters.i += 1


        elif cost_difference > 0 and rand_t < (mp.exp(-cost_difference/Parameters.temperature)):
            scenario = 3
            self.currentSolution = copy.deepcopy(self.tempSolution)
            Parameters.temperature = Parameters.temperature*Parameters.cooling_rate
            Parameters.j += 1
            print ("Accepted a worse solution")
            
        else: 
            scenario = 4
            Parameters.k += 1
        
        print (scenario)
        return scenario
              
    def updateWeights(self,scenario,repairOpNr,destroyOpNr):
        """
        Method that updates the weights of the destroy and repair operators
        The formula used for upatig the weights are:
        rho = Lambda * rho (preservation of last iterations) + (1 - lambda)*reward factor
        """
        
        if repairOpNr == 1: 
            Parameters.rr1 = Parameters.decay * Parameters.rr1 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif repairOpNr == 2:
            Parameters.rr2 = Parameters.decay * Parameters.rr2 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif repairOpNr == 3:
            Parameters.rr3 = Parameters.decay * Parameters.rr3 + (1 - Parameters.decay)*(Parameters.weights[scenario])

        if destroyOpNr == 1:
            Parameters.rd1 = Parameters.decay * Parameters.rd1 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif destroyOpNr == 2:
            Parameters.rd2 = Parameters.decay * Parameters.rd2 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif destroyOpNr == 3:
            Parameters.rd3 = Parameters.decay * Parameters.rd3 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        
        print ("Repair ",Parameters.rr1,Parameters.rr2,Parameters.rr3)
        print("Destroy ",Parameters.rd1,Parameters.rd2,Parameters.rd3)
        
        
        ################ NEWWWWWWWWWWWWW #######################33
        
        Parameters.rr_list1.append(Parameters.rr1)
        Parameters.rd_list1.append(Parameters.rd1)
        
        Parameters.rr_list2.append(Parameters.rr2)
        Parameters.rd_list2.append(Parameters.rd2)
        
        Parameters.rr_list3.append(Parameters.rr3)
        Parameters.rd_list3.append(Parameters.rd3)
        

    
    def determineDestroyOpNr(self):
        """
        Method that determines the destroy operator that will be applied. 
        Currently we just pick a random one with equal probabilities. 
        Could be extended with weights
        """
        
        #### SET NON USED INITIAL WEIGHTS to 0
        if self.nDestroyOps == 2:
            Parameters.rd3 = 0
        

        i = random.random()*(Parameters.rd1+Parameters.rd2+Parameters.rd3)
        p1 = Parameters.rd1
        p2 = Parameters.rd1 + Parameters.rd2
        
        if self.nDestroyOps == 1:
            choice = 1
        elif self.nDestroyOps > 3:
            if i<=p1:
                choice = 1
            elif i>p1 and i<=p2:
                choice = 2
        else:
            if i<=p1:
                choice = 1
            elif i>p1 and i<=p2:
                choice = 2
            else: 
                choice = 3
        
        # choice = self.randomGen.randint(1, self.nDestroyOps)
        #print ("Method for Destroy",choice)

        return choice

    def determineRepairOpNr(self):
        """
        Method that determines the repair operator that will be applied. 
        Currently we just pick a random one with equal probabilities. 
        Could be extended with weights
        """
        #### SET NON USED INITIAL WEIGHTS to 0
        if self.nRepairOps == 2:
            Parameters.rr3 = 0
        
        i = random.random()*(Parameters.rr1+Parameters.rr2+Parameters.rr3)
        p1 = Parameters.rr1
        p2 = Parameters.rr1 + Parameters.rr2
        # p3 = rr3
        
        if self.nRepairOps == 1:
            choice = 1
        elif self.nRepairOps > 3:
            if i<=p1:
                choice = 1
            elif i>p1 and i<=p2:
                choice = 2
        else:
            if i<=p1:
                choice = 1
            elif i>p1 and i<=p2:
                choice = 2
            else: 
                choice = 3
        
        # choice = self.randomGen.randint(1, self.nRepairOps)
        #print ("Method for Repair ", choice)

        
        return choice
        
    def destroyAndRepair(self,destroyHeuristicNr,repairHeuristicNr,sizeNBH):
        """
        Method that performs the destroy and repair. More destroy and/or
        repair methods can be added

        Parameters
        ----------
        destroyHeuristicNr : int
            number of the destroy operator.
        repairHeuristicNr : int
            number of the repair operator.
        sizeNBH : int
            size of the neighborhood.

        """
        #perform the destroy 
        if destroyHeuristicNr == 1:
            self.tempSolution.executeRandomRemoval(sizeNBH,self.randomGen, False)
        elif destroyHeuristicNr == 2:
            self.tempSolution.executeWorstRemoval(sizeNBH, False)
        else:
            self.tempSolution.executeDestroyMethod3(sizeNBH)
        
        #perform the repair
        if repairHeuristicNr == 1:
            self.tempSolution.executeRandomInsertion(self.randomGen)
        elif repairHeuristicNr == 2:
            self.tempSolution.greedyRepair(self.randomGen)
        else:
            self.tempSolution.executeRepairMethod3()


