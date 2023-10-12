# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
from Solution import Solution
import random
import copy
import time


class Parameters:
    """
    Class that holds all the parameters for ALNS
    """
    nIterations = 10  # number of iterations of the ALNS
    minSizeNBH = 1  # minimum neighborhood size
    maxSizeNBH = 45  # maximum neighborhood size
    randomSeed = 1  # value of the random seed
    # can add parameters such as cooling rate etc.


    #decay parameter
    decay = .95
    
    #initializing repair methods weight
    rd1 = 1
    rd2 = 1
    rd3= 1
    
    #initializing repair methods weight
    rr1 = 1
    rr2 = 1
    rr3= 1 
    
    #rewards per scenario
    w1 = 5 #new global solutions
    w2 = 1 #better than current
    w3 = 0 #rejected
    
    weights = {1:w1,2:w2,3:w3}



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

            ############## GET SCENARIO ##################
            scenario = self.checkIfAcceptNewSol()


            ############## ADD THE NECESSARY INFO FOR UPDATE WEIGHTS ##################
            #update the ALNS weights 
            self.updateWeights(scenario,repairOpNr,destroyOpNr)

        
        endtime = time.time() # get the end time
        cpuTime = round(endtime-starttime)

        print("Terminated. Final cost: "+str(self.bestSolution.cost)+", cpuTime: "+str(cpuTime)+" seconds")
    
    def checkIfAcceptNewSol(self):
        """
        Method that checks if we accept the newly found solution
        """
        # if we found a global best solution, we always accept
        if self.tempSolution.cost < self.bestCost:
            self.bestCost = self.tempSolution.cost
            self.bestSolution = copy.deepcopy(self.tempSolution)
            self.currentSolution = copy.deepcopy(self.tempSolution)
            print("Found new global best solution.")
            
            scenario = 1
        
        # currently, we only accept better solutions, no simulated annealing
        if self.tempSolution.cost<self.currentSolution.cost:
            self.currentSolution = copy.deepcopy(self.tempSolution)
            
            scenario = 2

        else: 
            scenario = 3
    
    def updateWeights(self):
        """
        Method that updates the weights of the destroy and repair operators
        The formula used for upatig the weights are:
        rho = Lambda * rho (preservation of last iterations) + (1 - lambda)*reward factor
        """
        if repairOpNr == 1: 
            Parameters.rr1 = Parameters.decay * Parameters.rr1 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif repairOpNr == 2:
            Parameters.rr2 = Parameters.decay * Parameters.rr2 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif repairOpNr == 2:
            Parameters.rr3 = Parameters.decay * Parameters.rr3 + (1 - Parameters.decay)*(Parameters.weights[scenario])

        if destroyOpNr == 1:
            Parameters.rd1 = Parameters.decay * Parameters.rd1 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif destroyOpNr == 2:
            Parameters.rd2 = Parameters.decay * Parameters.rd2 + (1 - Parameters.decay)*(Parameters.weights[scenario])
        elif destroyOpNr == 2:
            Parameters.rd3 = Parameters.decay * Parameters.rd3 + (1 - Parameters.decay)*(Parameters.weights[scenario])
    
    def determineDestroyOpNr(self):
        """
        Method that determines the destroy operator that will be applied. 
        Currently we just pick a random one with equal probabilities. 
        Could be extended with weights
        """
        i = random.random()*(Parameters.rd1+Parameters.rd2+Parameters.rd3)
        p1 = Parameters.rd1
        p2 = Parameters.rd1 + Parameters.rd2
        # p3 = rd3
        
        if i<=p1:
            choice = 1
        elif i>p1 and i<=p2:
            choice = 2
        else:
            choice = 3
        
        # choice = self.randomGen.randint(1, self.nDestroyOps)
        print ("Method for Destroy ",choice)

        return choice
    
    def determineRepairOpNr(self):
        """
        Method that determines the repair operator that will be applied. 
        Currently we just pick a random one with equal probabilities. 
        Could be extended with weights
        """
        i = random.random()*(Parameters.rr1+Parameters.rr2+Parameters.rr3)
        p1 = Parameters.rr1
        p2 = Parameters.rr1 + Parameters.rr2
        # p3 = rr3
        
        if i<=p1:
            choice = 1
        elif i>p1 and i<=p2:
            choice = 2
        else:
            choice = 3
        
        # choice = self.randomGen.randint(1, self.nDestroyOps)
        print ("Method for Repair ", choice)
        

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
            self.tempSolution.executeDestroyMethod2(sizeNBH)
        else:
            self.tempSolution.executeDestroyMethod3(sizeNBH)
        
        #perform the repair
        if destroyHeuristicNr == 1:
            self.tempSolution.executeRandomInsertion(self.randomGen)
        elif destroyHeuristicNr == 2:
            self.tempSolution.executeRepairMethod2()
        else:
            self.tempSolution.executeRepairMethod3()


