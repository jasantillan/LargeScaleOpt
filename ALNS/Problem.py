# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""

import numpy as np
import math
import sys


class Customer:
    """
    Class that represents a customer with its location and nodeID

    Attributes
    ----------

    deliveryLoc : Location
        The delivery location.
    ID : int
        id of customer.

    """

    def __init__(self, deliveryLoc, ID):

        self.deliveryLoc = deliveryLoc
        self.ID = ID

    def __str__(self):
        """
        Method that prints the customer ID and the delivery location
        """
        return f"({self.ID} ,{self.deliveryLoc})"


class Location:
    """
    Class that represents either (i) a location where a customer should be delivered
    (ii) the depot for the first-echelon vehicles (iii) satellites for the second-echelon vehicles
    Attributes
    ----------
    xLoc : int
        x-coordinate.
    yLoc : int
        y-coordinate.
    demand : int
        demand quantity. For depot and satellites, demand is 0.
    servTime : int
        service time. (Extra. Don't consider in the problem)
    typeLoc : int
        1 if satellites, -1 if customers, 0 if depot
    nodeID : int
        id of the node, used for the distance matrix
    """

    def __init__(self, xLoc, yLoc, demand, servTime, typeLoc, nodeID):

        self.xLoc = xLoc
        self.yLoc = yLoc
        self.demand = demand
        self.servTime = servTime
        self.typeLoc = typeLoc
        self.nodeID = nodeID

    def __str__(self):
        """
        Method that prints the location
        """
        return f"({self.nodeID} ,{self.typeLoc})"

    def getDistance(l1,l2):
        """
        Method that computes the rounded euclidian distance between two locations
        """
        dx = l1.xLoc-l2.xLoc
        dy = l1.yLoc-l2.yLoc
        return math.sqrt(dx**2+dy**2)
        
class TWO_E_CVRP: 
    """
    Class that represents two-echelon capacitated vehicle routing problem
    Attributes
    ----------
    name : string
        name of the instance.
    customers : List of customers
        The set containing all customers.
    customerLoc: List of Locations
        The list of all customer locations
    depot : Location
        the depot where all the first-echelon vehicles must start and end.
    satellites : Location
        the satellites where all the second-echelon vehicles must start and end.
    locations : Set of Locations
        The set containing all locations: a depot, satellites and customers.
    distMatrix : 2D array
         matrix with all distances between locations
    capacity_first : int
        first-echelon vehicle capacity
    cost_first : int
        first-echelon vehicle cost
    capacity_second : int
        second-echelon vehicle capacity
    cost_second : int
        second-echelon vehicle cost
    cost_handling : int
        handling fee per transshiped unit at satellite     
    """         
    def __init__(self,name,customerLoc, depots, satellites, customers):
        self.name = name
        self.customerLoc = customerLoc
        self.depot = depots[0]
        self.satellites = satellites
        self.customers = customers
        #construct the set for all location
        self.locations = set()
        self.locations.add(self.depot)
        count = 1
        for s in self.satellites:
            s.nodeID = count
            self.locations.add(s)
            count+=1
        for c in self.customerLoc:
            c.nodeID = count
            self.locations.add(c)
            count+=1
        #compute the distance matrix 
        self.distMatrix = np.zeros((len(self.locations),len(self.locations))) #init as nxn matrix
        for i in self.locations:
            for j in self.locations:
                # No connection between Depot and customers
                case_1 = i.nodeID == 0 and j.nodeID > len(self.satellites)
                case_2 = i.nodeID > len(self.satellites) and j.nodeID == 0
                if case_1 or case_2 :
                    distItoJ = sys.maxsize
                else:
                    distItoJ = Location.getDistance(i,j)
                self.distMatrix[i.nodeID,j.nodeID] = distItoJ
        # define problem instance attribute
        # note: These values are based on the Readme-Set8.txt except the cost_handling
        self.capacity_first = 200
        self.cost_first = 50 
        self.capacity_second = 50 
        self.cost_second = 25 
        self.cost_handling = 5 
        
    def __str__(self):
        return f" 2E-CVRP problem {self.name} with {len(self.customerLoc)} customers "

    def readInstance(fileName):
        """
        Method that reads an instance from a file and returns the instances
        """

        # Read filename
        instance_name = fileName[:-4]
        instance_type = instance_name[0:3]
        n_depots = int(instance_name[4])
        n_satellites = int(instance_name[6])
        n_customers = int(instance_name[8:])
        f = open(f"C:/Users/juanb/Desktop/Universidad/TUE/LSO/Assignment 2/Instances/Must/{fileName}")

        n_line = 0  # count number of line
        nodeCount = 1  # count number of nodes
        customerLoc = []  # store customers-location object
        depot = []  # store the depot location object
        satellites = []  # store the satellite location object
        customers = []  # store customers object 
        for line in f.readlines()[:-1]:
            asList = []
            n = 6
            for index in range(0, len(line), n):
                asList.append(line[index: index + n].strip())

            if line:
                x = int(asList[0])  # need to remove ".0" from the string
                y = int(asList[1])

                if n_line < n_customers:  # For customers
                    demand = int(asList[4])
                    servTime = int(asList[5])
                    typeLoc = -1
                    # This is to renumber the customer ID
                    nodeID = nodeCount + n_satellites
                    customerLoc.append(
                        Location(x, y, demand, servTime, typeLoc, nodeID))
                    cust = Customer(
                        Location(x, y, demand, servTime, typeLoc, nodeID), nodeID)
                    nodeCount += 1
                    customers.append(cust)
                elif n_line >= n_customers and n_line < n_customers + n_satellites:  # For satellites
                    demand = 0
                    servTime = int(asList[2])
                    typeLoc = 1
                    nodeID = nodeCount
                    nodeCount += 1
                    satellites.append(
                        Location(x, y, demand, servTime, typeLoc, nodeID))
                elif n_line == n_customers + n_satellites:  # 1 Depot
                    demand = 0
                    servTime = int(asList[2])
                    typeLoc = 0
                    nodeID = 0
                    depot.append(
                        Location(x, y, demand, servTime, typeLoc, nodeID))
                n_line += 1
        return TWO_E_CVRP(fileName, customerLoc, depot, satellites, customers)
    
    def get_n_customers(self):
        return self.n_customers
