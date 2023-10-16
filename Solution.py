# -*- coding: utf-8 -*-
"""
@author: Original template by Rolf van Lieshout and Krissada Tundulyasaree
"""
from Route import Route


class Solution:
    """
    Method that represents a solution to the 2E-CVRP

    Attributes
    ----------
    problem : 2E-CVR
        the problem that corresponds to this solution
    routes_1 : List of Routes
         Routes for the first echelon vehicle in the current solution
    routes_2 : List of Routes
         Routes for the second echelon vehicle the current solution
    served : List of Customers
        Customers served in the second echelon vehicle current solution
    notServed : List of Customers
         Customers not served in the second echelon vehicle current solution
    distance : double
        total distance of the current solution
    cost: double
        total cost consisting of handling, distance and vehicle cost.
    handling: double
        total handling cost of loads at satellites
    satDemandServed : List 
        Load served in the first echelon vehicle current solution
    satDemandNotServed : List 
         Load not served in the first echelon vehicle current solution
    """

    def __init__(self, problem, routes_2, served, notServed):

        self.problem = problem
        self.routes_2 = routes_2
        self.served = served
        self.notServed = notServed

    def computeDistance(self):
        """
        Method that computes the distance of the solution
        """
        self.distance = 0
        # Calculate the cost for the first echelon
        for routes_1 in self.routes_1:
            self.distance += routes_1.distance
        # Calculate the cost for the second echelon
        for routes_2 in self.routes_2:
            self.distance += routes_2.distance

    def computeCost(self):    
        """
        Method that computes total cost = load handling cost + vehicle cost + transportation cost

        """
        self.computeDistance()
        # Calculate the handling cost
        handling = self.problem.cost_handling * sum(self.satDemandServed)
        # Calculate the vehicle cost
        vehicle_cost = self.problem.cost_first * len(self.routes_1) + self.problem.cost_second * len(self.routes_2)
        # Calculate the total cost
        self.cost = handling + self.distance + vehicle_cost
                   
    def __str__(self): 
        """
        Method that prints the solution
        """
        # Check the second echelon to generate order for the first echelon
        print(f"First-echelon Solution with Satellite demand {self.satDemandServed}")
        for route in self.routes_1:
            print(route)
                
        nRoutes = len(self.routes_2)
        nNotServed = len(self.notServed)
        print("Second-echelon Solution with "+str(nRoutes)+" routes and "+str(nNotServed)+" unserved customers: ")
        for route in self.routes_2:
            print(route)
            
        s= ""
        
        return s
    
    def executeRandomRemoval(self,nRemove,random, firstEchelon):
        """
        Method that executes a random removal of locations
        
        This is destroy method number 1 in the ALNS

        Parameters
        ----------
        nRemove : int
            number of customers that is removed.                 
        randomGen : Random
            Used to generate random numbers        
        firstEchelon: Boolean
            True if choose to remove location from the first-echelon routes
            False otherwise

        """
        if firstEchelon is True:
            routes = self.routes_1
        else:
            routes = self.routes_2
        for i in range(nRemove):
            # terminate if no more customers/loads are served
            if len(routes)==0: 
                break
            #pick a random customer/load and remove it from the solution
            while True:
                route = random.choice(routes)
                # if the route has loads/customers.
                if len(route.locations) > 2:
                    break
                len_route = [len(i.locations) for i in routes]
                # All routes are empty: no served loads or customers
                if sum(len_route) == 2 * len(routes):
                    break
            if len(route.locations) == 2:
                break
            loc = random.choice(route.locations[1:-1]) 
            self.removeLocation(loc, firstEchelon, route)
            
            
    def executeWorstRemoval (self,nRemove, firstEchelon):
        if firstEchelon is True:
            routes = self.routes_1
        else:
            routes = self.routes_2
        for i in range(nRemove):
            if len(routes)==0: 
                break
            maxcost=-1
            maxroute=None
            for j in routes:
                if j.cost>maxcost:
                    maxcost=j.cost
                    maxroute=j
            if len(maxroute.locations) == 2:
                break
            mdemand=-1
            for k in maxroute.locations[1:-1]:
                if k.demand>mdemand:
                    mdemand=k.demand
                    loc=k
            
            self.removeLocation(loc,firstEchelon, maxroute)
                    
    
    def removeLocation(self,location, firstEchelon, route):
        """
        Method that removes a location from the indicated level of echelon vehicles


        Parameters
        ----------
        location : Location
            Location to be removed
        firstEchelon : Boolean
            True if choose to remove location from the first-echelon routes
            False otherwise        
        route : Route
            This is the route which have the location to be removed

        Returns
        -------
        None.

        """
        
        # Remove the location from the route
        location_index, load = route.removeLocation(location)
        if firstEchelon is True:
            # update lists with served and unserved load
            self.satDemandServed[location.nodeID - 1] -= load
            self.satDemandNotServed[location.nodeID - 1] += load
        else:
            # update lists with served and unserved customers
            customer = 0
            for i in self.served:
                if i.deliveryLoc.nodeID == location.nodeID:
                    customer = i
            self.served.remove(customer)
            self.notServed.append(customer)    
    

    def computeDemandSatellites(self):
        """
        Generate a list of total demand for all satellites given the second echelon routes

        Returns
        -------
        None.

        """
        # number of satellites
        nSat = len(self.problem.satellites)
        self.satDemandNotServed = [0 for i in range(nSat)]
        # Find total demand for each satellite from the second echelon routes          
        for i in self.routes_2:       
            sat = i.locations[0]
            totalDemand = sum(j.demand for j in i.locations)
            self.satDemandNotServed[sat.nodeID - 1] += totalDemand

    def executeRandomInsertion(self, randomGen):
        """
        Method that contruct randomly the routes for the first and second echelon vehicles by 
        1. randomly insert the customers to create the second echelon routes.
        2. depending on the constructed second echelon routes, randomly insert demand at the
        satellites to construct the first echelon routes.
        
        This is repair method number 1 in the ALNS

        Parameters
        ----------
        randomGen : Random
            Used to generate random numbers

        Returns
        -------
        None.

        """
        
        self.executeRandomInsertionSecond(randomGen)
        # Based on the second echelon routes, generate the first echelon routes
        self.executeRandomInsertionFirst(randomGen)

    def executeRandomInsertionFirst(self,randomGen):
        """
        Method that randomly inserts the demand of each satellite to construct the routes
        for the first echelon vehciles. Note that we assume given second-echelon routes
        to determine demand of each satellite.
                
        Parameters
        ----------
        randomGen : Random
            Used to generate random numbers

        """
        # Determine the first echelon from the given-second echelon routes
        # This is used to reset the existing first-echelon route.
        self.routes_1 = []
        # Derive demands for satellites
        self.computeDemandSatellites()
        # iterate over the list with unserved customers
        self.satDemandServed = [0 for i in range(len(self.satDemandNotServed))]
        while sum(self.satDemandNotServed) > 0:
            #pick a satellite with some loads for the first echelon vehicle to deliver
            load_max = 0
            while load_max == 0:
                load_max = randomGen.choice(self.satDemandNotServed)
            sat_ID = self.satDemandNotServed.index(load_max)
            # keep track of routes in which this load could be inserted
            potentialRoutes = self.routes_1.copy()
            # potentialRoutes = copy.deepcopy(self.routes_1)
            inserted = False
            while len(potentialRoutes) > 0:
                # pick a random route
                randomRoute = randomGen.choice(potentialRoutes)
                remain_load = self.problem.capacity_first - sum(randomRoute.servedLoad)
                if load_max > remain_load:
                     load = remain_load
                else:
                     load = load_max
                afterInsertion = randomRoute.greedyInsert(
                    self.problem.satellites[sat_ID], load)
                if afterInsertion == None:
                    # insertion not feasible, remove route from potential routes
                    potentialRoutes.remove(randomRoute)
                else:
                    # insertion feasible, update routes and break from while loop
                    inserted = True                   
                    self.routes_1.remove(randomRoute)
                    self.routes_1.append(afterInsertion)
                    break
            # if we were not able to insert, create a new route
            if not inserted:
                # create a new route with the load
                depot = self.problem.depot
                locList = [depot, self.problem.satellites[sat_ID], depot]
                remain_load = self.problem.capacity_first 
                if load_max > remain_load:
                     load = remain_load
                else:
                     load = load_max
                newRoute = Route(locList, self.problem, True, [load])
                # update the demand
                self.routes_1.append(newRoute)
            # update the lists with served and notServed customers
            self.satDemandNotServed[sat_ID] -= load
            self.satDemandServed[sat_ID] += load

    def executeRandomInsertionSecond(self, randomGen):
        """
        Method that randomly inserts the unserved customers in the solution for the second echelon routes.
        
        Parameters
        ----------
        randomGen : Random
            Used to generate random numbers

        """
        # iterate over the list with unserved customers
        while len(self.notServed) > 0:
            # pick a random customer
            cust = randomGen.choice(self.notServed)
            # keep track of routes in which customers could be inserted
            potentialRoutes = self.routes_2.copy()
            inserted = False
            while len(potentialRoutes) > 0:
                # pick a random route
                randomRoute = randomGen.choice(potentialRoutes)
                afterInsertion = randomRoute.greedyInsert(
                    cust.deliveryLoc, cust.deliveryLoc.demand)
                if afterInsertion is None:
                    # insertion not feasible, remove route from potential routes
                    potentialRoutes.remove(randomRoute)
                else:
                    # insertion feasible, update routes and break from while loop
                    inserted = True
                    afterInsertion.customers = randomRoute.customers
                    afterInsertion.customers.append(cust)
                    self.routes_2.remove(randomRoute)
                    self.routes_2.append(afterInsertion)
                    break

            # if we were not able to insert, create a new route
            if not inserted:
                # create a new route with the customer
                sat = randomGen.choice(self.problem.satellites)
                locList = [sat, cust.deliveryLoc, sat]
                newRoute = Route(locList, self.problem, False, [cust.deliveryLoc.demand])
                newRoute.customers.append(cust)
                self.routes_2.append(newRoute)
            # update the lists with served and notServed customers
            self.served.append(cust)
            self.notServed.remove(cust)

    def greedyRepair(self, randomGen):
        """
        Method that applies a greedy repair method to construct second echelon routes.

        Parameters
        ----------
        randomGen : Random
            Used to generate random numbers for the repair process.

        Returns
        -------
        None.
        """
        # Iterate over unserved customers
        for cust in self.notServed:
            # Keep track of potential routes for insertion
            potentialRoutes = self.routes_2.copy()
            inserted = False

            while len(potentialRoutes) > 0:
                # Pick a random route
                randomRoute = randomGen.choice(potentialRoutes)
                afterInsertion = randomRoute.greedyInsert(
                    cust.deliveryLoc, cust.deliveryLoc.demand)

                if afterInsertion is None:
                    # Insertion not feasible, remove route from potential routes
                    potentialRoutes.remove(randomRoute)
                else:
                    # Insertion feasible, update routes and break from the loop
                    inserted = True
                    afterInsertion.customers = randomRoute.customers
                    afterInsertion.customers.append(cust)
                    self.routes_2.remove(randomRoute)
                    self.routes_2.append(afterInsertion)
                    break

            # If we were not able to insert, create a new route
            if not inserted:
                # Create a new route with the customer
                sat = randomGen.choice(self.problem.satellites)
                locList = [sat, cust.deliveryLoc, sat]
                newRoute = Route(locList, self.problem, False, [cust.deliveryLoc.demand])
                newRoute.customers.append(cust)
                self.routes_2.append(newRoute)

            # Update the lists with served and notServed customers
            self.served.append(cust)
            self.notServed.remove(cust)