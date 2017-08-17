import search
from search import Problem

class Flight_Search(Problem):

    def actions(self, state):
    	# state[0] is the current city and state[1] is time 
	    flightList = []
	    for flight in flightDB:
		    if (state[0] == flight.start_city) and (state[1] <= flight.start_time):
			    flightList.append(flight)
	    return flightList

    def result(self, state, action):
    	# state is the current city and time 
    	# action is the flight
    	# new state is the next city and time
        nextState = []
        nextState.append(action.getStartCity())
        nextState.append(action.getStartTime())
        return nextState

    def goal_test(self, state):
        if (state[0] == self.goal[0]):
            return(state[1] <= self.goal[1])

    def value(self, state):
        pass

class Flight: 
    def __init__(self, start_city, start_time, end_city, end_time): 
        self.start_city = start_city 
        self.start_time = start_time 
        self.end_city = end_city 
        self.end_time = end_time

    def __str__(self): 
        return str((self.start_city, self.start_time))+' -> '+ str((self.end_city, self.end_time)) 

    __repr__ = __str__

    def getStartCity(self):
        return self.start_city

    def getStartTime(self):
        return self.start_time
    

    def flightMatch(self, start_city, start_time):
        for flight in flightDB:
            if (flight.start_city == start_city) and (flight.start_time >= start_time):
                return True
        return False

    def one_flight_search(self, search, flightList,itineraryList):
        for action in flightList:
                if(action == flightList[0]):
                    itineraryList.append(action)
                else:
                    print ('action flight:')
                    print(action)
                    search.state = search.result(search.state, action)
                    print ('next state:')
                    print(search.state)
                    if(search.goal_test(search.state)):
                        itineraryList.append(search.state)
                        print ('result:')
                        print itineraryList
                        return itineraryList
                    elif (self.flightMatch(search.state[0], search.state[1])): 
                        print ('action:')
                        print(action)
                        itineraryList.append(action)

    #def find_itinerary(self, start_city, start_time, end_city, deadline): 
    def find_itinerary(self): 
        search = Flight_Search(initial = [self.start_city, self.start_time], 
        	goal = [self.end_city, self.end_time])

        search.state = search.initial

        flightList = search.actions(search.state)
        print('flightList:')
        print(flightList)
        itineraryList = []
        while(not self.one_flight_search(search, flightList, itineraryList)):
        	search.state = search.result(search.state, action)
            flightList = search.actions(search.state)

        print('No itinery found')
        return []

flightDB = [Flight('Rome', 1, 'Paris', 4), 
            Flight('Rome', 3, 'Madrid', 5), 
            Flight('Rome', 5, 'Istanbul', 10), 
            Flight('Paris', 2, 'London', 4), 
            Flight('Paris', 5, 'Oslo', 7), 
            Flight('Paris', 5, 'Istanbul', 9), 
            Flight('Madrid', 7, 'Rabat', 10), 
            Flight('Madrid', 8, 'London', 10), 
            Flight('Istanbul', 10, 'Constantinople', 10)] 


def find_shortest_itinerary(start_city, end_city):
	pass
	'''
	Ben Bitdiddle wants to ﬁnd a way to get from Rome at time 1 to Istanbul 
	at the earliest time possible. He proposes to start with a deadline 
	argument (to find_itinerary of 1, and then increase it, one-by-one, 
	calling find_itinerary each time, until it successfully ﬁnds a path.

	return a list of (location,time) tuples representing the shortest path between the 
	two locations. You may assume that there is at least one path connecting the two locations.
	'''

def find_shortest_itinerary_reverse(start_city, end_city):
	pass
	'''
	Can you do it the other way round and start with some large deadline time? 
	As an additional challenge – program that
	'''

if __name__=='__main__':
    print ('starting flight search')
    #Flight_Search(object).actions(['Rome',1])
    #Flight_Search(object).result(['Rome',1], Flight('Rome', 3, 'Madrid', 5))
    #flight = Flight('Rome', 1, 'London', 10)
    flight = Flight('Rome', 1, 'Istanbul', 10)
    flight.find_itinerary()
