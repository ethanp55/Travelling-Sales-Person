#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
from BranchAndBoundState import *
from MatrixHandler import *
import heapq
import itertools
import copy

# Let n be the number of cities and b be the average number of sub problems a given problem can be expanded into for
#  the branch and bound algorithm
class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario(self, scenario):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()

		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	# Time complexity of O(n^2) since there are n recursive levels (we need to build a route with all n cities in it),
	#  and at each recursive level we need to iterate through any remaining cities we have and determine the best one
	#  to visit next (which could take n iterations)
	# Space complexity of O(1) since we don't create any copies of route or remaining_cities
	def recursive_greedy(self, route, remaining_cities):
		# Base case (if there are no more cities to visit, the route is complete)
		if len(remaining_cities) == 0:
			return route

		# Get the last city we visited
		curr_city = route[-1]
		# Initialize the next city and cost associated with going to the next city (we want to choose the city with the
		#  smallest travelling cost)
		min_cost = math.inf
		next_city = None

		# Iterate through any possible city we can visit and update min cost and next city as we go
		# Time complexity of O(n)
		for city in remaining_cities:
			if curr_city.costTo(city) < min_cost:
				min_cost = curr_city.costTo(city)
				next_city = city

		# If we can't get to another city (we hit a dead end), we're done
		if next_city is None:
			return None

		# Otherwise, add the next city to the route and remove it from the list of remaining cities to visit
		route.append(next_city)
		remaining_cities.remove(next_city)

		# Go to the next recursive level
		return self.recursive_greedy(route, remaining_cities)

	# Time complexity of O(n^3)
	# Space complexity of O(n)
	def greedy(self, time_allowance=60.0):
		# Initialize the results
		results = {}
		# Get the list of cities and set the number of solutions (count) to 0
		# Space complexity of O(n)
		cities = self._scenario.getCities()
		count = 0
		# We haven't found a solution yet
		foundTour = False
		bssf = None
		# Start
		start_time = time.time()

		# Iterate and keep trying a new starting city for the greedy algorithm until we either find a solution or run
		#  out of time
		# Time complexity of O(n^3) since we iterate n times through this loop and at each iteration we call
		#  recursive_greedy, which has a time complexity of O(n^2)
		# Space complexity of O(n) because at each iteration we have a route and list of remaining cities and once an
		#  iteration is complete those items are no longer stored
		for city in cities:
			# If we run out of time, stop
			if time.time() - start_time >= time_allowance:
				break

			# Get a copy of the list of cities and remove the starting city from it
			# Space complexity of O(n)
			remaining_cities = copy.copy(cities)
			remaining_cities.remove(city)
			route = [city]

			# Greedily try to find a route
			# Time complexity of O(n^2)
			route = self.recursive_greedy(route, remaining_cities)

			# If route is found, update bssf and see if it's valid
			if route is not None:
				bssf = TSPSolution(route)

				# If bssf is valid, we can successfully finish
				if bssf.cost < math.inf:
					foundTour = True
					count += 1
					break

		# Find the amount of time it took to run the greedy algorithm and report any results we found
		# Time and space complexity of O(1)
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	# Time and space complexity of O(n^2 * b^n)
	def branchAndBound(self, time_allowance=60.0):
		# Initialize the results
		results = {}
		# Get the list of cities and initialize the number of states, the max queue size, the number of solutions
		#  (count), and the number of pruned states
		cities = self._scenario.getCities()
		num_states = 1
		max_queue_size = 0
		count = 0
		pruned_states = 0
		# Use the first city as the start city
		start_city = cities[0]
		# Get the initial matrix and lower bound
		# Time and space complexity of O(n^2)
		initial_matrix_results = MatrixHandler.get_initial_matrix(cities)
		initial_matrix = initial_matrix_results[0]
		initial_lower_bound = initial_matrix_results[1]
		# Copy the list of cities to use for a list of remaining cities and remove the start state
		# Space complexity of O(n)
		initial_remaining_cities = copy.copy(cities)
		initial_remaining_cities.remove(start_city)
		# Make the start state and initialize the queue
		start_state = BranchAndBoundState(initial_matrix, initial_lower_bound, 0, start_city, [start_city], initial_remaining_cities)
		state_queue = [start_state]
		heapq.heapify(state_queue)
		# Start by running the greedy algorithm to get an initial bssf (let the greedy algorithm run for at most 10
		#  seconds)
		start_time = time.time()
		# Time complexity of O(n^3) and space complexity of O(n)
		greedy_results = self.greedy(10)
		bssf = greedy_results.get('soln')

		# Iterate until the state queue is empty or we run out of time
		# Time and space complexity of O(n^2 * b^n) since there are n cities, and each subproblem we create/expand will
		#  have an average of b new subproblems (so, worst case, there will be b^n states).  Each subproblem/state has an
		#  n by n matrix that must be reduced, which results in time and space complexities of O(n^2).
		while len(state_queue) != 0 and time.time() - start_time < time_allowance:
			# Update the max length of the queue
			queue_length = len(state_queue)
			max_queue_size = queue_length if queue_length > max_queue_size else max_queue_size

			# Get the next state from the queue
			curr_state = heapq.heappop(state_queue)

			# If the state's lower bound is larger than the bssf cost, prune it (don't even both expanding it)
			if curr_state.lower_bound >= bssf.cost:
				pruned_states += 1
				continue

			# Increment the depth and get a copy of the current remaining cities
			# Space complexity of O(b)
			depth = curr_state.depth + 1
			adjacent_cities = copy.copy(curr_state.remaining_cities)

			# Initialize a list that will hold the new states/subproblems that are created
			new_states = []

			# Iterate through each city we can visit
			# Time and space complexity of O(b * n^2) since we iterate through this loop b times, and at each iteration
			#  we call update_matrix, which has a time and space complexity of O(n^2)
			for city in adjacent_cities:
				# Get the new matrix and lower bound for the subproblem
				# Time and space complexity of O(n^2)
				new_matrix_results = MatrixHandler.update_matrix(curr_state, city)
				new_matrix = new_matrix_results[0]
				new_lower_bound = new_matrix_results[1]
				# Get and update the new route for the subproblem
				# Space complexity of O(n)
				new_route = copy.copy(curr_state.route)
				new_route.append(city)
				# Get and update the new list of remaining cities for the subproblem
				# Space complexity of O(n)
				new_remaining_cities = copy.copy(curr_state.remaining_cities)
				new_remaining_cities.remove(city)
				# Make a new state for the subproblem, add it to the list of new states, and increment the number of
				#  states created
				state = BranchAndBoundState(new_matrix, new_lower_bound, depth, city, new_route, new_remaining_cities)
				new_states.append(state)
				num_states += 1

			# Iterate through each new state/subproblem that we created
			# Time complexity of O(b) since we iterate through each new state/subproblem that we created
			for state in new_states:
				# If there are no more remaining cities (we've visited all of them), find a new bssf
				if len(state.remaining_cities) == 0:
					new_bssf = TSPSolution(state.route)

					# If the new bssf is better than the current bssf, update the current bssf
					if new_bssf.cost < bssf.cost:
						bssf = new_bssf
						count += 1

				# If the new state's lower bound shows potential, add it to the queue
				if state.lower_bound < bssf.cost:
					# Factor in lower bound and depth when adding a state to the queue
					state.comparator = state.lower_bound - state.depth
					heapq.heappush(state_queue, state)

				# Otherwise, don't add it (prune it)
				else:
					pruned_states += 1

		# Find the amount of time it took to run branch and bound algorithm and report any results we found
		# Time and space complexity of O(1)
		end_time = time.time()
		pruned_states += len(state_queue)
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_queue_size
		results['total'] = num_states
		results['pruned'] = pruned_states
		return results


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy(self, time_allowance=60.0):
		pass
