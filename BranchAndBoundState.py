# Class representing branch and bound states that are held in a priority queue as the branch and bound algorithm runs
class BranchAndBoundState:
    def __init__(self, matrix, lower_bound, depth, city, route, remaining_cities):
        # Each state has a reduced matrix, lower bound, comparator (for the priority queue), depth, city, route
        #  (representing the cities visited up to this point/state), and list of remaining cities that can be visited
        self.matrix = matrix
        self.lower_bound = lower_bound
        self.comparator = self.lower_bound
        self.depth = depth
        self.city = city
        self.route = route
        self.remaining_cities = remaining_cities

    # Use the state's comparator when inserting states into a priority queue
    def __lt__(self, other):
        return self.comparator < other.comparator
