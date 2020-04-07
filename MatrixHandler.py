from TSPClasses import *
import numpy as np
import copy

# Static class used for matrix operations
class MatrixHandler(object):
    # Static dictionary that maps city objects to their index in a matrix
    city_to_index = {}

    # Function for finding the initial matrix
    # Time and space complexity of O(n^2) since we iterate and create an n by n matrix
    @staticmethod
    def get_initial_matrix(cities):
        # Initialize the matrix and index
        matrix = []
        i = 0

        # Iterate through every city.  These cities will be used for rows
        for row_city in cities:
            row = []

            # Iterate through every city again.  These cities will be used for columns
            for col_city in cities:
                # If row and column cities are equal, set the distance to infinity
                if row_city == col_city:
                    row.append(math.inf)

                # Otherwise set the distance to the cost of going from the row city to the column city
                else:
                    row.append(row_city.costTo(col_city))
            # Add the row to the matrix
            matrix.append(row)

            # Add the row city to the dictionary of cities to indices and increment the index
            MatrixHandler.city_to_index[row_city] = i
            i += 1

        # Convert the initial matrix to a numpy array (in order to make future matrix operations easier)
        initial_matrix = np.array(matrix)

        # Reduce the matrix and find the initial lower bound
        initial_lower_bound = MatrixHandler.reduce_matrix(initial_matrix)

        # Return the initial matrix and initial lower bound
        return initial_matrix, initial_lower_bound

    # Function for updating a matrix for a new branch and bound state.  This function will also call the reduce_matrix
    #  function
    # Time complexity of O(n^2) since we set the values in a given row and column to infinity, which involves a time
    #   complexity of O(n) for each.  We then call reduce_matrix, which has a time complexity of O(n^2)
    # Space complexity of O(n^2) since we create a copy of an n by n matrix and update it
    @staticmethod
    def update_matrix(state, next_city):
        # Find the indices of the "from" city and the "to" city
        from_city_index = MatrixHandler.city_to_index[state.city]
        to_city_index = MatrixHandler.city_to_index[next_city]

        # Copy the previous state's matrix
        new_matrix = copy.copy(state.matrix)

        # Set the values in the "from" city row to infinity and the values in the "to" city column to infinity.  Also
        #  set the value representing the "to" city going to the "from" city to infinity (since we don't want to go
        #  back to the city we came from and thus create a cycle)
        new_matrix[from_city_index, :] = math.inf
        new_matrix[:, to_city_index] = math.inf
        new_matrix[to_city_index, from_city_index] = math.inf

        # Reduce the new matrix
        additional_cost = MatrixHandler.reduce_matrix(new_matrix)

        # Get the cost of going from the "from" city to the "to" city from the previous state's matrix
        cost_to_next_city = state.matrix[from_city_index, to_city_index]

        # Calculate the lower bound for this new state
        new_lower_bound = cost_to_next_city + state.lower_bound + additional_cost

        # Return the new matrix and new lower bound
        return new_matrix, new_lower_bound

    # Function for reducing a matrix
    # Time complexity of O(n^2) since we iterate through each row of the matrix, and then at each iteration we might
    #  have to reduce the entire row.  We do the same for the columns once we're done with the rows
    # Space complexity of O(1) since we're only modifying the matrix and not creating a copy of it
    @staticmethod
    def reduce_matrix(matrix):
        # Get the number of rows and columns of the matrix.
        num_rows = matrix.shape[0]
        num_cols = matrix.shape[1]

        # Initialize the additional cost of reducing the matrix to 0
        additional_cost = 0

        # Iterate every row
        for i in range(0, num_rows):
            # Find the row min
            row_min = min(matrix[i, :])

            # Reduce the row if the row min isn't 0 or infinity.  This will increment the additional cost of reducing
            #  matrix
            if row_min != 0 and row_min != math.inf:
                matrix[i, :] -= row_min
                additional_cost += row_min

        # Iterate through every column
        for i in range(0, num_cols):
            # Find the column min
            col_min = min(matrix[:, i])

            # Reduce the column if the column min isn't 0 or infinity.  This will increment the additional cost of
            #  reducing the matrix
            if col_min != 0 and col_min != math.inf:
                matrix[:, i] -= col_min
                additional_cost += col_min

        # Return the additional cost
        return additional_cost
