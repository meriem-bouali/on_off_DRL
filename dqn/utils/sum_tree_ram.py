import numpy as np

## source: https://github.com/pythonlessons/Reinforcement_Learning/blob/master/05_CartPole-reinforcement-learning_PER_D3QN/PER.py
## source: https://www.youtube.com/watch?v=1BfTrfiaWXg&t=838s
## source: https://pylessons.com/CartPole-PER


class SumTree:
    """
    SumTree is a binary tree where each parent's value is the sum of its children.
    This data structure is used for efficient priority sampling.
    The leaf nodes (final nodes) of `self.tree` contain the priority value of an experience,
    and the `self.data` array points to the experiences stored in the leaves.

    """

    def __init__(self, capacity: int):
        """
        Initialize the SumTree.

        Args:
            capacity (int):The maximum number of experiences the SumTree can store.
        """
        self.capacity = capacity  # the maximum number of experiences SumTree can store.

        # Initialize the tree with all nodes set to 0. As it's a binary tree, its size is 2 * capacity - 1.
        self.tree = np.zeros(2 * capacity - 1)

        # Initialize the data array with all values set to 0. This array will store experiences, so its size is equal to capacity.
        self.data = np.empty(capacity, dtype=object)

        self.data_pointer = 0  # Points to the next position/index to write in the data array.
        self.size = 0  # Current number of stored experiences
        self.max_priority_index = capacity - 1  # Index of current max priority leaf
        self.min_priority_index = capacity - 1  # Index of current min priority leaf

    def len(self):
        """Returns the number of stored experiences in the SummTree"""
        print(np.count_nonzero(self.data))
        return np.count_nonzero(self.data)

    def to_dict(self):
        """
        Convert the instance of SumTree to a dictionary.

        This method is useful for saving the state of the SumTree instance to a file.
        The dictionary representation allows for easy serialization and storage of
        the SumTree's state, which can later be used to reconstruct the instance.

        Returns:
            dict: A dictionary representation of the SumTree instance containing all essential attributes.

        """
        return {
            "capacity": self.capacity,  # Maximum number of experiences the SumTree can store.
            "tree": self.tree,  #  structure of the SumTree that holds data priorities.
            "data": self.data,  # The data stored in the SumTree (e.g., experience tuples).
            "data_pointer": self.data_pointer,  # Points to the next position/index to write in the data array.
            "size": self.size,  # Current number of stored experiences
            "max_priority_index": self.max_priority_index,  # Index of current max priority leaf
            "min_priority_index": self.min_priority_index,  # Index of current min priority leaf
        }

    def from_dict(self, sumTree_dict: dict):
        """
        Convert a dictionary back to an instance of SumTree.

        This method reconstructs a SumTree instance from a dictionary representation.
        The dictionary, typically loaded from a file, contains all necessary attributes
        to restore the state of the SumTree.

        Args:
            sumTree_dict (dict): A dictionary containing the attributes of the SumTree instance.
        """
        self.capacity = sumTree_dict["capacity"]  # Maximum number of experiences the SumTree can store.
        self.tree = np.array(sumTree_dict["tree"]).copy()  #  structure of the SumTree that holds data priorities.
        self.data = np.array(sumTree_dict["data"]).copy()  # The data stored in the SumTree (e.g., experience tuples).
        self.data_pointer = sumTree_dict["data_pointer"]  # Points to the next position/index to write in the data array.
        self.size = sumTree_dict["size"]  # Current number of stored experiences
        self.max_priority_index = sumTree_dict["max_priority_index"]  # Index of current max priority leaf
        self.min_priority_index = sumTree_dict["min_priority_index"]  # Index of current min priority leaf

    def update(self, tree_index: int, priority: float):
        """
        Update the priority of a leaf node and propagate the change through the tree.

        Args:
            tree_index (int): Index of the element in the tree to update.
            priority (float): The new priority value.
        """
        max_p, min_p = self.tree[self.max_priority_index], self.tree[self.min_priority_index]

        # Calculate the change in priority
        change = priority - self.tree[tree_index]  # Change = new priority score - former priority score

        self.tree[tree_index] = priority  # Update the priority.

        # Update max priority indices
        if priority >= max_p:
            self.max_priority_index = tree_index
        elif tree_index == self.max_priority_index:
            self.max_priority_index = np.argmax(self.tree[self.capacity - 1 : self.capacity + self.size - 1]) + self.capacity - 1

        # Update min priority indices
        if priority <= min_p:
            self.min_priority_index = tree_index
        elif tree_index == self.min_priority_index:
            self.min_priority_index = np.argmin(self.tree[self.capacity - 1 : self.capacity + self.size - 1]) + self.capacity - 1

        # propagate the change through tree
        while not tree_index == 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def add(self, priority: float, data: tuple):
        """
        Add a new data point to the SumTree with a given priority.
        The experience-priority score is added to the tree leaf, and the experience-data is stored in the data array.

        Args:
            priority (float): Priority value of the new element.
            data (tuple): The data to store.
        """
        # Calculate the tree index where the new experience (priority) will be put.
        tree_index = self.data_pointer + self.capacity - 1

        # Add the experience (data) to the data array.
        self.data[self.data_pointer] = data  # add experience to data_aray

        # Update data_pointer for adding the next new element.
        # If we exceed the capacity, we go back to first index
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        # increment number of stored experiences , ensuring it doesn't exceed capacity.
        self.size = min([self.size + 1, self.capacity])

        # Update the priority on the leaf node.
        self.update(tree_index, priority)

    def get_leaf(self, v):
        """
        Get the leaf_index, priority value of that leaf and experience associated with that leaf index
        """
        parent_index = 0  # Start from the root of the tree

        # Traverse the tree to find the appropriate leaf node.
        while True:
            left_child_index = 2 * parent_index + 1  # Get the left child index
            right_child_index = left_child_index + 1  # Get the right child index

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # downward search, always search for a higher priority node
            # Traverse left or right depending on the value of `v`
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1  # Get the index of the data in data array

        return leaf_index, self.tree[leaf_index], self.data[data_index]  # Return the leaf index, priority, and data

    @property
    def total_priority(self) -> float:
        """Get the total priority value stored in the tree."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Get the maximum priority value stored in the tree."""
        return self.tree[self.max_priority_index]

    @property
    def min_priority(self) -> float:
        """Get the minimum priority value stored in the tree."""
        return self.tree[self.min_priority_index]
