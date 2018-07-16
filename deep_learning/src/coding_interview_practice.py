import math
import numpy as np
import matplotlib.pyplot as plt


""" Tester functions from Deep Learning course online. """

def softmax(x):

    """Compute softmax values for x"""
    e_x = np.array(x, dtype=np.float)
    for i in np.ndindex(e_x.shape):
        e_x[i] = math.exp(e_x[i])

    #e_x_sum = e_x.sum()
    softmax  = np.array(e_x)  #, dtype=np.float
    for i in np.ndindex(softmax.shape):
        softmax[i] /= float(e_x.sum())

    softmax2 = np.exp(x) / np.sum(np.exp(x), axis=0)
    return softmax2



def softmax_main():

    # Find softmax of given vectors
    scores = np.array([1.0, 2.0, 3.0])
    scores = np.array([3.0, 1.0, 0.2])
    probs = softmax(scores / 10)
    print "\nSoftmax (1d array) = " + str(probs)




    scores = np.array([[1, 2, 3, 6],
                       [2, 4, 5, 6],
                       [3, 8, 7, 6]])
    probs = softmax(scores)
    print "\nSoftmax (2d array) = \n" + str(probs)


    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
    probs = softmax(scores)




    # Plot softmax curves
    if probs is not None:
        plt.plot(x, probs.T, linewidth=2)
        plt.show()



def performOps(A):
    m = len(A)
    n = len(A[0])
    B = []
    for i in xrange(len(A)):
        B.append([0] * n)
        for j in xrange(len(A[i])):
            B[i][n - 1 - j] = A[i][j]
    return B



def diffPossible(A, B):
    # check edge cases
    if A is None or len(A) <= 1 or B is None:
        return 0

    # now check for target difference
    i = 0
    j = i + 1

    while (i < len(A) - 1 and j < len(A)):
        if (A[j] - A[i]) == B:
            return 1  # true
        elif (A[j] - A[i]) < B:
            j += 1
        else:
            if j == i + 1:
                j += 1
            i += 1
    return 0  # false


class SinglyLinkedList(object):
    def __init__(self, data=None, neighbors=None):
        self.__data = data
        self.__neighbors = neighbors
        if self.__neighbors is None:
            self.__neighbors = []

    def __str__(self):
        return self.__data


    def data(self):
        return self.__data

    def neighbors(self):
        return self.__neighbors


    def has_neighbors(self):
        return len(self.neighbors()) > 0


    def link_neighbor(self, n):
        self.__neighbors.append(n)


    def link_neighbors(self, nodes):
        self.__neighbors.extend(nodes)


def hasCycle(V, visited):
    if V in visited:
        visited.append(V)
        return True
    else:
        visited.append(V)
        if V.has_neighbors():
            return hasCycle(V.neighbors()[0], visited)
        else:
            return False


def factorial(n):
    if n > 1:  # recursive case
        return n * factorial(n-1)
    else:  # base case
        return 1



def code_interview_practice():
    '''
    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    #performOps(A)


    A = [1,2,3]
    b = 0
    print diffPossible(A, b)
    '''

    # create graph as linked list
    A = SinglyLinkedList('a')
    B = SinglyLinkedList('b')
    C = SinglyLinkedList('c')
    D = SinglyLinkedList('d')

    A.link_neighbor(B)
    B.link_neighbor(C)
    C.link_neighbor(D)
    #D.link_neighbor(B)

    visited = []
    has_cycle = hasCycle(A, visited)

    print 'has cycle: ' + str(has_cycle)
    print 'traversed path: ' + str([str(i) for i in visited])


    n = 5
    print 'factorial of ' + str(n) + ': ' + str(factorial(n))

    print '\n\nend of program.'




if __name__ == "__main__":
    #softmax_main()

    code_interview_practice()





