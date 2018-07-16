import numpy as np
import torch
from torch.autograd import Variable


USING_GPU = False


def main():
    print 'hello deep world...'





def numpy_representation(N,D):

    # Using numpy (and thus...CPU)
    np.random.seed(0)
    x = np.random.randn(N, D)
    y = np.random.randn(N, D)
    z = np.random.randn(N, D)

    a = x * y
    b = a + z
    c = np.sum(b)


    grad_c = 1.0
    grad_b = grad_c * np.ones_like(x)
    grad_a = grad_b.copy()
    grad_z = grad_b.copy()
    grad_x = grad_a * y
    grad_y = grad_a * x

    print '\nNumpy Output...'
    print grad_x
    print grad_y
    print grad_z

    print '\nokay, used numpy.'


def torch_framework(N,D):

    # using pytorch libraries (and thus...GPU capability)
    x = Variable(torch.randn(N, D), requires_grad=True)
    y = Variable(torch.randn(N, D), requires_grad=True)
    z = Variable(torch.randn(N, D), requires_grad=True)

    if USING_GPU:
        x = Variable(torch.randn(N, D).cuda(), requires_grad=True)
        y = Variable(torch.randn(N, D).cuda(), requires_grad=True)
        z = Variable(torch.randn(N, D).cuda(), requires_grad=True)


    a = x * y
    b = a + z
    c = torch.sum(b)

    c.backward()

    print '\nPyTorch Output...'
    print(x.grad.data)
    print(y.grad.data)
    print(z.grad.data)

    print '\nokay, used pytorch.'


def fc_example():
    N,D,H = 64, 1000, 100

    # initialize variables
    x = Variable(torch.randn(N, D), requires_grad=True)
    y = Variable(torch.randn(N, D), requires_grad=True)
    w1 = Variable(torch.randn(D, H), requires_grad=True)
    w2 = Variable(torch.randn(H, D), requires_grad=True)




if __name__ == '__main__':
    N,D = 3,4

    main()
    #numpy_representation(N,D)
    #torch_framework(N,D)