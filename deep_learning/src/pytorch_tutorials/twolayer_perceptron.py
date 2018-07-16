import time
import numpy as np
import torch



def numpy_array_version(num_time_steps):
    print'\n------------numpy version------------'

    t0 = time.time()
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)


    learning_rate = 1e-6
    print '\nLoss Values over Time...'
    for t in range(num_time_steps):
        # forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)


        # compute and print loss
        loss = np.square(y_pred - y).sum()
        print '\t' + str((t, loss))


        # backprop: compute gradient of loss with respect to w1 and w2
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)


        # update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        pass
    t1 = time.time()
    print '\n(Numpy Version) Time Taken: ' + str(t1 - t0)



def pytorch_tensor_version(num_time_steps):
    print'\n\n------------pytorch (manual) tensor version------------'

    t0 = time.time()
    dtype = torch.float
    device = torch.device('cpu')
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    learning_rate = 1e-6
    print '\nLoss Values over Time...'
    for t in range(num_time_steps):
        # Forward Pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)


        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
        print '\t' + str((t, loss))


        # Backward Pass: compute gradient of loss wrt w1 and w2
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        pass

    t1 = time.time()
    print '\n(Pytorch Manual Version) Time Taken: ' + str(t1 - t0)




def numpy_three_layer_network():
    print'\n------------numpy version------------'

    t0 = time.time()
    # N is batch size; D_in is input dimension;
    # H1 is first hidden dimension; H2 is second hidden dimension; D_out is output dimension.
    N, D_in, H1, H2, D_out = 30, 50, 10, 20, 4

    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)

    # Randomly initialize weight vectors
    w1 = np.random.randn(D_in, H1)
    w2 = np.random.randn(H1, H2)
    w3 = np.random.randn(H2, D_out)


    learning_rate = 1e-6
    num_epochs = 100
    print '\nLoss Values over Time...'
    for t in range(num_epochs):

        # forward pass: compute predicted y
        h1 = x.dot(w1)
        a1 = np.maximum(h1, 0)
        h2 = a1.dot(w2)
        a2 = np.maximum(h2, 0)
        h3 = a2.dot(w3)

        # compute and print loss
        y_pred = np.maximum(h3, 0)
        loss = np.square(y_pred - y).sum()
        print '\t' + str((t, loss))


        # backprop: compute gradient of loss with respect to w1, w2,  and w3
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w3 = a2.T.dot(grad_y_pred)
        grad_a2 = grad_y_pred.dot(w3.T)
        grad_a2[h2 < 0] = 0


        grad_w2 = a1.T.dot(grad_a2)
        grad_a1 = grad_a2.dot(w2.T)
        grad_a1[h1 < 0] = 0

        grad_w1 = x.T.dot(grad_a1)



        # update weights
        w1 -= (learning_rate * grad_w1)
        w2 -= (learning_rate * grad_w2)
        w3 -= (learning_rate * grad_w3)




    t1 = time.time()
    print '\n(Numpy Three-Layer Network) Time Taken: ' + str(t1 - t0)

    pass



def pytorch_fully_integrated_version(num_time_steps):
    print'\n\n------------pytorch (autograd) tensor version------------'

    t0 = time.time()
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold input and outputs.
    # Setting requires_grad=False indicates that we do not need to compute gradients
    # with respect to these Tensors during the backward pass.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Create random Tensors for weights.
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    print '\nLoss Values over Time...'
    for t in range(num_time_steps):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()

    t1 = time.time()
    print '\n(Pytorch Autograd Version) Time Taken: ' + str(t1 - t0)



def main():
    num_time_steps = 500  # 500

    #numpy_array_version(num_time_steps)
    #pytorch_tensor_version(num_time_steps)
    #pytorch_fully_integrated_version(num_time_steps)

    numpy_three_layer_network()




if __name__ == '__main__':
    main()