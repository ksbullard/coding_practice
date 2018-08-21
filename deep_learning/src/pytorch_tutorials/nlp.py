import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def get_familiar():
    t0 = time.time()
    print '\nBasic Pytorch Data Structures....'

    # torch.tensor(data) creates a torch.Tensor object with the given data.
    V_data = [1., 2., 3.]
    V = torch.tensor(V_data)
    print(V)

    # Creates a matrix
    M_data = [[1., 2., 3.], [4., 5., 6]]
    M = torch.tensor(M_data)
    print(M)

    # Create a 3D tensor of size 2x2x2.
    T_data = [[[1., 2.], [3., 4.]],
              [[5., 6.], [7., 8.]]]
    T = torch.tensor(T_data)
    print(T)


    print '\n\nOperations with Tensors....'
    # operations with tensors
    x = torch.tensor([1., 2., 3.])
    y = torch.tensor([4., 5., 6.])
    z = x + y
    print(z)

    # By default, it concatenates along the first axis (concatenates rows)
    x_1 = torch.randn(2, 5)
    y_1 = torch.randn(3, 5)
    z_1 = torch.cat([x_1, y_1])
    print(z_1)

    # Concatenate columns:
    x_2 = torch.randn(2, 3)
    y_2 = torch.randn(2, 5)
    # second arg specifies which axis to concat along
    z_2 = torch.cat([x_2, y_2], 1)
    print(z_2)

    # If your tensors are not compatible, torch will complain.  Uncomment to see the error
    # torch.cat([x_1, x_2])


    print '\n\nReshaping Tensors....'
    x = torch.randn(2, 3, 4)
    print(x)
    print(x.view(2, 12))  # Reshape to 2 rows, 12 columns
    # Same as above.  If one of the dimensions is -1, its size can be inferred
    print(x.view(2, -1))



    print '\n\nComputation Graphs and Automatic Differentiation....'
    # Tensor factory methods have a ``requires_grad`` flag
    x = torch.tensor([1., 2., 3], requires_grad=True)

    # With requires_grad=True, you can still do all the operations you previously
    # could
    y = torch.tensor([4., 5., 6], requires_grad=True)
    z = x + y
    print(z)

    # BUT z knows something extra.
    print(z.grad_fn)

    # Lets sum up all the entries in z
    s = z.sum()
    print(s)
    print(s.grad_fn)

    # calling .backward() on any variable will run backprop, starting from it.
    s.backward()
    print(x.grad)



    print '\n\nTesting Knowledge of Basic DL Concepts....'
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    # By default, user created Tensors have ``requires_grad=False``
    print(x.requires_grad, y.requires_grad)
    z = x + y
    # So you can't backprop through z
    print(z.grad_fn)

    # ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
    # flag in-place. The input flag defaults to ``True`` if not given.
    x = x.requires_grad_()
    y = y.requires_grad_()
    # z contains enough information to compute gradients, as we saw above
    z = x + y
    print(z.grad_fn)
    # If any input to an operation has ``requires_grad=True``, so will the output
    print(z.requires_grad)

    # Now z has the computation history that relates itself to x and y
    # Can we just take its values, and **detach** it from its history?
    new_z = z.detach()

    # ... does new_z have information to backprop to x and y?
    # NO!
    print(new_z.grad_fn)
    # And how could it? ``z.detach()`` returns a tensor that shares the same storage
    # as ``z``, but with the computation history forgotten. It doesn't know anything
    # about how it was computed.
    # In essence, we have broken the Tensor away from its past history


    # Can also stop autograd from tracking history on Tensors by using 'with torch.no_grad()' code block
    print(x.requires_grad)
    print((x ** 2).requires_grad)

    with torch.no_grad():
        print((x ** 2).requires_grad)

    t1 = time.time()
    print '\nFunction Running Time: ' + str(t1 - t0) + '\n\n'


def basic_model_training():
    # data is 2x5.  A maps from 5 to 3... can we map "data" under A?
    data = torch.randn(2, 5)
    print(data)

    # weight matrix maps from R^5 to R^3, parameters A, b
    lin = nn.Linear(5, 3)  #lin: Linear(in_features=5, out_features=3, bias=True)
    lin_data = lin(data)

    print(lin_data)  # yes

    # In pytorch, most non-linearities are in torch.functional (we have it imported as F)
    # Note that non-linearites typically don't have parameters like affine maps do.
    # That is, they don't have weights that are updated during training.
    #data = torch.randn(2, 2)
    print(F.relu(lin_data))



    # Softmax is also in torch.nn.functional
    print '\n\nTesting softmax functionality....'
    data = torch.randn(5)
    print(data)
    print(F.softmax(data, dim=0))
    print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
    print(F.log_softmax(data, dim=0))  # theres also log_softmax


    pass



def use_word_embeddings():
    num_embedding_dimensions = 5
    word_to_index = {'hello': 0, 'world': 1}
    embeds = nn.Embedding(len(word_to_index), num_embedding_dimensions)

    lookup_tensor = torch.tensor([word_to_index['hello']], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print hello_embed


    pass


def pretrain_word_embeddings():
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()


    # By deriving a set from `raw_text`, we deduplicate the array
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []
    for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
        context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))
    print data[:5]

    # construct model and train
    make_context_vector(data[0][0], word_to_ix)  # example



def make_context_vector(context, word_to_ix):
    context_idxs = [word_to_ix[w] for w in context]
    return torch.tensor(context_idxs, dtype=torch.long)



class CBOW(nn.Module):  # Continuous Bag of Words
    def __init__(self):
        super(CBOW, self).__init__()




    def forward(self, inputs):
        pass





def train_ngram_model():
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10

    # Will use Shakespeare Sonnet 2
    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old, 
    And see thy blood warm when thou feel'st it cold.""".split()

    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]

    # print the first 3, just so you can see what they look like
    print(trigrams[:3])

    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    losses = []
    loss_function = nn.NLLLoss()
    language_model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(language_model.parameters(), lr=0.001)


    # train language model
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in trigrams:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            language_model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next words
            log_probs = language_model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!

    pass



class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # layers in NN
        # in_features: size of each input sample
        # out_features: size of each output sample
        # bias: if set to False, the layer will not learn an additive bias.
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)  #lin: Linear(in_features=cs*ed, out_features=128, bias=True)
        self.linear2 = nn.Linear(128, vocab_size)  #lin: Linear(in_features=128, out_features=vs, bias=True)


    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs



class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)  #lin: Linear(in_features=26, out_features=2, bias=True)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)



    def probabilities(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.softmax(self.linear(bow_vec), dim=1)




def train_bag_of_words_model():
    data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
            ("Give it to me".split(), "ENGLISH"),
            ("No creo que sea una buena idea".split(), "SPANISH"),
            ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

    test_data = [("Yo creo que si".split(), "SPANISH"),
                 ("it is lost on me".split(), "ENGLISH")]

    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    word_to_index = {}
    for sentence, _ in data + test_data:
        for word in sentence:
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)

    print 'Language Indices: ' + str(word_to_index) + '\n'

    VOCAB_SIZE = len(word_to_index)
    NUM_LABELS = 2

    language_model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

    # the model knows its parameters.  The first output below is A, the second is b.
    # Whenever you assign a component to a class variable in the __init__ function
    # of a module, which was done with the line
    # self.linear = nn.Linear(...)
    # Then through some Python magic from the PyTorch devs, your module
    # (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
    for param in language_model.parameters():
        print(param)

    # To run the model, pass in a BoW vector
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    i = 0
    print '\n\nTraining Example...'
    with torch.no_grad():
        sample = data[i]
        bow_vector = create_bow_vector(sample[0], word_to_index)
        log_probs = language_model(bow_vector)
        print(log_probs)


    label_to_index = {"SPANISH": 0, "ENGLISH": 1}

    # Run on test data before we train, just to see a before-and-after
    print '\n\nTest Dataset...'
    with torch.no_grad():
        for instance, label in test_data:
            bow_vec = create_bow_vector(instance, word_to_index)
            log_probs = language_model(bow_vec)
            print(log_probs)

    print '\n\n-------------------------TRAINING PHASE-------------------------\n'

    # Print the matrix column corresponding to "creo" (before)
    print 'Pretest: Weights for \'creo\' token (BEFORE TRAINING)'
    print str(next(language_model.parameters())[:, word_to_index["creo"]]) + '\n\n'


    # Objective Function and Training Algorithm
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(language_model.parameters(), lr=0.1)


    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(100):
        for instance, label in data:
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            language_model.zero_grad()

            # Step 2. Make our BOW vector and also we must wrap the target in a
            # Tensor as an integer. For example, if the target is SPANISH, then
            # we wrap the integer 0. The loss function then knows that the 0th
            # element of the log probabilities is the log probability
            # corresponding to SPANISH
            bow_vec = create_bow_vector(instance, word_to_index)
            target = create_int_label(label, label_to_index)

            # Step 3. Run our forward pass.
            log_probs = language_model(bow_vec)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probs, target)
            print '\t' + str((epoch, loss.item()))

            loss.backward()
            optimizer.step()


    # Test model trained
    print '\n\n-------------------------TESTING PHASE-------------------------\n'
    with torch.no_grad():
        for instance, label in test_data:
            bow_vec = create_bow_vector(instance, word_to_index)
            log_probs = language_model(bow_vec)
            probs = language_model.probabilities(bow_vec)
            print '\t' + str(probs)

    # Print the matrix column corresponding to "creo" (after)
    print '\n\nPosttest: Weights for \'creo\' token (AFTER TRAINING)'
    print(next(language_model.parameters())[:, word_to_index["creo"]])

    pass



def create_bow_vector(sentence, word_to_index):
    bow_vector = torch.zeros(len(word_to_index))
    for word in sentence:
        bow_vector[word_to_index[word]] += 1
    return bow_vector.view(1,-1)  #bow_vector


def create_int_label(label, label_to_index):
    return torch.LongTensor([label_to_index[label]])


def main():
    #get_familiar()
    #basic_model_training()

    #train_bag_of_words_model()
    use_word_embeddings()
    train_ngram_model()




if __name__ == '__main__':
    main()

