import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "This should compute the dot product of the stored weight vector and the given input, returning an nn.DotProduct object."
        return nn.DotProduct(x, self.w)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 if non negative or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        while True:
            converged = True  # assume converged until shown otherwise
            for x, y in dataset.iterate_once(batch_size):
                if (self.get_prediction(x) != nn.as_scalar(y)):
                    converged = False #shown otherwise (when prediction doesnt match expected)
                    self.w.update(x, nn.as_scalar(y)) #update weights
            if converged:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # 1 hidden layer works
        hidden_layer_dim = 40
        self.m1 = nn.Parameter(1,hidden_layer_dim)
        self.b1 = nn.Parameter(1,hidden_layer_dim)
        self.m2 = nn.Parameter(hidden_layer_dim,1)
        self.b2 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.m1),self.b1))
        layer2 = nn.AddBias(nn.Linear(layer1,self.m2),self.b2)
        return layer2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)
    
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = 0.01
        batch_size = 100
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2  = nn.gradients(loss, [self.m1, self.b1, self.m2, self.b2])
                #update weights and biases
                self.m1.update(grad_wrt_m1, -alpha)
                self.b1.update(grad_wrt_b1, -alpha)
                self.m2.update(grad_wrt_m2, -alpha)
                self.b2.update(grad_wrt_b2, -alpha)
            loss = self.get_loss(x,y)
            # was having issue with being < 0.02, but grader said it was higher
            if nn.as_scalar(loss) < 0.015:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_dims = [100,50,20]
        img_dim = 28*28
        self.m1 = nn.Parameter(img_dim,hidden_layer_dims[0])
        self.b1 = nn.Parameter(1,hidden_layer_dims[0])
        #m2 and b2 are 2nd hiddne layer
        self.m2 = nn.Parameter(hidden_layer_dims[0],hidden_layer_dims[1])
        self.b2 = nn.Parameter(1,hidden_layer_dims[1])
        #m3 and b3 are 3rd hidden layer
        self.m3 = nn.Parameter(hidden_layer_dims[1],hidden_layer_dims[2])
        self.b3 = nn.Parameter(1,hidden_layer_dims[2])
        #m4 and b4 are output layer
        self.m4 = nn.Parameter(hidden_layer_dims[2],10)
        self.b4 = nn.Parameter(1,10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(x,self.m1),self.b1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1,self.m2),self.b2))
        layer3 = nn.ReLU(nn.AddBias(nn.Linear(layer2,self.m3),self.b3))
        layer4 = nn.AddBias(nn.Linear(layer3,self.m4),self.b4)
        return layer4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = 0.05
        batch_size = 100
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                grad_wrt_m1, grad_wrt_b1, grad_wrt_m2, grad_wrt_b2, grad_wrt_m3, grad_wrt_b3, grad_wrt_m4, grad_wrt_b4 = nn.gradients(loss, [self.m1, self.b1, self.m2, self.b2, self.m3, self.b3, self.m4, self.b4])
                #update weights and biases
                self.m1.update(grad_wrt_m1, -alpha)
                self.b1.update(grad_wrt_b1, -alpha)
                self.m2.update(grad_wrt_m2, -alpha)
                self.b2.update(grad_wrt_b2, -alpha)
                self.m3.update(grad_wrt_m3, -alpha)
                self.b3.update(grad_wrt_b3, -alpha)
                self.m4.update(grad_wrt_m4, -alpha)
                self.b4.update(grad_wrt_b4, -alpha)
            accuracy = dataset.get_validation_accuracy()
            # similar issue to Q2, so added 0.04 to threshold (the largest difference observed)
            if accuracy > 0.974:
                break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        hidden_layer_dim = 50

        self.W1 = nn.Parameter(self.num_chars, hidden_layer_dim)
        self.b1 = nn.Parameter(1, hidden_layer_dim)

        self.Wh = nn.Parameter(hidden_layer_dim, hidden_layer_dim)

        self.W2 = nn.Parameter(hidden_layer_dim, 5)
        self.b2 = nn.Parameter(1, 5)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # first function h=f(x0)=xoW1+b1
        h = nn.AddBias(nn.Linear(xs[0], self.W1), self.b1)
        # zi = xiW+hiWh, skip first element of xs as it was covered in h
        for x in xs[1:]:
            h = nn.ReLU(nn.Add(nn.Linear(x, self.W1), nn.Linear(h, self.Wh)))
        return nn.AddBias(nn.Linear(h, self.W2), self.b2)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        alpha = 0.04
        batch_size = 100
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                grad_W, grad_b, grad_hidden, grad_W_out, grad_b_out = nn.gradients(loss, [self.W1, self.b1, self.Wh, self.W2, self.b2])
                #update weights and biases
                self.W1.update(grad_W, -alpha)
                self.b1.update(grad_b, -alpha)
                self.Wh.update(grad_hidden, -alpha)
                self.W2.update(grad_W_out, -alpha)
                self.b2.update(grad_b_out, -alpha)

            accuracy = dataset.get_validation_accuracy()
            # similar issue to Q2, so added 0.04 to threshold (the largest difference observed)
            if accuracy > 0.86:
                break