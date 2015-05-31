import numpy as np
import collections
import utility as nn

class RNN:
    def __init__(self, wvec_dim, output_dim, num_words, mb_size=30, rho=1e-4):
        self.wvec_dim = wvec_dim
        self.output_dim = output_dim
        self.num_words = num_words
        self.mb_size = mb_size
        self.default_vec = lambda: np.zeros((wvec_dim,))
        self.rho = rho

    def init_params(self):
        np.random.seed(12341)

        # Word vectors
        self.L = np.random.randn(self.wvec_dim, self.num_words) * 0.01

        # Hidden layer parameters
        self.W = np.random.randn(self.wvec_dim, 2 * self.wvec_dim) * 0.01
        self.b = np.zeros(self.wvec_dim)

        # Softmax weights
        self.Ws = np.random.randn(self.output_dim, self.wvec_dim) * 0.01
        self.bs = np.zeros(self.output_dim)

        self.stack = [self.L, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dW = np.empty(self.W.shape)
        self.db = np.empty(self.b.shape)
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty(self.bs.shape)

    def cost_and_grad(self, mbdata, test=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns
           cost, correctArray, guessArray, total

        """
        cost = 0.0
        correct = []
        guess = []

        self.L, self.W, self.b, self.Ws, self.bs = self.stack

        # Zero gradients
        self.dW[:] = 0
        self.db[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.default_vec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c, predict = self.forward_prop(tree)
            cost += c
            guess.append(predict)
            correct.append(tree.label)
        if test:
            return (1. / len(mbdata)) * cost, correct, guess

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.back_prop(tree)

        # scale cost and grad by mb size
        scale = (1. / self.mb_size)
        for v in self.dL.itervalues():
            v *= scale

        # Add L2 Regularization
        cost += (self.rho / 2) * np.sum(self.W ** 2)
        cost += (self.rho / 2) * np.sum(self.Ws ** 2)

        return scale * cost, [self.dL, scale * (self.dW + self.rho * self.W), scale * self.db,
                              scale * (self.dWs + self.rho * self.Ws), scale * self.dbs]

    def forward_prop(self, tree):
        cost = self.forward_prop_node(tree.root)
        theta = self.Ws.dot(tree.root.hActs1) + self.bs
        tree.probs = np.exp(theta - np.max(theta))
        tree.probs /= np.sum(tree.probs)

        cost += -np.log(tree.probs[tree.label])
        return cost, np.argmax(tree.probs)

    def forward_prop_node(self, node, nonlinearity=nn.relu):
        cost = 0.0
        if node.isLeaf:
            node.hActs1 = self.L[:, node.word].copy()
        else:
            cost_left = self.forward_prop_node(node.left)
            cost_right = self.forward_prop_node(node.right)
            cost += (cost_left + cost_right)
            node.hActs1 = self.W.dot(np.hstack((node.left.hActs1, node.right.hActs1))) + self.b
            nonlinearity(node.hActs1)
        return cost

    def back_prop(self, tree):
        deltas = tree.probs.copy()
        deltas[tree.label] -= 1.0
        self.dWs += np.outer(deltas, tree.root.hActs1)
        self.dbs += deltas
        deltas = deltas.dot(self.Ws)
        self.back_prob_node(self, tree.root, deltas)
        pass

    def back_prop_node(self, node, error=None):
        if error is not None: deltas += error
        if node.isLeaf:
            self.dL[node.word] += deltas
        else:
            deltas *= (node.hActs1 > 0)
            self.dW += np.outer(deltas, np.hstack((node.left.hActs1, node.right.hActs1)))
            self.db += deltas
            deltas = deltas.dot(self.W)
            self.backProp(node.left, deltas[:self.wvecDim])
            self.backProp(node.right, deltas[self.wvecDim:])

    def updateParams(self, scale, update, log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P, dP in zip(self.stack[1:], update[1:]):
                pRMS = np.sqrt(np.mean(P ** 2))
                dpRMS = np.sqrt(np.mean((scale * dP) ** 2))
                print "weight rms=%f -- update rms=%f" % (pRMS, dpRMS)

        self.stack[1:] = [P + scale * dP for P, dP in zip(self.stack[1:], update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:, j] += scale * dL[j]

    def toFile(self, fid):
        import cPickle as pickle

        pickle.dump(self.stack, fid)

    def fromFile(self, fid):
        import cPickle as pickle

        self.stack = pickle.load(fid)

    def check_grad(self, data, epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dW..."
        for W, dW in zip(self.stack[1:], grad[1:]):
            W = W[..., None]  # add dimension since bias is flat
            dW = dW[..., None]
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i, j] += epsilon
                    costP, _ = self.costAndGrad(data)
                    W[i, j] -= epsilon
                    numGrad = (costP - cost) / epsilon
                    err = np.abs(dW[i, j] - numGrad)
                    err1 += err
                    count += 1

        if 0.001 > err1 / count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1 / count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i, j] += epsilon
                costP, _ = self.costAndGrad(data)
                L[i, j] -= epsilon
                numGrad = (costP - cost) / epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2 += err
                count += 1

        if 0.001 > err2 / count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2 / count)


if __name__ == '__main__':
    import tree as treeM

    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    rnn = RNN(wvecDim, outputDim, numW, mbSize=4)
    rnn.initParams()

    mbData = train[:4]

    print "Numerical gradient check..."
    rnn.check_grad(mbData)
