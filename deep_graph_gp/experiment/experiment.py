import numpy as np

def train(gp, data, num_epochs=100, eval_epochs=None):
    """
    Train on given data for some specified number of epochs
    - Save the states and do evaluation on the specified eval_epochs
    
    This should return
        train_results (dict)
            eval_epochs (list)
            states (list) list of gp.parameters at eval_epochs

    """

    

    return

def split_data(g, split):
    """

    """
    # Return 3 objects
    # train_data, test_data and val_data


    # What should the data set object be

    return

def evaluate(gp, states, data):

    return

class Experiment():
    def __init__(self, gp, graph, n_trials=25, split=[0.5, 0.1, 0.5], n_epochs=100):
        self.gp = gp
        self.graph = graph
        self.n_trials = n_trials
        self.split = split
        self.n_epochs = n_epochs

    def run(self):
        avg_accuracy = 0.

        all_train_results = []
        all_test_results  = []
        all_val_results   = []

        for i in range(self.n_trials):
            # Do data split (split the nodes into labeled and labeled nodes)
            train_data, val_data, test_data = split_data(self.graph)

            # Get training results
            train_results = train(self.gp, train_data, self.n_epochs)

            # Get testing results
            test_results = evaluate(self.gp, test_data)

            # Get validation results
            val_results  = evaluate(self.gp, val_data)

            all_train_results.append(train_results)
            all_test_results.append(test_results)
            all_val_results.append(val_results)

        average_train_results = self.average_results(all_train_results)
        average_test_results  = self.average_results(all_test_results)
        average_val_results   = self.average_results(all_val_results)

        return average_train_results, average_test_results, average_val_results

    def average_results(self, all_results):
        """
        Given a list of results over some n trials 
        Regardless whether it's train results or test results
        Compute the the average across all the trials
        """

        return