import torch


class InstaNet(torch.nn.Module):
    def __init__(self, n_input_neurons, n_hidden_neurons):
        super(InstaNet, self).__init__()
        # self.fc1 = torch.nn.Linear(X_train.shape[1], 1)
        self.fc1 = torch.nn.Linear(n_input_neurons, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 3)

        self.sm = torch.nn.Softmax(dim=1)
        # !!! anything else ?

    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)

        # !!! anything else ?

        return x

    def inference(self, x):  #!!! where inference is used?
        x = self.forward(x)
        x = self.sm(x)
        return x