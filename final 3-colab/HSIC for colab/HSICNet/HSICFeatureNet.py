import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from HSICNet.HSICNet import HSICNet
from sparsemax import Sparsemax

class HSICFeatureNet(HSICNet):
    '''
        The superclass for different neural networks for optimizing HSIC.
        Modifies to include feature-specific networks for feature transformation.
        Inputs:
        - input_dim: the input dimensions of the data
        - layers: List[int]: a list of integers, each indicating the number of neurons of a layer for each feature network
    '''
    def __init__(self, input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params):
        super(HSICFeatureNet, self).__init__(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)

        if act_fun_featlayer is None: act_fun_featlayer = nn.Sigmoid
        self.act_fun_featlayer = act_fun_featlayer


        # Create feature-wise networks (feature_net)
        self.feature_nets = nn.ModuleList([self._build_feature_net(feature_layers) for _ in range(input_dim)])

        # Initializing the sigmas based on the input (which is median heuristic)
        self.sigmas = nn.Parameter(sigma_init_X)
        self.sigma_y = nn.Parameter(sigma_init_Y)

    def _build_feature_net(self, layers):
        """
        Build a feature-specific network that transforms each feature.
        """
        net = []
        input_size = 1  # Each feature has a scalar input

        for layer_size in layers:
            net.append(nn.Linear(input_size, layer_size))
            if self.act_fun_featlayer == nn.Sigmoid:
                net.append(nn.BatchNorm1d(layer_size))
            net.append(self.act_fun_featlayer())  # Use activation by default
            input_size = layer_size

        net.append(nn.Linear(input_size, 1))  # Output one value per feature
        return nn.Sequential(*net)

    def forward(self, x):
        # Process each feature independently through its respective feature-specific network
        feature_transformed = [net(x[:, i:i+1]) for i, net in enumerate(self.feature_nets)]
        feature_transformed = torch.cat(feature_transformed, dim=1)  # Concatenate transformed features
        
        logit = super(HSICFeatureNet, self).forward(feature_transformed)

        return logit, feature_transformed

    def train_model(self, X, y, num_epochs=300, lr=1e-3, BATCH_SIZE = 100):
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(num_epochs):
            self.train()
            for inputs, outputs in train_loader:
                optimizer.zero_grad()

                s, sigmas, sigma_y, features_net = self(inputs)  # importance weights from Gumbel-Softmax
                loss = -self.hsic_loss_adaptive(features_net, outputs, s, sigmas, sigma_y)  # Minimize negative HSIC

                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


"""
Training a network with maximizing HSIC with Gumbel Sparsemax
"""

class HSICFeatureNetGumbelSparsemax(HSICFeatureNet):
    def __init__(self, input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_samples, temperature=10, **params):
        super(HSICFeatureNetGumbelSparsemax, self).__init__(input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)
        
        self.num_samples = num_samples
        self.temperature = temperature
        
    def forward(self, x):
        logits, feature_transformed = super(HSICFeatureNetGumbelSparsemax, self).forward(x)

        self.importance_weights = self.gumbel_sparsemax_sampling(logits, temperature=self.temperature, num_samples = self.num_samples)

        return self.importance_weights, self.sigmas, self.sigma_y, feature_transformed


"""
Training a network with maximizing HSIC with Gumbel Softmax
"""

class HSICFeatureNetGumbelSoftmax(HSICFeatureNet):
    def __init__(self, input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_samples, **params):
        super(HSICFeatureNetGumbelSoftmax, self).__init__(input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)
        
        self.num_samples = num_samples
        
    def forward(self, x):
        logits = super(HSICFeatureNetGumbelSoftmax, self).forward(x)

        self.importance_weights =  self.gumbel_softmax_sampling(logits, temperature=.01, num_samples=self.num_samples)

        return self.importance_weights, self.sigmas, self.sigma_y



"""
Training a network with maximizing HSIC with Sparsemax only
"""

class HSICFeatureNetSparsemax(HSICFeatureNet):
    def __init__(self, input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params):
        super(HSICFeatureNetSparsemax, self).__init__(input_dim, feature_layers, act_fun_featlayer, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)
        
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, x):
        logits = super(HSICFeatureNetSparsemax, self).forward(x)

        self.importance_weights = self.sparsemax(logits)

        return self.importance_weights, self.sigmas, self.sigma_y
