# Neural Network with Gumbel-Softmax
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from sparsemax import Sparsemax

import torch.nn.utils.weight_norm as weight_norm


class HSICNet(nn.Module):
    '''
        The superclass for different neural networks for optimizing HSIC
        inputs:
        input_dim: the input dimensions of the data
        layers: List[int]: a list of integers, each indicating the number of neurons of a layer 
    '''
    def __init__(self, input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params):
        super().__init__()

        self.mode = params.get('mode', 'regression')
        if self.mode == 'classification':
            self.kernel_y_func = self.categorical_kernel
        else:
            self.kernel_y_func = self.rbf_kernel_y

        self.layers = nn.ModuleList()

        self.layers.append((nn.Linear(input_dim, layers[0])))

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        self.layers.append(nn.Linear(layers[-1],input_dim))

        if act_fun_layer == None: act_fun_layer = nn.Sigmoid()
        self.activation_func = act_fun_layer() #Tanh() ReLU() SELU() Sigmoid()

        # Initializing the sigmas based on the input (which is median heuristic)
        self.sigmas = nn.Parameter(sigma_init_X)
        self.sigma_y = nn.Parameter(sigma_init_Y)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if i < len(self.layers) - 1:  # Apply activation function to all but the last layer
                # if isinstance(self.activation_func, nn.Sigmoid):
                #     batch_norm = nn.BatchNorm1d(x.size(1))
                #     x = batch_norm(x)
                x = self.activation_func(x)
        
        return x

    # Gumbel-Softmax Sampling
    def gumbel_softmax_sampling(self, logits, temperature=0.1, num_samples=5):
        """
        Perform Gumbel-Softmax sampling multiple times and return the max value for each feature.
        """

        batch_size, d = logits.size()

        # Expand logits to [BATCH_SIZE, 1, d] for sampling multiple times
        logits_expanded = logits.unsqueeze(1)  # [BATCH_SIZE, 1, d]

        # Uniform random noise [BATCH_SIZE, k, d]
        uniform = torch.rand(batch_size, num_samples, d, device=logits.device)

        # Sample Gumbel noise and add to logits
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        noisy_logits = (gumbel + logits_expanded) / temperature

        # Apply softmax to get Gumbel-Softmax samples
        samples = F.softmax(noisy_logits, dim=-1)  # [BATCH_SIZE, k, d]

        # Take the maximum over the k samples
        samples = torch.max(samples, dim=1)[0]

        return samples

    # Gumbel-Sparsemax Sampling
    def gumbel_sparsemax_sampling(self, logits, temperature=10, num_samples=5):
        """
        Perform Gumbel-Sparsemax sampling multiple times and return the max value for each feature.
        """
        # np.random.seed(0)

        batch_size, d = logits.size()

        # Expand logits to [BATCH_SIZE, 1, d] for sampling multiple times
        logits_expanded = logits.unsqueeze(1)  # [BATCH_SIZE, 1, d]

        # Uniform random noise [BATCH_SIZE, k, d]
        uniform = torch.rand(batch_size, num_samples, d, device=logits.device)

        # Sample Gumbel noise and add to logits
        gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        noisy_logits = (gumbel + logits_expanded) / temperature

        # changed softmax to Sparsemax
        sparsemax = Sparsemax(dim=-1)
        samples = sparsemax(noisy_logits)

        # Take the maximum over the k samples
        samples = torch.mean(samples, dim=1) # torch.max(samples, dim=1)[0] # 

        return samples

    # ANOVA Kernel 
    def anova_kernel(self, X1, X2, s, sigmas):
        """
        ANOVA kernel incorporating feature importance (s) and per-feature sigma.
        Args:
        - X1, X2: Input tensors of shape (num_samples, num_features)
        - s: Feature importance weights (output of the Gumbel-Softmax)
        - sigmas: Trainable sigmas for each feature
        Returns:
        - Kernel matrix for inputs X1 and X2
        """
        prod = torch.ones((X1.size(0), X2.size(0)), device=X1.device)
        for i in range(X1.size(1)):  # iterate over features
            dists = (X1[:, i].unsqueeze(1) - X2[:, i].unsqueeze(0)) ** 2
            prod *= (1 + s[:, i].unsqueeze(1) * torch.exp(-dists / (2 * sigmas[i]**2)))
        return prod

    # Modified RBF Kernel for label y with learnable sigma
    def rbf_kernel_y(self, y1, y2, sigma_y):
        """
        RBF Kernel for the label y, incorporating a trainable sigma_y.
        """
        dists = (y1.unsqueeze(1) - y2.unsqueeze(0)) ** 2
        return torch.exp(-dists / (2 * sigma_y**2))

    # Categorical kernel for classification problems
    def categorical_kernel(self, y1, y2, sigma_y=None):
        """
        Compute the categorical kernel between two tensors of categorical variables.

        Parameters:
        y1 (torch.Tensor): 1D tensor of categorical values.
        y2 (torch.Tensor): 1D tensor of categorical values.

        Returns:
        torch.Tensor: Kernel matrix where element (i, j) is 1 if y1[i] == y2[j], and 0 otherwise.
        """
        # Ensure inputs are 1D tensors
        if y1.ndim != 1 or y2.ndim != 1:
            raise ValueError("Input tensors must be 1D tensors of categorical values.")
        
        # Compute the kernel matrix
        kernel_matrix = (y1.unsqueeze(1) == y2.unsqueeze(0)).float()
        
        return kernel_matrix
    
    # The Loss function based on HSIC
    def hsic_loss_adaptive(self, X, y, s, sigmas, sigma_y):
        X_kernel = self.anova_kernel(X, X, s, sigmas)
        y_kernel = self.kernel_y_func(y, y, sigma_y) #self.rbf_kernel_y(y, y, sigma_y)

        # Centering the kernels
        n = X.size(0)
        H = torch.eye(n).to(X.device) - (1 / n) * torch.ones(n, n).to(X.device)

        X_centered = H @ X_kernel @ H
        y_centered = H @ y_kernel @ H

        # HSIC value with proper scaling
        hsic_value = torch.trace(X_centered @ y_centered) / n ** 2
        return hsic_value

    # Training function
    def train_model(self, X, y, num_epochs=300, lr=1e-3, BATCH_SIZE = 100):
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(num_epochs):
            self.train()
            for inputs, outputs in train_loader:
                optimizer.zero_grad()

                s, sigmas, sigma_y = self(inputs)  # importance weights from Gumbel-Softmax
                loss = -self.hsic_loss_adaptive(inputs, outputs, s, sigmas, sigma_y)  # Minimize negative HSIC

                loss.backward()
                optimizer.step()

                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    ## Instance-wise Shapley Value      
    def instancewise_shapley_value(self, X_train, y_train, X_samples, y_samples, num_samples, sigmas, sigma_y, weights):
        n, d = X_train.shape
        n_samples = X_samples.shape[0]

        sv = torch.zeros(n_samples, d)
        hsic_values = torch.zeros(n_samples)

        # Loop through each sample in X_samples
        for idx in range(n_samples):
            x = X_samples[idx].unsqueeze(1)
            y = y_samples[idx]

            Ks = torch.zeros(n, d)
            anova_k = torch.ones(n)

            # Compute kernels for the current sample
            for i in range(d):
                dists = (x[i] - X_train[:, i]) ** 2
                k = weights[idx, i] * torch.exp(-dists / (2 * sigmas[i]**2))
                anova_k *= (1 + k)
                Ks[:, i] = k
            anova_k -= 1

            k_y = torch.exp(- (y - y_train) ** 2 / (2 * sigma_y**2))

            k_x_avg = torch.mean(anova_k)
            k_y_avg = torch.mean(k_y)

            # Compute Shapley values for the current sample
            for i in range(d):
                sv[idx, i], k_x_tilde = self.instancewise_sv_dim(Ks, k_y, k_x_avg, k_y_avg, i, num_samples)
            
            # Compute HSIC for the current sample
            hsic_values[idx] = (anova_k - k_x_avg) @ (k_y - k_y_avg)

        return sv, hsic_values

    def instancewise_sv_dim(self, Ks, k_y, k_x_avg, k_y_avg, dim, num_samples):
        dp = torch.zeros(self.input_dim, self.input_dim, num_samples)
        n, d = Ks.shape

        Ks_copy = Ks.clone().detach()
        Ks_copy[:, 0] = Ks[:, dim]
        Ks_copy[:, dim] = Ks[:, 0]

        sum_current = torch.zeros((n,))

        # Fill the first order dp (base case)
        for j in range(d):
            dp[0, j, :] = Ks_copy[:, j]
            sum_current += Ks_copy[:, j]

        for i in range(1, self. input_dim):
            temp_sum = torch.zeros((num_samples,))
            for j in range(self. input_dim):
                # Subtract the previous contribution of this feature when moving to the next order
                sum_current -= dp[i - 1, j, :]

                dp[i, j, :] = (i/(i+1)) * Ks_copy[:, j] * sum_current
                temp_sum += dp[i, j, :]

            sum_current = temp_sum

        k_x_tilde = torch.sum(dp[:, 0, :], axis=0)
        # print(k_x_tilde)

        X_centered = k_x_tilde - k_x_avg
        y_centered = k_y - k_y_avg

        # HSIC value with proper scaling
        sv_i = (X_centered @ y_centered)

        return sv_i, k_x_tilde

    ## Global Shapley Value
    def global_shapley_value(self, X_train, y_train, sigmas, sigma_y, weights):
        n, d = X_train.shape
        Ks = torch.zeros(d, n, n, device=X_train.device)
        anova_k = torch.ones(n, n, device=X_train.device)

        for i in range(X_train.size(1)):  # iterate over features
            dists = (X_train[:, i].unsqueeze(1) - X_train[:, i].unsqueeze(0)) ** 2
            k = (weights[:, i].unsqueeze(1) *
                torch.exp(-dists / (2 * sigmas[i]**2)))
            anova_k *= (1 + k)
            Ks[i, :, :] = k
        anova_k -= 1

        dists = (y_train.unsqueeze(1) - y_train.unsqueeze(0)) ** 2
        k_y = torch.exp(-dists / (2 * sigma_y**2))
        del dists, k

        H = (torch.eye(n) - (1 / n) * torch.ones(n, n)).to(X_train.device)

        # We define inclusive and noninclusive weights for value functions that inlcude/not-include the the corresponding feature
        # inclusive_weights = torch.zeros(d, 1)
        # noninclusive_weights = torch.zeros(d, 1)

        # for i in range(d):
        #     inclusive_weights[i] = math.factorial(i) * math.factorial(d - i - 1)
        #     if i < d-1:
        #         noninclusive_weights[i] = math.factorial(
        #             i+1) * math.factorial(d - (i + 1) - 1)
        # inclusive_weights /= math.factorial(d)
        # noninclusive_weights /= math.factorial(d)

        sv = torch.zeros(d, 1, device=X_train.device)
        for i in range(d):
            with torch.no_grad():
                sv[i] = self.global_sv_dim_efficient(Ks, k_y, H, i)

        hsic = 0 #torch.trace(H @ anova_k @ H @ k_y) / (n - 1) ** 2

        del Ks, k_y, H, anova_k
        return sv, hsic

    def global_sv_dim(self, Ks, k_y, H, dim):
        d, n = Ks.shape[0], Ks.shape[1]

        dp = torch.zeros(d, d, n, n)

        Ks_copy = Ks.clone().detach()
        Ks_copy[0, :, :] = Ks[dim, :, :]
        Ks_copy[dim, :, :] = Ks[0, :, :]

        sum_current = torch.zeros((n, n))

        # Fill the first order dp (base case)
        for j in range(d):
            dp[0, j, :, :] = Ks_copy[j, :, :]
            sum_current += Ks_copy[j, :, :]

        for i in range(1, d):
            temp_sum = torch.zeros((n, n))
            for j in range(d-i):
                # Subtract the previous contribution of this feature when moving to the next order
                sum_current -= dp[i - 1, j, :, :]

                dp[i, j, :, :] = (i / (i+1)) * Ks_copy[j, :, :] * sum_current
                temp_sum += dp[i, j, :, :]

            sum_current = temp_sum

        k_tilde = torch.sum(dp[:, 0, :, :], axis=0)

        sv_i = torch.trace(H @ k_tilde @ H @ k_y) / (n - 1) ** 2

        return sv_i#, k_tilde, dp

    def global_sv_dim_efficient(self, Ks, k_y, H, dim):
        d, n = Ks.shape[0], Ks.shape[1]

        dp = torch.zeros(d, n, n, device=Ks.device)
        d0 = Ks.clone().detach()  # Clone and prepare for swapping
        d0[0, :, :] = Ks[dim, :, :]
        d0[dim, :, :] = Ks[0, :, :]
        dp.copy_(d0)

        sum_current = torch.sum(Ks, axis=0)
        k_tilde = dp[0, :, :].clone()

        for i in range(1, d):
            temp_sum = torch.zeros((n, n), device=Ks.device)
            for j in range(d-i):
                # Subtract the previous contribution of this feature when moving to the next order
                sum_current -= dp[j, :, :]

                dp[j, :, :] = (i / (i+1)) * d0[j, :, :] * sum_current
                temp_sum += dp[j, :, :]
                if j == 0:
                    k_tilde += dp[j, :, :]

            sum_current = temp_sum

        #k_tilde = torch.sum(dp[:, 0, :, :], axis=0)

        sv_i = torch.trace(H @ k_tilde @ H @ k_y) / (n - 1) ** 2
        del k_tilde, d0, dp, sum_current
        return sv_i



"""
Training a network with maximizing HSIC with Gumbel Sparsemax
"""

class HSICNetGumbelSparsemax(HSICNet):
    def __init__(self, input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_samples, temperature=10, **params):
        super(HSICNetGumbelSparsemax, self).__init__(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)
        
        self.num_samples = num_samples
        self.temperature = temperature
        
    def forward(self, x):
        logits = super(HSICNetGumbelSparsemax, self).forward(x)

        self.importance_weights = self.gumbel_sparsemax_sampling(logits, temperature=self.temperature, num_samples = self.num_samples)

        return self.importance_weights, self.sigmas, self.sigma_y


"""
Training a network with maximizing HSIC with Gumbel Softmax Layer
"""
class HSICNetGumbelSoftmax(HSICNet):
    def __init__(self, input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, num_samples, temperature=.1, **params):
        super(HSICNetGumbelSoftmax, self).__init__(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)
        
        self.num_samples = num_samples
        self.temperature = temperature


    def forward(self, x):
        logits = super(HSICNetGumbelSoftmax, self).forward(x)

        # Apply Gumbel-Softmax sampling with 5 samples and temperature of 1
        self.importance_weights = self.gumbel_softmax_sampling(logits, temperature=self.temperature, num_samples=self.num_samples)

        return self.importance_weights, self.sigmas, self.sigma_y



"""
Training a network with maximizing HSIC with sparsemax wihtout sampling with Gumbel
"""

# Neural Network with Sparsemax
class HSICNetSparsemax(HSICNet):
    def __init__(self, input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params):
        super(HSICNetSparsemax, self).__init__(input_dim, layers, act_fun_layer, sigma_init_X, sigma_init_Y, **params)
        
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, x):
        logits = super(HSICNetSparsemax, self).forward(x)

        self.importance_weights = self.sparsemax(logits)

        return self.importance_weights, self.sigmas, self.sigma_y
