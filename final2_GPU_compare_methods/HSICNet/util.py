import torch 

def initialize_sigma_median_heuristic(X):
        """
        X: Tensor of shape (n_samples, n_features)
        Returns: Initial sigma values for each feature using the median heuristic
        """
        n, d = X.size()
        sigma_init = torch.zeros(d)

        # Calculate median of pairwise distances for each feature
        for i in range(d):
            feature_values = X[:, i].unsqueeze(1)
            pairwise_dists = torch.cdist(feature_values, feature_values, p=2).squeeze()
            sigma_init[i] = torch.median(pairwise_dists)

        return sigma_init

def initialize_sigma_y_median_heuristic(Y):
    """
    Y: Tensor of shape (n_samples, 1) for the outputs
    Returns: Initial sigma_Y using the median heuristic
    """
    # Ensure Y is 2D (n_samples, 1)
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(1)

    # Compute pairwise distances for outputs
    pairwise_dists = torch.cdist(Y, Y, p=2).squeeze()
    sigma_Y_init = torch.median(pairwise_dists)

    return sigma_Y_init
