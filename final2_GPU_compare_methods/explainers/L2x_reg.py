import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

BATCH_SIZE = 100
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# The number of key features for each dataset
ks = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5}

class SampleConcrete(nn.Module):
    """
    Layer for sampling Concrete / Gumbel-Softmax variables.
    """
    def __init__(self, tau0, k):
        super(SampleConcrete, self).__init__()
        self.tau0 = tau0
        self.k = k

    def forward(self, logits, training=True):
        # logits: [BATCH_SIZE, d]
        logits_ = logits.unsqueeze(1)  # [BATCH_SIZE, 1, d]
        
        batch_size = logits_.size(0)
        d = logits_.size(2)
        
        uniform = torch.rand((batch_size, self.k, d)).clamp_min(1e-10).to(logits.device)
        gumbel = -torch.log(-torch.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = F.softmax(noisy_logits, dim=-1)
        samples = torch.max(samples, dim=1)[0]
        
        if training:
            return samples
        else:
            threshold = torch.topk(logits, self.k, dim=-1)[0][:, -1].unsqueeze(-1)
            discrete_logits = (logits >= threshold).float()
            return discrete_logits

class L2XModel(nn.Module):
    def __init__(self, input_shape, num_feature_imp, tau=0.1):
        super(L2XModel, self).__init__()
        self.tau = tau
        self.num_feature_imp = num_feature_imp
        
        # P(S|X) network
        self.s_dense1 = nn.Linear(input_shape, 100)
        self.s_dense2 = nn.Linear(100, 100)
        self.logits = nn.Linear(100, input_shape)
        
        # q(X_S) network
        self.dense1 = nn.Linear(input_shape, 200)
        self.dense2 = nn.Linear(200, 200)
        self.preds = nn.Linear(200, 1)
        
        # Concrete layer for sampling
        self.sample = SampleConcrete(self.tau, self.num_feature_imp)

    def forward(self, x, training=True):
        # P(S|X) network
        net = F.relu(self.s_dense1(x))
        net = F.relu(self.s_dense2(net))
        logits = self.logits(net)

        # Sampling with Concrete
        samples = self.sample(logits, training)

        # q(X_S) network
        new_model_input = x * samples
        net = F.relu(self.dense1(new_model_input))
        net = F.relu(self.dense2(net))
        preds = self.preds(net)
        
        return preds, samples

def train_L2X(X_train, y_train, num_feature_imp, epochs=100, batch_size=BATCH_SIZE):
    # Initialize model
    n, d = X_train.shape
    model = L2XModel(input_shape=d, num_feature_imp=num_feature_imp)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch, y_batch
            optimizer.zero_grad()
            preds, _ = model(X_batch, training=True)
            loss = criterion(preds.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    # Inference: Generate feature scores for test set
    # model.eval()
    X_test_tensor = torch.tensor(X_train, dtype=torch.float32)
    with torch.no_grad():
        _, feature_scores = model(X_test_tensor, training=False)

    return model, feature_scores.cpu().numpy()

# Example usage:
# scores = train_L2X(X_train, y_train, input_shape=..., num_feature_imp=..., X_test=X_test)
