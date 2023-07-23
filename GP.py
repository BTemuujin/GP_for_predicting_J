import torch
import gpytorch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the GP model
class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GaussianProcess, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Load data from file
data = np.loadtxt('input.txt')  # Replace 'input.txt' with your file name

# Extract magnetic moments of site 1, site 2, and distance
moments_site1 = torch.tensor(data[:, 0])
moments_site2 = torch.tensor(data[:, 1])
distances = torch.tensor(data[:, 2])
interactions = torch.tensor(data[:, 3])

# Normalize the inputs and outputs (optional but recommended)
moments_site1 = (moments_site1 - moments_site1.mean()) / moments_site1.std()
moments_site2 = (moments_site2 - moments_site2.mean()) / moments_site2.std()
distances = (distances - distances.mean()) / distances.std()
interactions = (interactions - interactions.mean()) / interactions.std()

# Create a random permutation of indices
n_samples = len(interactions)
indices = np.random.permutation(n_samples)

# Split indices into training and testing sets
n_train = 152 # training data quantity
train_indices = indices[:n_train]
test_indices = indices[n_train:]

# Create PyTorch tensors for training and testing
train_x = torch.stack((moments_site1[train_indices], moments_site2[train_indices], distances[train_indices]), dim=1)
train_y = interactions[train_indices]

test_x = torch.stack((moments_site1[test_indices], moments_site2[test_indices], distances[test_indices]), dim=1)
test_y = interactions[test_indices]

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GaussianProcess(train_x, train_y, likelihood)

# Set model and likelihood in training mode
model.train()
likelihood.train()

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calculate loss and backpropagate gradients
    loss = -mll(output, train_y)
    loss.backward()
    # Update the parameters
    optimizer.step()
    #print(loss)

# Set model and likelihood in evaluation mode
model.eval()
likelihood.eval()

# Perform predictions on the test data
with torch.no_grad():
    preds = model(test_x)

#print(preds)
# Denormalize the predictions (optional)
preds = preds.mean * interactions.std() + interactions.mean()
#print(preds)

# Calculate the mean absolute error (MAE) for evaluation
mae = torch.mean(torch.abs(preds- test_y))

# Convert the predictions and true values to numpy arrays for easier comparison
preds = preds.numpy()
test_y = test_y.numpy()

# Print the predicted values and true values side by side
for pred, true in zip(preds, test_y):
    print(f"{pred:.4f} \t {true:.4f}")

print(f"Mean Absolute Error: {mae.item()}")
