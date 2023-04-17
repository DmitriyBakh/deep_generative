import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# # Create a mixture of two Gaussians
# np.random.seed(1242314)
# N = 2000
# first_mode = norm(0, 1)
# second_mode = norm(5, 2)
# data = np.concatenate([first_mode.rvs(N), second_mode.rvs(N)]).reshape(-1, 1)

# # Plot the probability densities for different values of bandwidth
# fig, ax = plt.subplots()
# x = np.linspace(-5, 15, num=1000).reshape(-1, 1)
# for bandwidth in (0.1, 1, 3):
#     estimator = KernelDensity(bandwidth=bandwidth).fit(data)
#     predictions = estimator.score_samples(x)
#     print(f"Bandwidth {bandwidth}; mean test log-likelihood {predictions.mean():.3}")
#     # Predictions are log(p(x))
#     probabilities = np.exp(predictions)
#     ax.plot(x, probabilities, label=f"Bandwidth={bandwidth}")
# ax.plot(x, 0.5*(first_mode.pdf(x.flatten()) + second_mode.pdf(x.flatten())), label="Groud truth")
# ax.set_xlabel("$x$")
# ax.set_ylabel("Probability density")
# ax.legend();

# # Load the digits data. It's almost like MNIST -- but a lot simplier for the simplier model
# digits = load_digits()

# # Project the 64-dimensional data to a lower dimension
# pca = PCA(n_components=15, whiten=False)
# data = pca.fit_transform(digits.data)

# # Use grid search cross-validation to optimize the bandwidth
# params = {'bandwidth': np.logspace(-1, 1, 20)}
# grid = GridSearchCV(KernelDensity(), params)
# grid.fit(data)
# optimal_kde = grid.best_estimator_

# print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# # Use the best estimator to compute the kernel density estimate
# kde = grid.best_estimator_

# # Sample 44 new points from the data
# new_data = kde.sample(44, random_state=0)
# new_data = pca.inverse_transform(new_data)

# # Turn data into a 4x11 grid
# new_data = new_data.reshape((4, 11, -1))
# real_data = digits.data[:44].reshape((4, 11, -1))

# # Plot the real digits and the sampled digits
# fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
# for j in range(11):
#     ax[4, j].set_visible(False)
#     for i in range(4):
#         im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
#                              cmap=plt.cm.binary, interpolation='nearest')
#         im.set_clim(0, 16)
#         im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
#                                  cmap=plt.cm.binary, interpolation='nearest')
#         im.set_clim(0, 16)

# ax[0, 5].set_title('Selection from the input data')
# ax[5, 5].set_title('"New" digits drawn from the kernel density model')

# plt.show()

first_mode = norm(0, 1)
second_mode = norm(5, 2)
def generate_sample(sample_size:int):
    """
    Produces a toy dataset
    Args:
        sample_size:int -- the desired sample size
    Returns:
        np.array(sample_size, 1) -- a toy dataset with single feature and sample_size examples
    """
    half_size = sample_size // 2
    # In the case of an odd sample size
    second_half_size = sample_size - half_size
    return np.concatenate([first_mode.rvs(half_size), second_mode.rvs(second_half_size)]).reshape(-1, 1)

sizes = np.linspace(10, 5000, num=10, dtype=np.uint32) # training data sizes

# Task 1
# Plot the optimal bandwidth for the combination of two Gaussians as a function of the training dataset size for training dataset size in `np.linspace(10, 5000, num=10)`. Use `generate_sample` for generating the training data. Given that it is possible to generate an infinite amount of data, it is possible to find the answer with arbitrary precision. For the purpose of this task it is enough to evaluate the performance with bandwidth in ` np.logspace(-1, 1, 20)` on 4-fold cross-validation.

# # find optimal bandwidth for each training size, with 4-fold cross-validation
# bandwidths = []
# for size in sizes:
#     data = generate_sample(size)
#     params = {'bandwidth': np.logspace(-1, 1, 20)}
#     grid = GridSearchCV(KernelDensity(), params, cv=4)
#     grid.fit(data)
#     optimal_kde = grid.best_estimator_
#     bandwidths.append(optimal_kde.bandwidth)

# # plot optimal bandwidth from training size
# plt.plot(sizes, bandwidths)
# plt.xlabel("Training size")
# plt.ylabel("Optimal bandwidth")
# plt.title("Optimal bandwidth from training size")
# plt.show()


# Task 2
# Use your creativity, physical intuition, knowledge of statistics and ability to read the [KernelDensity documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) to find the  optimal kernel density parameters for training dataset size=134. Use `generate_sample` for generating as much training and testing data as you desire. Note: 134 is a low number, so make sure your solution is not overfitted to a single training sample. The goal is to have the mean log-likelihood on a test sample drawn from the same distribution > -2.38

# optimal_params = <your params>
# E.g. optimal_params = {"bandwidth": 2., "kernel": "epanechnikov"}

params_grid = {
  ### YOUR CODE HERE    
  'bandwidth': np.logspace(-1, 1, 20),
#   'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
  'kernel': ['gaussian', 'cosine']
}

# optimal_params = ??? ### YOUR CODE HERE
iterations = 25
train_size = 134
test_size = 1000

# Generate test data
test_data = generate_sample(test_size)

best_params = None
best_mean_log_likelihood = float('-inf')
optimal_params = []
mean_log_likelihoods = []

print(f"Training size: {train_size}")

for i in range(iterations):
    train_data = generate_sample(train_size)

    grid = GridSearchCV(KernelDensity(), params_grid, cv=4)
    grid.fit(train_data)
    optimal_kde = grid.best_estimator_

    optimal_params.append(grid.best_params_)
    test_log_likelihood = optimal_kde.score_samples(test_data)
    mean_test_log_likelihood = test_log_likelihood.mean()
    mean_log_likelihoods.append(mean_test_log_likelihood)

    if mean_test_log_likelihood > best_mean_log_likelihood:
        best_mean_log_likelihood = mean_test_log_likelihood
        best_params = grid.best_params_

optimal_params = best_params
