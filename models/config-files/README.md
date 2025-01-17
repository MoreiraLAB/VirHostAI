This folder includes configuration files that set up the hyperparameter searches for the Multi-Layer Perceptron (MLP) and Autoencoder (AE) models. Each configuration defines the search spaces and parameters used to optimize the models’ training and performance.

### config-ae.yaml - Autoencoder Configuration:
This file outlines the hyperparameter search for the Autoencoder model, using a random search approach aimed at minimizing the loss across training epochs. Key parameters include:
- encoder_layers: Specifies different layer architectures for the encoder, offering a range of layer sizes to adjust model depth and feature extraction capacity.
- latent_vector: Sets possible latent space dimensions, ranging from 180 down to 40.
- activation: Tests various activation functions (relu, leaky-relu, gelu) to find the most effective non-linear transformation.
- criterion: Uses Mean Squared Error (mse) as the loss function for model training.
- optimizer: Applies the adam optimizer to enable adaptive learning rates.
- learning_rate: Explores a range between 0.0001 and 0.01 for learning rate selection.
- splits: Sets a data split ratio of [0.9, 0.1, 0], allocating 90% of the data for training and 10% for validation. First value is for training, second for validation and third for test. All values must add up to 1.
- epochs: Tests epochs ranging from 6 to 14.
- batch_size: Evaluates batch sizes between 32 and 256 to ensure stability during training.
- shuffle: Shuffles data at each epoch to improve model generalization.

### config-mlp.yaml - MLP Configuration:
This file configures the hyperparameter search for the MLP model, using random search to minimize training epoch loss. Primary parameters include:

- layers: Defines options for MLP layer architecture and depth, such as num_layers, with flexible architecture types (e.g., =, <, >, <>).
- activation: Tests multiple activation functions (relu, leaky-relu, gelu, tanh, sigmoid) to evaluate non-linearity effects.
- optimizer: Tests both adam and sgd optimizers.
- learning_rate: Adjusts learning rates over a range from 0 to 0.1.
- splits: Data split configurations include options like [0.7, 0.3, 0] and [0.8, 0.2, 0]. First value is for training, second for validation and third for test. All values must add up to 1.
- epochs: Ranges between 4 and 10 epochs.
- batch_size: Tests batch sizes from 32 to 512 to optimize training throughput.
- shuffle: Ensures data shuffling to enhance training robustness.

These configuration files allow extensive hyperparameter tuning, providing flexibility to adapt both models to the dataset’s specific requirements for optimal performance.