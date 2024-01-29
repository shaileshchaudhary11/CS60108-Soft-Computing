## Overview
This repository contains code for training and evaluating an Artificial Neural Network (ANN) using PyTorch on the MNIST dataset. The ANN is designed with a customizable architecture, and the training process includes monitoring various performance metrics.

## Code Structure
ANN class: This class defines the architecture of the artificial neural network. It can be customized by specifying the input size, number of classes, number of hidden layers, and sizes of hidden layers. The network is implemented using PyTorch's nn.Module and includes methods for forward pass, training, and printing architecture details.

Training Script (train_ann.py): This script demonstrates how to instantiate an ANN object, train the network on the MNIST dataset, and evaluate its performance. It includes functionality for specifying hyperparameters such as the number of epochs, learning rate, optimizer (SGD in this case), and loss criterion (CrossEntropy).

Metrics and Visualization (metrics.py): This script contains functions for calculating and visualizing performance metrics such as accuracy, precision, recall, and F1-score. It uses the scikit-learn library for metric calculations and seaborn for plotting.

## Prerequisites
Ensure that you have the following dependencies installed:

Python (3.6 or later)
PyTorch
torchvision
scikit-learn
seaborn
matplotlib
You can install the required packages using the following command:

bash
Copy code
pip install torch torchvision scikit-learn seaborn matplotlib
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repo.git
cd your-repo
Run the training script:

bash
Copy code
python train_ann.py
This will train the ANN on the MNIST dataset using the specified hyperparameters.

View Metrics:
After training, the script will print average validation accuracy, precision, recall, and F1-score. Additionally, it will generate line plots showing how these metrics change over epochs.

Customize Architecture (Optional):
If you want to customize the architecture of the ANN, modify the ANN class instantiation in train_ann.py by providing your desired parameters.

## Acknowledgments
This code is a basic implementation for educational purposes and can be extended for more complex use cases.
The ANN architecture and training script can be modified to suit specific requirements and datasets.
Feel free to experiment, tweak, and extend this code for your own projects. If you encounter any issues or have suggestions, please open an issue on GitHub.

Happy coding!
