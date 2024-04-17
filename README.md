# Visualizing Neural Network Inner Workings

This project focuses on visualizing the inner workings of neural networks by generating plots of hidden layer activations. The goal is to provide insights into how neural networks process and transform data, particularly using the MNIST dataset for handwritten digit recognition.

## Description

Neural networks are powerful models for learning complex patterns from data. Visualizing the activations of hidden layers helps understand how these networks transform input data through various stages of computation. This project demonstrates this process using a trained neural network on the MNIST dataset.

## Usage

### Training the Model

1. **Training Script:** Run `train.py` to train the neural network model.
   ```bash
   python train.py

This script will train the model and generate the following files:
1. train_dict.pkl: Training dictionary containing model parameter details.
2. digits_mnist.model: Trained model file saved for future use.


### Generating Visualizations

1. **Visualizing Script:** After training, run visual.py to generate visualizations of hidden layer activations.
   ```bash
   python visual.py

This script will  generate the folder of .png images

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.


## License
This project is licensed under the MIT License.

