# Movie-Review-Sentiment-Classification
A neural network model for sentiment analysis of movie reviews . The model is built using [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer) 

**Note:** This README.md file contains an overview of the project, it is recommended to open [notebook](https://github.com/kwasiasomani/Movie-Review-Sentiment-Classification/blob/master/notebook/Capstone.ipynb) as it contains the code and further explanation for the results.

## Table of Contents
- [Movies-Reviews-Classification](#Movie-Review-Sentiment-Classification)
  * [Dataset](#dataset)
    + [Data Splitting](#data-splitting)
    + [Data Preprocessing](#data-preprocessing)
  * [Model Architecture](#model-architecture)
  * [Improving the Model](#improving-the-model)
    + [Hyperparameter Tuning](#hyperparameter-tuning)
  * [Model Evaluation](#model-evaluation)
  * [Results](#results)
  

## Dataset
- The project needs a dataset for movies and TV shows reviews, [Zindi](https://zindi.africa/competitions/movie-review-sentiment-classification-challenge) is a popular website for competition. Using a dataset from this website will be a good choice for the project to train our neural network and test it.


### Data Splitting
- Since the dataset is already balanced, we will split the dataset into 75% training set, 25% validation set. The training set will be used to train the neural network, validation set is used to further tune the hyperparameters and the testing set will be used to evaluate the neural network.

### Data Preprocessing
- Text pre-processing is essential for NLP tasks. So, you will apply the following steps on
our data before used for classification:
    * Remove Special characters.
    * Lowercase all characters.
    * Tokenization of words.

## Model Architecture
- The project uses [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer) to build the neural network. The neural network is a simple feedforward neural network with 5 layers.
- The network's input layer takes in 100 inputs
- Our network consists of 1 hidden layer with 64 units respectively.
- The hidden layers have ReLU activation function.
- The output layer have sigmoid activation function to classify the vector
- The network uses [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) optimizer and [Binary Cross Entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)

## Improving the Model
- The model can be improved by using different hyperparameters and regularization techniques. The following techniques are used to improve the model:
### Hyperparameter Tuning
- The following hyperparameters can be tuned:
    * Learning Rate
    * Batch Size
    * Number of Epochs
    * Number of Hidden Layers
    * Number of Units in each Hidden Layer
    * Activation Function
    * Optimizer
    * Loss Function
- We are only tuning the learning rate in this project since the other hyperparameters will have slight to no effect on the model's performance.
### Regularization using Dropout
- Dropout is a regularization technique that randomly drops out some of the neurons in the network. This technique is used to prevent overfitting.
- Dropout is applied to the hidden layers of the network. The dropout rate can be specified while initializing the network. The dropout rate is the probability of a neuron to be dropped out. The dropout rate is set to 0.4 in this project.

## Model Evaluation
- The evaluation metrics was accuracy

## Results
- The model's performance on the raw test set is as follows:
    * Accuracy: 86%
    
 


