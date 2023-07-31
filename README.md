# Movie-Review-Sentiment-Classification
A neural network model for sentiment analysis of movie reviews . The model is built using [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/random_uniform_initializer) 

**Note:** This README.md file contains an overview of the project, it is recommended to open [notebook](https://github.com/kwasiasomani/Movie-Review-Sentiment-Classification/blob/master/notebook/Capstone.ipynb) as it contains the code and further explanation for the results.

## Table of Contents
- [Movies-Reviews-Classification](#movies-reviews-classification)
  * [Dataset](#dataset)
    + [IMDB Dataset](#imdb-dataset)
    + [Data Splitting](#data-splitting)
    + [Data Preprocessing](#data-preprocessing)
  * [Model Architecture](#model-architecture)
  * [Improving the Model](#improving-the-model)
    + [Hyperparameter Tuning](#hyperparameter-tuning)
    + [Regularization using Dropout](#regularization-using-dropout)
  * [Model Evaluation](#model-evaluation)
  * [Results](#results)
  * [Contributers](#contributers)

## Dataset
- The project needs a dataset for movies and TV shows reviews, [IMDb](https://www.imdb.com/) is a popular website for movies and TV shows. It has a database of over 8 million movies and TV shows. Using a dataset from this website will be a good choice for the project to train our neural network and test it.

### IMDB Dataset
- Instead of using the whole dataset, we will use a subset of the dataset. The dataset contains 50,000 reviews for movies and TV shows. The dataset is already balanced, meaning that it contains an equal number of positive and negative reviews. The dataset is available on [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Data Splitting
- Since the dataset is already balanced, we will split the dataset into 70% training set, 20% validation set, 10% testing set . The training set will be used to train the neural network, validation set is used to further tune the hyperparameters and the testing set will be used to evaluate the neural network.

### Data Preprocessing
- Text pre-processing is essential for NLP tasks. So, you will apply the following steps on
our data before used for classification:
    * Remove punctuation.
    * Remove stop words.
    * Lowercase all characters.
    * Lemmatization of words.
- The data preprocessing is done using the [NLTK](https://www.nltk.org/) library.

## Model Architecture
- The project uses [PyTorch](https://pytorch.org/) to build the neural network. The neural network is a simple feedforward neural network with 5 layers.
- The network's input layer takes in 768 inputs corresponding to the vector provided by BERT's pooled output (classification output)
- Our network consists of 4 hidden layers with 512, 256, 128, 64 units respectively.
- The hidden layers have ReLU activation function.
- The output layer have sigmoid activation function to classify the vector
- The network uses [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) optimizer and [Binary Cross Entropy](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss) loss function.

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
- You can find the model's performance for different learning rates in the [results](results) folder 
### Regularization using Dropout
- Dropout is a regularization technique that randomly drops out some of the neurons in the network. This technique is used to prevent overfitting.
- Dropout is applied to the hidden layers of the network. The dropout rate can be specified while initializing the network. The dropout rate is the probability of a neuron to be dropped out. The dropout rate is set to 0.4 in this project.

## Model Evaluation
- The model is able to classify the reviews with 93% accuracy on raw test data. On the other hand, the accuracy reached 90% when using the preprocessed data. This indicates that not all preprocessing steps are necessary for the model to perform well.

## Results
- The model's performance on the raw test set is as follows:
    * Accuracy: 93.6%
    
 
- The confusion matrix:

![image](https://user-images.githubusercontent.com/41492875/218091822-64f96317-e683-4ec4-88df-3e65fd7136e2.png)

> Note: See [notebook](/Review_Classification.ipynb) for more details on the results.

## Contributers

- [Yousef Kotp](https://github.com/yousefkotp)

- [Mohamed Farid](https://github.com/MohamedFarid612)

- [Adham Mohamed](https://github.com/adhammohamed1)

