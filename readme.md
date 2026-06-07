# Neural Networks Week 1 (1 - 7 June)

## Parth Nikam 

## Github link: https://github.com/parthnikam/Neural-Networks

## TLDR: I implemented a neural network from scratch with numpy and then using pytorch to run on my GPU. Took a sample dataset from kaggle and went through some different ways to stack layers and add dropouts to see how the accuracy of the output changes. 


### What I learnt this week
1. I dove deep into neural networks and various techniques that go into training a neural network
  * Neural networks are just matrices which are passed through a series of layers
  * In a Layer of a neural network, the input matrix is multiplied with a weight matrix and add a bias matrix to that
  * This is done for each layer where the output of the previous layer becomes the input of the next layer
  * Activation functions act as nonlinear rectifiers to the network which allow it to learn patterns
  * A network cannot be trained all at once, we need to chunk the training process down into batches 
  * To make the network learn patterns robustly, we shut off some neurons so that it doesn't overfit to a certain input pattern.

2. Used pytorch to make a model. 
  * took a dataset from kaggle on **Gen Z Dating in India** and trained a neural network to predict their level of satisfaction from the dating scene. 
  * cleaned the data and transformed it into numerical form 
  * used some important columns to act as input and dropped the others
  * trained the base model with 2 dense layers and an input and an output layer 
  * took the help of chatgpt to optimize the model architecture
  * took the help of chatgpt to figure out how to use pytorch on GPU to train the model faster

### What code did I actually write
1. I built the Neural Network classes myself along with the pytorch implementation 
2. I normalized the dataset and converted text columns to numbers 
3. I came up with the model architecture for the dataset 

### Where did I need help from chatgpt
I needed chatgpt to help me when: 
1. I didn't know how to write the mathematical part of the Layer Normalization and Dropout classes in the neural network ***- it just went over my head***
2. I didn't know how to use pytorch with cuda ***- turns out you just have to set the device to use GPU nothing else***
3. I didn't know how to think about an optimal model architecture from the beginning ***- which after some brainstorming sessions, I understood how many layers and how many dropouts I should be adding***
4. There were certain parts of the code where I knew what I wanted but didn't exactly know what should be written or how it should be carried out.

### What were the challenges I faced
1. Initially I was a little skeptical about using neural networks on a classification task
  * turns out they work just fine
2. finding out why accuracy is different in training and testing and that layer dimensions, number of layers and learning rate have a lot to do with it. 
  * Tried adding dropout and batch normalization to the model

### What can I confidently do now
1. I can assess the kind of neural network a problem might need
2. I can write a neural network architecture in pytorch and train it with different hyper parameters to improve the output accuracy 
3. I can take a dataset and find features that are important to train the network




