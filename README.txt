CS747 Deep Learning Assignment 1 - CIFAR-10 Classification
Author: Shasank kamineni

Overview
This repository contains the implementation of three classical machine learning models — Perceptron, Support Vector Machine (SVM), and Softmax Classifier — for classification of the CIFAR-10 dataset. Each model is implemented from scratch using NumPy and Scikit-learn and optimized using Stochastic Gradient Descent (SGD).

Models Implemented:
Perceptron
Support Vector Machine (SVM)
Softmax Classifier
All models were implemented within a Jupyter Notebook and colab environment for interactive experimentation. The implementations were followed by training and evaluation of the models on the CIFAR-10 dataset, including validation and test accuracies.

Dataset: CIFAR-10
CIFAR-10 is a collection of 60,000 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. Each image is 32x32 pixels with three color channels (RGB).

For this assignment, the training set was further divided into a training set and a validation set. The division was:

Training Set: 40,000 images
Validation Set: 10,000 images
Test Set: 10,000 images
Project Structure
1. CS747_DL_Assignment-1.ipynb
This notebook contains the main code for training and evaluating the Perceptron, SVM, and Softmax models on the CIFAR-10 dataset. It includes the following:

Data Preprocessing: Reshaping and standardizing the images.
Model Training: Training each model with its respective hyperparameters.
Accuracy Evaluation: Calculating accuracy on training, validation, and test sets.
Kaggle Submission: Outputting the test predictions for Kaggle submission.
2. skamine3_assignment1_output.pdf
This PDF file includes the output of the models, showing the training process, loss curves, and accuracy scores for each model.


Why the models are implemented directly in the notebook instead of calling from separate folders/files:
In the assignment notebooks, I chose to combine multiple code segments into fewer, larger blocks rather than using separate cells for each step. This approach streamlines the workflow, making the code easier to read and execute without having to jump between isolated cells. By grouping related operations like model initialization, training, and evaluation into a single block, I ensured that the full process could be run in one go, improving both efficiency and debugging. This also keeps the context of the model's flow intact, allowing for easier understanding of how different elements interact, especially when tuning parameters. While the skeleton code might separate things for learning purposes, my approach is more practical and suited to real-world coding environments where reducing redundancy and simplifying execution are key.



Models:


Perceptron:


The perceptron model was trained with a focus on experimenting with different learning rates and epochs. The goal was to minimize classification errors by updating the weights based on misclassified examples.


SVM (Support Vector Machine):
The SVM model maximizes the margin between classes using hinge loss. Implementing it in the notebook allowed for detailed gradient descent optimization with real-time monitoring of the hinge loss for each class.


Softmax Classifier:
The softmax classifier computes probabilities across multiple classes and uses cross-entropy loss for training. Regularization was added to prevent overfitting, and the notebook format allowed for easy tracking of loss curves and tuning of hyperparameters like regularization strength.


How to Run the Code
Clone the repository or download the Jupyter notebook.

Ensure you have the following dependencies installed:

Python 3.x
NumPy
pandas
Scikit-learn
TensorFlow or Keras (for data loading)
You can install the dependencies using pip:

bash
Copy code
pip install numpy pandas scikit-learn tensorflow
Open the CS747_DL_Assignment-1.ipynb file in Jupyter Notebook or Google Colab.

Run the notebook cells sequentially:

The notebook will load the CIFAR-10 dataset, preprocess it, and train each model in turn.
After training, it will output the training, validation, and test accuracies.
The final section of the notebook will generate a CSV file containing the test predictions, which can be submitted to the Kaggle competition for grading.

Results Summary

 (Results were very tiny change is observed compare to Kaggle submission because i have ran the whole code again for cross checking before submission) 

The performance of the three models is summarized below:

Model	Training Accuracy	Validation Accuracy	Test Accuracy
Perceptron	52.34%			48.30%		49.39%
SVM		35.28%			34.84%		35.74%
Softmax 	40.06%			39.14%		39.19%

