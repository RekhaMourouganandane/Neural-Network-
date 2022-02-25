# Neural-Network-
# CE889 – Neural Networks and Deep Learning Assignment
# Autumn 2021
# 1. Objectives
• To translate the theoretical knowledge gained throughout the lectures into practise by designing and implementing a neural network capable of solving a dynamic environment problem.

• Understand how different deep learning architectures work.

• Be capable of implementing a deep learning architecture using TensorFlow and PyTorch.

• Present the results of using a dataset from a Kaggle Competition to train a deep learning architecture and make predictions with them. 

# 2. Description of the Assignment
The assignment for the module consist of two parts: an individual project and a group project.
# Individual project
• In the lunar lander game the user controls an spaceship that is trying to land on a specific target area of the map. The user needs to move the spaceship
towards the target area and then slowly move it down so that it lands correctly.
• The student will design and implement a neural network with a single hidden layer that will be able to play the lunar lander game simulator.
• The neural network should be implemented in python and external libraries are NOT allowed.
• The neural network will receive two inputs (distance to target in X and distance to target in Y) and predict what should be the expected velocity in X and in Y (two outputs).
• The game simulator of the lunar lander game is going to be provided. The student should only focus on designing and implementing the neural network.
• The neural network should be trained offline and the weights should be saved in a file.
• The neural network should be tested online (playing the game) using the weights that were saved after the training process was done.
• Evaluate the performance of your neural network with the metric Root Mean Square Error (RMSE). You need to evaluate the performance at the end of the training process and after each epoch. Two RMSE metrics should be presented, one for the training set and another one for the validation set.
