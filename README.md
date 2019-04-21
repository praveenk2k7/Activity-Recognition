# Activity-Recognition
Multi class classification for activity recognition using Motion Sense Dataset
Aim: 

Implement and study the DNN model for multi class classification for Activity Recognition using Motion Sense Dataset from github Protecting Sensory Data against Sensitive Inferences.

1.	Build DNN and tune hyperparameters using grid search to obtain the best fit for the data with high accuracy.
Model: 
Linear layers with ReLU as an activation function for all the hidden and the input layers whereas the activation function for the output layer is Softmax.
	Loss Function: Cross Entropy Loss
	Optimizer: Adam optimizer
Hyperparameters: 
•	Number of hidden layers
•	Number of hidden units
•	Learning rate
•	Number of iterations
•	Batch size
Five different values are given to each parameter and the combination is formed randomly to create a set of five values each hyperparameter set. So there are total of 5^5 sets to test on. Many sets gave good accuracy but poor F1 score. The best hyperparameters in terms of accuracy and F1 score are reported below after performing the grid  search, 

Number of hidden layers=1
Number of neurons=10
epochs=2000
learning Rate=0.01
batch size=128

2. Evaluate Model performance using various metrics such as precision, Recall and F1 Score.

3. Test the model with different optimizers like Grad descent and adagrad using pytorch.

4. Visualize the testing set with PCA using first two pricipal components

5. Visulaize the testing set with t-SNE 

6. Compare the visualization methods
