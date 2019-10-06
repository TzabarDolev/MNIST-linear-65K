# MNIST-linear-65K
A linear neural network for the MNIST challenge which operates with 65k parameters, achieving 98.21% accuracy.

Deep learning: Warm-up exercise

Created by Tzabar Dolev and Yotam Amitai (https://github.com/yotamitai)

The Model receives as input: 
1) the size of each image, as determined by the number of pixels - 28 · 28 = 784.
2) the number of classes - 10.

Our Model Architecture is built out of 4 fully connected layers where in each of the first three layers the following actions are performed:

1. Linear function
2. ReLU activation
3. Batch normalization

On the last layer a Linear function reduces the number of neurons to the number of classes (10) and a Log Soft-max function evaluates the label of a given image.
The number of neurons in each layer:

• Layer #1: from 784 to 75
• Layer #2: from 75 to 50
• Layer #3: from 50 to 30
• Layer #4: from 30 to 10

All in all our model consists of 64825 parameters. Figure 1 is an illustration of our model’s layers where the initial input was represented as a single node instead of
showing 784.

Our model’s training procedure consists of the following steps:
1. Splitting the 60,000 training images into batches of 150.
2. Running each batch through our model.
3. Calculating Loss with Cross-Entropy for each batch.
4. Optimizing using ADAM.
This whole process was done for 80 epochs with a learning Rate of 0.005625
