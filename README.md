
Final Project Report - Ben Beiter, Roghaiyeh Ansari


## Replication of the Paper: System Identification Using Artificial Neural Networks

A significant part of mechanical and electrical engineering is using differential equations to create mathematical models of how real, dynamic systems behave. A dynamic system refers to the general formulation of the equations of motion that define how something moves in the real world. There are three parts to such equations: states (referred to as x), inputs (referred to as u), and outputs (referred to as y), which are all related by differential equations. Controls engineering is then the study of how to choose the inputs u to a system in order to get a desired output y. This process of creating a controller relies on knowing what the differential equations that define a system are, however, for many systems these equations are unknown. The process of finding the equations of motion of an unknown system is called system identification. There are well-established processes to perform system identification, but they often rely on the assumption that the system is linear. For complex, nonlinear systems it is very hard to create an accurate mathematical model.

Recently, neural network research has gained increasing attention, particularly for its ability to learn and reconstruct complex nonlinear mappings. This shows promise for use in system identification, and in the paper chosen, “System Identification Using Artificial Neural Networks”[1],  the author proposes the use of a neural network to do system identification for a non-linear system. Specifically, they propose a Multi-Layer Perceptron (MLP) to predict the output, y, of a system for a given input, u. The paper verified this in simulation by training an MLP on a particular nonlinear system and then comparing the output of the simulated system with the predicted output and showing that they are approximately the same. 

The first step in replicating the results from this paper was to write an algorithm to train an MLP Neural Network for our purposes. The main difference between this algorithm and a standard MLP algorithm is that the final output y=f(x), should be a continuous output with a range of all real numbers, so there is no squashing function on this output. Otherwise the algorithm is a standard MLP. It is trained using back-propagation and gradient descent to minimize the output error. 

The second step was to generate training data and choose what would be used as the input features to the neural network. To generate training data we define a system with a nonlinear differential equation and simulate it’s response given a series of predefined inputs. This gives a known input-output relationship of the system to train the MLP with. At any time, the output, y, of a dynamic system depends on its current state, x, and current input, u, so the input features used to train the MLP were chosen to be the state and the input at every timestep. The output of the MLP is the predicted output of the system, y.

Finally, the last step was to choose the parameters that defined the structure and training of the neural network that would be used to model the system. These parameters include the number of hidden layers, the number of nodes in those layers, the learning rate, the regularization parameter, and the number of training epochs for the gradient descent. Additionally, the initial values of the MLP weights were determined randomly.

After training we simulated the system again and used the new data generated as testing data to evaluate the accuracy of the MLP’s prediction


### Results
The paper being replicated [1] gave the following plot as a result to show that their MLP system identification worked.
<p align="center">
  <img width="460" height="300" src="https://github.com/bbeiter1/MLE5_Ansari_Betir/blob/master/Report%20Figures/PaperPlot.JPG">
</p>
This plot shows the predicted output of the system approximately tracking the real output of the system despite noise present in the system. To replicate this result we trained a separate MLP to predict the output of a first-order non-linear system and the test of the MLP gave the following result. 
<p align="center">
  <img width="460" height="300" src="https://github.com/bbeiter1/MLE5_Ansari_Betir/blob/master/Report%20Figures/Fig_result.jpg">
</p>
This verifies the results of the paper, confirming that a MLP can predict the output of a non-linear system fairly accurately. However, there were a couple issues we found that were not explored in the original paper. The most impactful of which is that the initial values of the MLP weights can have a significant impact on the accuracy of the MLP. Training the MLP on the same data will yield different results based on the initial values of the weights. The image above was a good result, however, the following plot shows a particularly poor result. 
<p align="center">
  <img width="460" height="300" src="https://github.com/bbeiter1/MLE5_Ansari_Betir/blob/master/Report%20Figures/Fig_Poor.jpg">
</p>
Very poor results like this were not very common, but their possibility indicates that a more rigorous way of choosing the initial weight values is needed to be able to use an MLP for reliable system identification. 

Another result of interest is the effect of the other neural network parameters. We trained the MLP with 5 hidden layers, each with 5 nodes, a learning rate of a = 0.001, and a regularization parameter of b = 0.02. The training is successful for any values of these parameters, but different values have varying effects on the performance of the MLP. An optimization could be performed to find the parameters that would maximize the accuracy of the trained MLP, however, such an optimization would likely only apply to the specific model being identified. Alternatively a strategy similar to what was shown in [2] could be adopted. In this paper the authors augment the neural network training algorithm to adaptively change its own size to better model the behavior of the system. The network size directly reflects the capacity of the neural network to approximate an arbitrary function, so having a properly sized network will help avoid overfitting and underfitting.

An additional topic that the paper did not address is the number of training epochs to use in training the MLP. This number is the parameter with the most impact on the length of time it takes to train the MLP, however it is also directly related to the accuracy of the MLP. Choosing the number of training epochs will always be a trade-off between the MLP accuracy and computational cost. The following figure illustrates this relationship between the number of training epochs and the total prediction error of the MLP over the training data.
<p align="center">
  <img width="460" height="300" src="https://github.com/bbeiter1/MLE5_Ansari_Betir/blob/master/Report%20Figures/Fig_Loss.jpg">
</p>
A final observation about the paper is that the result, predicting the output of a system, is not all that is needed to create a controller for the system. The trained neural network is a model of the system, but it is not a model in a form that is used to design a controller. However the neural network model can be transformed into a model that can be used to design a controller. The authors in [3] show how to extract meaningful system parameters from a neural network model. That is the last step to using a learned neural network model to design a controller for an unknown system.

All the code used in this replication study is available in this repository, with comments to sufficiently explain its functionality. Running the matlab file ‘MLP_sysID.m’ will run the replication process shown here.

### References

[1] [K.J. Nidhil, S. Sreeraj, B. Vijay y V. Bagyaveereswaran, “System identification using artificial neural network”, Circuit, Power and Computing Technologies (ICCPCT), 2015 International Conference, Nagercoil, 2015.](https://ieeexplore.ieee.org/document/7159360) 

[2] [Mekki, Hassen, and Mohamed Chtourou. "Variable structure neural networks for real-time approximation of continuous-time dynamical systems using evolutionary artificial potential fields." submitted (2012).](https://www.semanticscholar.org/paper/Variable-Structure-Neural-Networks-for-Real-Time-of-Mekki-Chtourou/92c8286aa0d8c6161072b9c516add292eb36de32)

[3] [T. A. Tutunji, “Parametric system identification using neural networks,” Appl. Soft Comput. J., vol. 47, pp. 251–261, 2016 ](https://www.sciencedirect.com/science/article/pii/S1568494616302137)

# this is in master