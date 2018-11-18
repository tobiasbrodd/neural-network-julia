# neural-network-julia
A neural network implemented from scratch in Julia

## Information
This neural network implementation is just a port Julia port of my Python implementation. [Here](https://github.com/tobiasbrodd/neural-network-python) is a link to that repository if you want to learn more about the original implementation. There are two main ways to run the included examples in Julia.
### Run as a script
The following command will run the examples as a script without using the interactive command-line REPL (read-eval-print loop).
```
julia examples.jl
```
**Note**: Running a*examples.jl*s a script means that it needs to call one of the examples, otherwise nothing will be run. This might also take a lot of time as Julia uses JIT compilation and needs to compile everything that's going to be run every time the above command is used. Each run will therefore take approximately 30-60 seconds.

### Run in the REPL
```
julia
include("examples.jl");
```
The first command, *julia* will start the command-line REPL. The second command will include the *examples.jl* file and make it possible to run functions such as *Examples.decision_boundary_example()*. The first time an example is run a lot of code will be compiled and the execution time will be about the same as when running the examples asa script. However, subsequent calls to the same functions will take significantly less time since everything will already have been compiled once.

**Note:** Changing the code will need *include("examples.jl");* to be run again.

## How To
The file *examples* currently includes four neural network examples:
* **exponential_sequence_example** - NN applied on an exponential sequence
* **normal_sequence_example** - NN applied on a normally distributed sequence
* **random_sequence_example** - NN applied on a random sequence
* **decision_boundary_example** - Displays the decision boundary for a 2D sequence

### Neural network arguments
It is possible to change a number of arguments for the neural network:
* Input layer nodes
* Hidden layers and hidden nodes
* Output layer nodes
* Training iterations

#### Input layer nodes
By changing the number of input layer nodes you change how many inputs there are. For example, all sequences/series are 1D so the input layer should have one node. However, moons have *x* and *y* positions so they require two input nodes.

#### Hidden layers and hidden nodes
Hidden layers are in this implementation represented as an array of sizes. By adding a new number to the array you add a new hidden layer. These layers are placed between the input layer and the output layer. Changing a number in the array changes the number of nodes in that hidden layer. The number of layers and nodes required varies between problems and experimentation is often the key to success here.

#### Output layer nodes
By changing the number of output layer nodes you change how many outputs there are. For example, moons have two inputs (*x* and *y* poisitions), but they only have one output (a binary value, *0* or *1*).

#### Training iterations
You can change the number of traing iterations that the neural network should perform when training. The greater the number the more iterations it will perform which in turn hopefully makes the network better at predicting. One important thing to note is that it takes time to traing the neural network. So the greater the number, the longer it will take to train.

### Training
The neural network training function takes two arguments as input: input training data and output training data. Both arguments are specified as Numpy arrays/matrices. Columns are the input features and will be mapped to input layer nodes and rows are just input data entries. For example, the input matrix for moons should be two columns specifying the *x* and *y* positions and the rows should be the moon entries. The same thing applies for the output matrix, though in the moons example it should be a 1D matrix where each row is an expected value for a moon. Important to note is that all input and output should be scaled and only have values between *0* and *1*.

### Predicting
Predicting is basically done as with the training, but without specifying an expected output. Remember to use the same number of features and use the same scale as for the training input. The output will also have to be scale, but this time scaled back to its original scale as the output from the predict function will be numbers in the range *0* to *1*.