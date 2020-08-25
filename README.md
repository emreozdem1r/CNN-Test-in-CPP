# CNN-Test-in-CPP

- **Fully Connected Layer**

First of all you should a train with FashionMnist dataset:
>You can find this train code in the fully connected file.

After you did, you should save the train's weights.
>also it is below the train code 

After that you should take this weights. To do this, use the h5_read file to convert text file.

After the converting text file you should split the weights of all the layer.
>For example: Input 28 * 28 and first neuron size 10 then
first 10 weights is bias weights and after 28 * 28 * 128 is first layer neuron weights.

CPP code is in the Keras_1D_2D Test in Cpp file. 

In the end it should be compare that whether it is working correctly. 
To do this, there is a test python code in the fully connected file. Here you should be carefull that
there is not activation function in the layer.

Now, run two of code and compare. 
