# CNN-Test-in-CPP

- **Fully Connected Layer**

First of all you should train with FashionMnist dataset:
>You can find this train code in the fully connected file.

After you did, you should save the train's weights.
>also it is below the train code 

After that you should take this weights. To do this, use the h5_read file to convert text file.

After the converting text file you should split the weights of all the layer.
>For example: Input 28 * 28 and first neuron size is 128 then
first 128 weights is bias weights and after 28 * 28 * 128 is first layer neuron weights.

CPP code is in the Keras_1D_2D Test in Cpp file. 

In the end it should be compare that whether it is working correctly. 
To do this, there is a test python code in the fully connected file. Here you should be carefull that
there is not activation function in the layer.

Now, run two of code and compare results. 

- **Conv2D**

First of all you should train with Cifar10 dataset
>You can find this train code in the conv2d file

After you did, you should save the trains's weights.
>also it is below the train code

After that you should take this weights. To do this, use h5_read file to convert text file. You should read explaination in the file.
Take care here, there are 2 layer in this train. The first is Conv layer, second one is full connected layer. 
If you want Conv layer weights, the function should be like below. 

>weights = getWeightsForLayer("conv2d", 'weights.hdf5')

>AAgetWeightsForLayer("conv2d", 'weights.hdf5')

You can see the functions in the h5_read.py file. If you want to take full connected layer weights, you will change conv2d as dense.

> In the cpp, you should read the weights.
> After the convert text file you should split the weights of all the layer.

For example: Input depth is 3, first conv layer kernel size is 3 * 3 and conv layer depth is 32
So that The first 3 * 3 * 3 * 32 = 864 weigths is first conv layer weigths. The others can be calculated like this.

Full connected layer weights can be calculated like Full Connected Layer.

All is done. Now, You can read a image Cifar10 dataset in cpp project. And apply the read weights to the image.

Now, run two of code and compare results. 
