#include <iostream>
#include "mnist/mnist_reader.hpp"
#include "cifar10_reader.hpp"
#define MNIST_DATA_LOCATION "./fashion-mnist"
#include <stdlib.h>
#include <math.h>
#include "Neuron.h"
#include "Weights.h"
#include "Conv2D.h"
#include "Conv3D.h"
#define FIRSTLAYERNEURONSIZE 128
#define SECONDLAYERNEURONSIZE 10
#define C2_FIRSTLAYERNEURONSIZE 64
#define C2_SECONDLAYERNEURONSIZE 10
#define FULLYCONNECTED_INPUTSIZE 1024

using namespace std;

float** iki_boyutlu(int rows, int columns)
{
    float** dizi;
    dizi = new float* [rows] {};
    for (int i = 0; i < rows; i++)
    {
        dizi[i] = new float[columns] {};
    }
    return dizi;
}
float** fashion_mnist_oku(int goruntu)
{
    float** input = iki_boyutlu(28, 28);
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    for (int row = 0; row < 28; row++)
    {
        for (int column = 0; column < 28; column++)
        {
            input[row][column] = unsigned(dataset.test_images[goruntu][(row * 28 + column)]);
        }
    }
    return input;
}

float*** uc_boyutlu(int rows, int columns, int depth)
{
    float*** dizi;
    dizi = new float** [rows] {};
    for (int i = 0; i < rows; i++)
    {
        dizi[i] = new float* [columns] {};
        for (int j = 0; j < columns; j++)
        {
            dizi[i][j] = new float[depth] {};
        }
    }
    return dizi;
}

float*** cifar_10_oku(int goruntu)
{
    float*** input = uc_boyutlu(32, 32, 3);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    int r, g, b;
    for (int row = 0; row < 32; row++)
    {
        for (int column = 0; column < 32; column++)
        {
            r = unsigned(dataset.training_images[goruntu][(row * 32 + column)]);
            g = unsigned(dataset.training_images[goruntu][(row * 32 + column) + 1024]);
            b = unsigned(dataset.training_images[goruntu][(row * 32 + column) + 2048]);
            input[row][column][0] = r;
            input[row][column][1] = g;
            input[row][column][2] = b;
        }
    }
    return input;
}
void maxPooling(float ***input, float ***output,int neuronSize, int height, int width)
{
    //cout<<"noron: "<<neuronSize<<"height:"<<height<<"width:"<<width<<endl;
    int bbb = 0, ccc = 0;
    for(int a=0; a<neuronSize; a++)   /// input derinlik
    {
        bbb=0;
        for(int b=0; b<height; b=b+2)  /// input height
        {
            ccc=0;
            for(int c=0; c<width; c=c+2)   /// input width
            {
                float buyuk = input[b][c][a];
                for(int bb=0; bb<2; bb++)
                {
                    for(int cc=0; cc<2; cc++)
                    {
                        if(input[b+bb][c+cc][a] > buyuk)
                        {
                            buyuk = input[b+bb][c+cc][a];
                        }
                    }
                }
                output[bbb][ccc][a] = buyuk;
                ccc++;
            }
            bbb++;
        }
    }

}
float*** arrayAllocate(int a, int b, int c)
{
    float ***input;
    input = new float**[a];

    for(int i=0; i<a; i++)
    {
        input[i]= new float*[b];

        for(int j=0; j<b; j++)
        {
            input[i][j]=new float[c];
        }
    }
    return input;
}
float**** arrayAllocate3d(int a, int b, int c, int d)
{
    float ****input;
    input = new float***[a];

    for(int i=0; i<a; i++)
    {
        input[i]= new float**[b];

        for(int j=0; j<b; j++)
        {
            input[i][j] = new float*[c];

            for(int k=0;k<c; k++){
                input[i][j][k]=new float[d];
            }
        }
    }
    return input;
}
float* flatten(float*** input, int width, int height, int depth)
{
    int arraySize = width * height * depth;
    float* flattenArray = new float[arraySize];
    int index = 0;
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            for(int k=0; k<depth; k++)
            {
                flattenArray[index++] = input[i][j][k];
            }
        }
    }
    return flattenArray;
}
float* flatten3d(float**** input, int width, int height, int depth, int frame)
{
    int arraySize = width * height * depth * frame;
    float* flattenArray = new float[arraySize];
    int index = 0;
    for(int i=0; i<width; i++)
    {
        for(int j=0; j<height; j++)
        {
            for(int k=0; k<depth; k++)
            {
                for(int t=0; t<frame; t++)
                {
                    flattenArray[index++] = input[i][j][k][t];
                }
            }
        }
    }
    return flattenArray;
}
void convolution(Conv2D conv, float ***input, float ***output,int neuronSize, int width, int height, int kernelWidth, int kernelHeight, int kernelDepth){

for(int m=0; m<neuronSize; m++)    /// noron sayısı
    {
        for(int a=1; a<width-1; a++)  /// katmana gelen veri Height
        {
            for(int b=1; b<height-1; b++)  /// katmana gelen veri Width
            {
                float sum =0;
                for(int d=0; d<kernelWidth; d++)   /// noron height
                {
                    for(int e=0; e<kernelHeight; e++) /// noron width
                    {
                        for(int f=0; f<kernelDepth; f++)  /// noron derinlik
                        {
                            sum += input[a-1+d][b-1+e][f] * conv.getLayerWeights(d,e,f,m);
                        }
                    }
                }
                sum +=conv.getBias(m);
                output[a-1][b-1][m] = sum;
            }
        }
    }

}
int main()
{

    /*
//HERE IS MNIST TEST CODE
        float** image = fashion_mnist_oku(0);

        Weights weights(128, 100352, 10, 1280);
        //128:      FIRST LAYER NEURON SIZE
        //100352:   FIRST LAYER WEIGHTS SIZE
        //10:       SECOND LAYER NEURON SIZE
        //1280      SECOND LAYER WEIGHTS SIZE

        weights.setFirstLayerBias("bias_1.txt");
        weights.setSecondLayerBias("bias_2.txt");
        weights.setFirstLayerWeight("first_weights.txt");
        weights.setSecondLayerWeight("second_weights.txt");


        float tempArray[128] = {0},    tempArray2[10] = {0};
        Neuron firstLayerNeuron;

        int counter=0, index;

        for(int id = 0; id < FIRSTLAYERNEURONSIZE; id++) /// first layer neuron size
        {
            counter = id;
            index = 0;
            firstLayerNeuron.setId(id);
            for(int i = 0; i < 28; i++)
            {
                for(int j = 0; j < 28; j++)
                {
                    tempArray[id] = tempArray[id] + (image[i][j] *
                                                     weights.getFirstLayerWeight((counter + index*FIRSTLAYERNEURONSIZE)));
                    index++;
                }
            }
            tempArray[id] +=  weights.getFirstLayerBias(id);
            firstLayerNeuron.setValue(tempArray[id], id);
        }

        Neuron secondLayerNeuron;

        for(int id = 0; id < SECONDLAYERNEURONSIZE; id++){ /// second layer neuron size

            counter = id;
            index = 0;
            for(int j = 0; j < FIRSTLAYERNEURONSIZE; j++)
            {
               tempArray2[id] = tempArray2[id] + (firstLayerNeuron.getValue(j) *
                                                  weights.getSecondLayerWeight((counter + index*SECONDLAYERNEURONSIZE)));
               index++;
            }
            tempArray2[id] += weights.getSecondLayerBias(id);
            secondLayerNeuron.setId(id);
            secondLayerNeuron.setValue(tempArray2[id], id);
        }
        for(int i = 0;i<SECONDLAYERNEURONSIZE;i++)
            cout<<secondLayerNeuron.getValue(i)<<endl;
*/


    //CONV2D KOD CODE

    //3*3 KERNEL MATRIX
    //3 INPUT DEPTH
    //32 CURRENT LAYER DEPTH

    ///FIRST CONVOLUTION LAYER
    Conv2D conv;

    conv.setLayerArrayAllocate(3, 3, 3, 32);

    conv.setLayerWeights("conv_kernel1.txt", 3, 3, 3, 32);

    conv.setBias("conv_bias1.txt", 32);

    float*** resim = cifar_10_oku(7);
    //float result[30][30][32];
    float ***result = arrayAllocate(30, 30, 32);
    convolution(conv, resim, result, 32, 32, 32, 3, 3, 3);

    /// MaxPooling
    //float maxResult[15][15][32];    /// MaxPooling
    float ***maxResult = arrayAllocate(15, 15, 32);

    maxPooling(result, maxResult, 32, 30, 30);

    //3*3 kernel matrisi
    //32 önceki katmanın derinliği
    //64 şu anki katmanın derinliği

    ///SECOND CONVOLUTION LAYER
    Conv2D conv2;
    conv2.setLayerArrayAllocate(3, 3, 32, 64);
    conv2.setLayerWeights("conv_kernel2.txt", 3, 3, 32, 64);
    conv2.setBias("conv_bias2.txt", 64);

    //float result_2[13][13][64];
    float ***result_2 = arrayAllocate(13, 13, 64);

    convolution(conv2, maxResult,result_2,64,15,15,3,3,32);

    /// MaxPooling
    float ***maxResult_2 = arrayAllocate(6, 6, 64);

    maxPooling(result_2, maxResult_2, 64, 12, 12);

    //float result_3[4][4][64];
    float ***result_3 = arrayAllocate(4, 4, 64);

    ///THIRD CONVOLUTION LAYER
    Conv2D conv3;
    conv3.setLayerArrayAllocate(3, 3, 64, 64);
    conv3.setLayerWeights("conv_kernel3.txt", 3, 3, 64, 64);
    conv3.setBias("conv_bias3.txt", 64);

    convolution(conv3, maxResult_2, result_3,64,6,6,3,3,64);


    ///FULLY CONNECTED LAYER
        float* flattenArray = flatten(result_3, 4,4,64);


        Weights convWeights(64, 65536, 10, 640);
        convWeights.setFirstLayerBias("fully_bias_1.txt");
        convWeights.setFirstLayerWeight("fully_kernel_1.txt");
        convWeights.setSecondLayerBias("fully_bias_2.txt");
        convWeights.setSecondLayerWeight("fully_kernel_2.txt");

        float firstTemp[64] = { 0 } , secondTemp[10] = { 0 };


        Neuron firstLayerNeuron;

        int counter=0, index;
        ofstream yazz("yaz.txt");
        for(int id = 0; id < C2_FIRSTLAYERNEURONSIZE; id++)
        {
            counter = id;
            index = 0;
            firstLayerNeuron.setId(id);
            for(int i = 0; i < FULLYCONNECTED_INPUTSIZE; i++)
            {
                firstTemp[id] = firstTemp[id] + (flattenArray[i] *
                                                     convWeights.getFirstLayerWeight((counter + index*C2_FIRSTLAYERNEURONSIZE)));
                //yazz<< flattenArray[i]<<"   "<<convWeights.getFirstLayerWeight((counter + index*C2_FIRSTLAYERNEURONSIZE))<<endl;
                index++;
            }
            firstTemp[id] +=  convWeights.getFirstLayerBias(id);
            firstLayerNeuron.setValue(firstTemp[id], id);

        }

        Neuron secondLayerNeuron;

        for(int id = 0; id < C2_SECONDLAYERNEURONSIZE; id++){

            counter = id;
            index = 0;
            for(int j = 0; j < C2_FIRSTLAYERNEURONSIZE; j++)
            {
               secondTemp[id] = secondTemp[id] + (firstLayerNeuron.getValue(j) *
                                                  convWeights.getSecondLayerWeight((counter + index*C2_SECONDLAYERNEURONSIZE)));
               index++;
            }
            secondTemp[id] += convWeights.getSecondLayerBias(id);
            secondLayerNeuron.setId(id);
            secondLayerNeuron.setValue(secondTemp[id], id);
            cout<<secondTemp[id]<<endl;
        }


}
