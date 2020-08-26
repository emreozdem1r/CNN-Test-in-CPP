
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "Conv3D.h"
#include "Weights.h"
#include "Neuron.h"
#include <omp.h>
using namespace std;

#define FIRSTLAYERSIZE 32
#define SECONDLAYERSIZE 16
#define THIRDLAYERSIZE 16
#define FLATTEN_ARRAY_SIZE 1200


float**** arrayAllocate3d(int a, int b, int c, int d)
{
    float**** input;
    input = new float*** [a];

    for (int i = 0; i < a; i++)
    {
        input[i] = new float** [b];

        for (int j = 0; j < b; j++)
        {
            input[i][j] = new float* [c];

            for (int k = 0; k < c; k++) {
                input[i][j][k] = new float[d];
            }
        }
    }
    return input;
}
void readInput(float**** input, string fileName) {
    fstream oku("c3d_input.txt");
    string row;
    //float input[32][32][10][3];
    if (oku.is_open())
    {
        while (!oku.eof())
        {
            for (int a = 0; a < 32; a++)
            {
                for (int b = 0; b < 32; b++)
                {
                    for (int c = 0; c < 10; c++)
                    {
                        for (int d = 0; d < 3; d++)
                        {
                            oku >> row;
                            input[a][b][c][d] = atof(row.c_str());
                        }
                    }
                }
            }
            break;
        }
    }

}
float* flatten3d(float**** input, int width, int height, int depth, int frame)
{
    int arraySize = width * height * depth * frame;
    float* flattenArray = new float[arraySize];
    int index = 0;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < depth; k++)
            {
                for (int t = 0; t < frame; t++)
                {
                    flattenArray[index++] = input[i][j][k][t];
                }
            }
        }
    }
    return flattenArray;
}
float**** convolution(float**** input, Conv3D conv3d, int noron_size, int frame_size, int  width, int  height, int depth ) {
    int bb = 0;
    int cc = 0;
    int dd = 0;
    float**** result_conv = arrayAllocate3d(width - 4, height - 4, frame_size-2, noron_size);
    //float result_conv[width-4][height-4][frame_size][noron_size];
#pragma omp parallel for schedule(static) shared(input, conv3d)
    for (int a = 0; a < noron_size; a++)  // nöron sayısı
    {
        bb = 0;
        for (int b = 1; b < frame_size - (1); b++)///input frame sayısı 
        {
            cc = 0;
            for (int c = 2; c < height - (2); c++) ///input height
            {
                dd = 0;
                for (int d = 2; d < width - (2); d++)/// input width
                {
                    float top = 0;
                    for (int e = 0; e < 5; e++) /// mask height
                    {
                        for (int f = 0; f < 5; f++) /// mask width
                        {
                            for (int g = 0; g < 3; g++) /// mask  frame depth
                            {
                                for (int h = 0; h < depth; h++) /// mask depth
                                {
                                    top = top + input[c - 2 + e][d - 2 + f][b - 1 + g][h] * conv3d.getLayerWeights(e, f, g, h, a);
                                }
                            }
                        }
                    }
                    top = top + conv3d.getBias(a);
                    //result_conv[cc][dd][bb][a] =top;
                    result_conv[c - 2][d - 2][b - 1][a] = top;

                    dd = dd + 1;
                }
                cc = cc + 1;
            }
            bb = bb + 1;
        }
    }
    return result_conv;
}
float**** maxPooling(float**** result_conv,int noron_size, int frame_size, int width, int height, int mask_depth ) {
    int bb3 = 0;
    int cc3 = 0;
    int dd3 = 0;
    //float max_result[14][14][8][8];
    float**** max_result = arrayAllocate3d(width/2, height/2, frame_size, noron_size);
#pragma omp parallel for schedule(static) shared(result_conv)
    for (int a = 0; a < noron_size; a++)  /// nöron derinliği
    {
        bb3 = 0;
        for (int b = 0; b < frame_size; b++) /// input frame sayısı
        {
            cc3 = 0;
            for (int c = 0; c < height - 1; c = c + 2)       /// input frame height
            {
                dd3 = 0;
                for (int d = 0; d < width - 1; d = d + 2)     /// input frame width
                {
                    float enbuyuk = result_conv[c][d][b][a];
                    for (int bb = 0; bb < mask_depth; bb++)    /// mask frame depth
                    {
                        for (int cc = 0; cc < 2; cc++)  /// mask height
                        {
                            for (int dd = 0; dd < 2; dd++)  /// mask weight
                            {
                                if (result_conv[c + cc][d + dd][b + bb][a] > enbuyuk)
                                {
                                    enbuyuk = result_conv[c + cc][d + dd][b + bb][a];
                                }
                            }
                        }
                    }
                    max_result[cc3][dd3][bb3][a] = enbuyuk;
                    dd3 = dd3 + 1;
                }
                cc3 = cc3 + 1;
            }
            bb3 = bb3 + 1;
        }
    }
    return max_result;
}

int main()
{
    //READ INPUT VIDEOS
    float**** input = arrayAllocate3d(32, 32, 10, 3);
    readInput(input, "c3d_input.txt");

    double start = omp_get_wtime();

    //FIRST CONVOLUTION LAYER 
    Conv3D conv3d;
    conv3d.setLayerArrayAllocate(5, 5, 3, 3, 8);
    conv3d.setLayerWeights("conv3d_kernel_1.txt", 5, 5, 3, 3, 8);
    conv3d.setBias("conv3d_bias_1.txt", 8);

    float**** result_conv = convolution(input, conv3d, 8, 10, 32, 32, 3);
 
    //FIRST MAXPOOLING LAYER  
    float**** max_result = maxPooling(result_conv, 8, 8, 28, 28, 1);
   
    //SECOND CONVOLUTION LAYER
    Conv3D conv2_3d;
    conv2_3d.setLayerArrayAllocate(5, 5, 3, 8, 16);
    conv2_3d.setLayerWeights("conv3d_kernel_2.txt", 5, 5, 3, 8, 16);
    conv2_3d.setBias("conv3d_bias_2.txt", 16);

    float**** result_conv2 = convolution(max_result, conv2_3d, 16, 8, 14, 14, 8);
   
    //SECOND MAX POOING LAYER
    float**** max_result_2 = maxPooling(result_conv2, 16, 3, 10, 10,2);

    // READ THE WEIGHTS
    Weights w1(32, 38400);
    w1.setLayerBias("conv3d_fc_bias_1.txt");
    w1.setLayerWeight("conv3d_fc_kernel_1.txt");

    Weights w2(16, 512);
    w2.setLayerBias("conv3d_fc_bias_2.txt");
    w2.setLayerWeight("conv3d_fc_kernel_2.txt");

    Weights w3(1, 17);
    w3.setLayerBias("conv3d_fc_bias_3.txt");
    w3.setLayerWeight("conv3d_fc_kernel_3.txt");

    // FULLY CONNECTED
    float* flattenArray = flatten3d(max_result_2, 5, 5, 3, 16);


    Neuron firstLayerNeuron;

    float firstTemp[FIRSTLAYERSIZE] = { 0 };
    int counter = 0, index;
    //FIRST NEURAL LAYER
#pragma omp parallel for
    for (int id = 0; id < FIRSTLAYERSIZE; id++) 
    {
        counter = id;
        index = 0;
        firstLayerNeuron.setId(id);
        for (int i = 0; i < FLATTEN_ARRAY_SIZE; i++)
        {
            firstTemp[id] = firstTemp[id] + (flattenArray[i] *
                w1.getLayerWeight((counter + index * FIRSTLAYERSIZE)));
             index++;
        }
        firstTemp[id] += w1.getLayerBias(id);
        firstLayerNeuron.setValue(firstTemp[id], id);
    }


    Neuron secondLayerNeuron;
    float secondTemp[SECONDLAYERSIZE] = { 0 };
    
    //SECOND NEURAL LAYER
#pragma omp parallel for
    for (int id = 0; id < SECONDLAYERSIZE; id++) { 

        counter = id;
        index = 0;
        for (int j = 0; j < FIRSTLAYERSIZE; j++)
        {
            secondTemp[id] = secondTemp[id] + (firstLayerNeuron.getValue(j) *
                w2.getLayerWeight((counter + index * SECONDLAYERSIZE)));
            index++;
        }
        secondTemp[id] += w2.getLayerBias(id);
        secondLayerNeuron.setId(id);
        secondLayerNeuron.setValue(secondTemp[id], id);
    }

    float resultFc = 0;
    //THIRD NEURAL LAYER
    for (int i = 0; i < THIRDLAYERSIZE; i++)
    {
        resultFc += secondLayerNeuron.getValue(i) +
            w3.getLayerWeight(i);
    }
    resultFc += w3.getLayerBias(0);

    //RESULT
    cout <<"RESULT = "<< resultFc << endl;

    double end = omp_get_wtime();

    cout << "Geçen sure: " << end - start << endl;
    /*ofstream xx("vs.txt");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 16; l++) {
                    xx << max_result_2[i][j][k][l] << "  ";
                }
                xx << endl;
            }
        }
        xx << endl;
    }*/

    cout << "tamam" << endl;
    return 0;

}

