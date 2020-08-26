#ifndef WEIGHTS_H
#define WEIGHTS_H
#include <iostream>
using namespace std;
#include <stdlib.h>

class Weights
{
    public:
        Weights(int firstLayerBiasSize, int firstLayerKernelSize, int secondLayerBiasSize, int secondLayerKernelSize);
        void setFirstLayerWeight(string nameOfTxt);
        float getFirstLayerWeight(int index);

        void setSecondLayerWeight(string nameOfTxt);
        float getSecondLayerWeight(int index);

        void setFirstLayerBias(string nameOfTxt);
        float getFirstLayerBias(int index);

        void setSecondLayerBias(string nameOfTxt);
        float getSecondLayerBias(int index);

        virtual ~Weights();
    private:
        float *first_layer_bias;
        float *first_layer_weights;
        float *second_layer_bias;
        float *second_layer_weights;

};

#endif // WEIGHTS_H
