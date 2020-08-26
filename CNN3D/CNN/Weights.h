#pragma once

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
using namespace std;
class Weights
{
private:
    float* layer_bias;
    float* layer_weights;

public:
    Weights(){}

    Weights(int firstLayerBiasSize, int firstLayerKernelSize) {

        this->layer_bias = new float[firstLayerBiasSize];
        this->layer_weights = new float[firstLayerKernelSize];
    }
    void setLayerWeight(string fileName) {

        fstream read(fileName);
        string row;
        int i = 0;
        if (read.is_open())
        {
            while (!read.eof())
            {
                read >> row;
                layer_weights[i] = atof(row.c_str());
                i++;
            }
        }
    }
    float getLayerWeight(int index) {

        return this->layer_weights[index];
    }

    void setLayerBias(string fileName) {

        fstream read(fileName);
        string row;
        int i = 0;
        if (read.is_open())
        {
            while (!read.eof())
            {
                read >> row;
                layer_bias[i] = atof(row.c_str());
                i++;
            }
        }
    }
    float getLayerBias(int index) {

        return this->layer_bias[index];
    }


};

