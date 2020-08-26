#include "Weights.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
using namespace std;
Weights::Weights(int firstLayerBiasSize, int firstLayerKernelSize, int secondLayerBiasSize, int secondLayerKernelSize)
{
    this->first_layer_bias = new float[firstLayerBiasSize];
    this->first_layer_weights= new float[firstLayerKernelSize];
    this->second_layer_bias = new float[secondLayerBiasSize];
    this->second_layer_weights = new float[secondLayerKernelSize];

    /*this->first_layer_bias = new float[128];
    this->first_layer_weights= new float[100352];
    this->second_layer_bias = new float[10];
    this->second_layer_weights = new float[1280];*/

}
void Weights::setFirstLayerWeight(string nameOfTxt){

    fstream read(nameOfTxt);
    string row;
    int i=0;
    if(read.is_open())
    {
        while(!read.eof())
        {
            read >> row;
            first_layer_weights[i] = atof(row.c_str());
            i++;
        }
    }
}
float Weights::getFirstLayerWeight(int index){

    return this->first_layer_weights[index];
}
void Weights::setSecondLayerWeight(string nameOfTxt){

    fstream read(nameOfTxt);
    string row;
    int i=0;
    if(read.is_open())
    {
        while(!read.eof())
        {
            read >> row;
            second_layer_weights[i] = atof(row.c_str());
            i++;
        }
    }
}
float Weights::getSecondLayerWeight(int index){

    return this->second_layer_weights[index];
}
void Weights::setFirstLayerBias(string nameOfTxt){

    fstream read(nameOfTxt);
    string row;
    int i=0;
    if(read.is_open())
    {
        while(!read.eof())
        {
            read >> row;
            first_layer_bias[i] = atof(row.c_str());
            i++;
        }
    }
    //this->first_layer_bias[index] = w;
}
float Weights::getFirstLayerBias(int index){

    return this->first_layer_bias[index];
}
void Weights::setSecondLayerBias(string nameOfTxt){

    fstream read(nameOfTxt);
    string row;
    int i=0;
    if(read.is_open())
    {
        while(!read.eof())
        {
            read >> row;
            second_layer_bias[i] = atof(row.c_str());
            i++;
        }
    }
}
float Weights::getSecondLayerBias(int index){

    return this->second_layer_bias[index];
}
Weights::~Weights()
{
}
