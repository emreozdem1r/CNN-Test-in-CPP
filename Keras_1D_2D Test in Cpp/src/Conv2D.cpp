#include "Conv2D.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
using namespace std;
Conv2D::Conv2D()
{
}
float Conv2D::getLayerWeights(int i, int j, int m, int n)
{

    return this->layerWeights[i][j][m][n];
}
void Conv2D::setLayerArrayAllocate(int a, int b, int c, int d)
{
   this->layerWeights=new float***[a];

    for(int i=0; i<a; i++)
    {
        this->layerWeights[i]= new float**[b];

        for(int j=0; j<b; j++)
        {
            this->layerWeights[i][j]=new float*[c];

            for(int k=0; k<c; k++)
            {
                this->layerWeights[i][j][k]=new float[d];
            }
        }
    }
}
void Conv2D::setLayerWeights(string fileName, int x, int y, int z, int t){

    fstream oku(fileName);
    string row;
    if(oku.is_open())
    {
        while(!oku.eof())
        {
            for(int a=0; a<x; a++)
            {
                for(int b=0; b<y; b++)
                {
                    for(int c=0; c<z; c++)
                    {
                        for(int d=0; d<t; d++)
                        {
                            oku >> row;
                            this->layerWeights[a][b][c][d] = atof(row.c_str());
                        }
                    }
                }
            }
        }
    }
}
void Conv2D::setBias(string fileName,int size){

    this->bias = new float[size];
    fstream oku(fileName);
    string row;
    int i = 0;
    if(oku.is_open())
    {
        while(!oku.eof())
        {
            oku >> row;
            this->bias[i] = atof(row.c_str());
            //cout<<this->bias[i];
            i++;

        }
    }
    else{
        cout<<"okunamadı";
    }
}
float Conv2D::getBias(int i){

    return this->bias[i];
}
Conv2D::~Conv2D()
{
    //dtor
}
