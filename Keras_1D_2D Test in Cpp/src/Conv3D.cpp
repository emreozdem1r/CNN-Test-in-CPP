#include "Conv3D.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
using namespace std;

Conv3D::Conv3D()
{
}
float Conv3D::getLayerWeights(int i, int j, int m, int n, int t)
{

    return this->layerWeights[i][j][m][n][t];
}
void Conv3D::setLayerArrayAllocate(int a, int b, int c, int d, int e)
{
    this->layerWeights=new float****[a];

    for(int i=0; i<a; i++)
    {
        this->layerWeights[i]= new float***[b];

        for(int j=0; j<b; j++)
        {
            this->layerWeights[i][j]=new float**[c];

            for(int k = 0; k < c; k++)
            {
                this->layerWeights[i][j][k]=new float*[d];

                for(int t = 0; t < d; t++)
                {
                    this->layerWeights[i][j][k][t] = new float[e];
                }
            }
        }
    }
}
void Conv3D::setLayerWeights(string fileName, int x, int y, int z, int t, int s)
{

    fstream read(fileName);
    string row;
    if(read.is_open())
    {
        while(!read.eof())
        {
            for(int a=0; a<x; a++)
            {
                for(int b=0; b<y; b++)
                {
                    for(int c=0; c<z; c++)
                    {
                        for(int d=0; d<t; d++)
                        {
                            for(int e = 0; e < s; e++)
                            {
                                read >> row;
                                this->layerWeights[a][b][c][d][e] = atof(row.c_str());
                            }
                        }
                    }
                }
            }
        }
    }
}
void Conv3D::setBias(string fileName,int size)
{

    this->bias = new float[size];
    fstream read(fileName);
    string row;
    int i = 0;
    if(read.is_open())
    {
        while(!read.eof())
        {
            read >> row;
            this->bias[i] = atof(row.c_str());
            //cout<<this->bias[i];
            i++;

        }
    }
    else
    {
        cout<<"it couldnt read";
    }
}
float Conv3D::getBias(int i)
{

    return this->bias[i];
}
Conv3D::~Conv3D()
{
}
