#ifndef CONV3D_H
#define CONV3D_H
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

class Conv3D
{
    public:
        Conv3D();
        float getLayerWeights(int i, int j, int m, int n, int t);
        void setLayerArrayAllocate(int a, int b, int c, int d, int e);
        void setLayerWeights(string fileName, int x, int y, int z, int t, int s);
        void setBias(string fileName, int size);
        float getBias(int i);
        virtual ~Conv3D();

    protected:

    private:
         float *bias;

        float *****layerWeights;
};

#endif // CONV3D_H
