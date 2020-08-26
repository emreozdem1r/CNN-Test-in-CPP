#ifndef CONV2D_H
#define CONV2D_H
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;
class Conv2D
{
    public:
        Conv2D();
        float getLayerWeights(int i, int j, int m, int n);
        void setLayerArrayAllocate(int a, int b, int c, int d);
        void setLayerWeights(string fileName, int x, int y, int z, int t);
        void setBias(string fileName, int size);
        float getBias(int i);
        virtual ~Conv2D();

    protected:

    private:
        float *bias;

        float ****layerWeights;
};

#endif // CONV2D_H
