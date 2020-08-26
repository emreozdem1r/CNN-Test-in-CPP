#include "Neuron.h"
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
using namespace std;
Neuron::Neuron()
{
    head = NULL;
}

void Neuron::setId(int id)
{
    this->id = id;
}
int Neuron::getId()
{
    return this->id;
}
void Neuron::setValue(float v, int id)
{
    Neuron *new_neuron  = new Neuron;
    new_neuron->id = id;
    new_neuron->value = v;
    new_neuron->next =  0;

    Neuron *temp = head;
    if(temp == NULL)
    {
        head = new_neuron;
    }
    else
    {
        Neuron *first = head;

        while(first->next != NULL)
        {
            first = first->next;
        }
        first->next = new_neuron;
        first->next->next = NULL;
    }

}
float Neuron::getValue(int i)
{
    if(head->id == i)
        return head->value;
    else
    {
        Neuron *first = head;
        while(first->next->id != i)
        {
            first = first->next;
        }
        return first->next->value;
    }
}
Neuron::~Neuron()
{

}
