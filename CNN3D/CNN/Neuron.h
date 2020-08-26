#pragma once

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
using namespace std;
class Neuron
{

private:
    int id;
    float  value;
    Neuron* next;
    Neuron* head;

public:
    Neuron(){ 
        head = NULL; 
    }
    void setId(int id) {
        this->id = id;
    }
    int getId() {
        return this->id;
    }

    void setValue(float v, int id) {
        Neuron* new_neuron = new Neuron;
        new_neuron->id = id;
        new_neuron->value = v;
        new_neuron->next = 0;

        Neuron* temp = head;
        if (temp == NULL)
        {
            head = new_neuron;
        }
        else
        {
            Neuron* first = head;

            while (first->next != NULL)
            {
                first = first->next;
            }
            first->next = new_neuron;
            first->next->next = NULL;
        }
    }
    float getValue(int i) {
        if (head->id == i)
            return head->value;
        else
        {
            Neuron* first = head;
            while (first->next->id != i)
            {
                first = first->next;
            }
            return first->next->value;
        }
    }

};

