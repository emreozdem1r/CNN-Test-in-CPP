#ifndef NEURON_H
#define NEURON_H


class Neuron
{
    public:
        Neuron();
        void setId(int id);
        int getId();

        void setValue(float w, int id);
        float getValue(int i);
        virtual ~Neuron();

    private:
        int id;
        float  value;
        Neuron* next;
        Neuron* head;

};

#endif // NEURON_H
