#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>

using namespace std;

struct Connection
{
    double weight;
    double dataWeight;
};

class Neuron { };

typedef vector<Neuron> Layer;

// ************************ class Neuron ***********************

class Neuron
{
    public:
        Neuron(unsigned numOuputs, unsigned myIndex);
        void setOutputVal(double val) { m_outputVal = val; }
        double getOutputVal(void) const { return m_outputVal; }
        void feedForward(const Layer &prevLayer);
        void calOutputGradients(double targetVal);
        void calHiddenGradients(const Layer &nextLayer); 
        void updateInputWeights(Layer &prevLayer);   
    private: 
        static double eta; // [0.0 .. 1.0] overall net training rate
        static double alpha;  // [0.0 .. n] muliplier of last weight change (momentum) 
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX); }
        double sumDOW(const Layer &nextLayer) const;
        double m_outputVal;
        vector<Connection> m_outputWeights;
        unsigned m_myIndex;
        double m_gradient;
};

double Neuron::eta = 0.15; // overall net learing rate
double Neuron::alpha = 0.5; // momentum multipler of last deltaWeight 

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // the weights to be updated are in the Connection container 
    // in the neurons in the preceding layer 

    for(unsigned n = 0; n < prevLayer.size(); ++ n) 
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta 
                    * neuron.getOutputVal()
                    * m_gradient
                    // Also add momentum = a fraction of the previous delta weight
                    + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight; 
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const 
{
    double sum = 0.0;

    /// SUm our contribution of the errors at the nodes we feed

    for(unsigned n = 0; n < nextLayer.size() - 1; ++ n)
    {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal); 
} 

void Neuron::calOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);   
}

double Neuron::transferFunction(double x) 
{   
    // tanh - output range [-1.0 ... 1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x*x;
}

void Neuron::feedForward(const Layer &prevLayer)
{   
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++ n) {
        sum += prevLayer[n].getOutputVal() 
                * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFuntcion(sum);
}

Neuron::Neuron(unsigned numOuputs, unsigned myIndex)
{
    for(unsigned c = 0; c < numOuputs; ++ c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;  
}

// ************************ class Net **************************
class Net 
{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVals) const;
    private: 
        vector<Layer> m_layers; // m_layers[layerNum][neuroNum]
        double m_error;
        double m_recentAverageError;
        double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double> &resultVals) const
{
     resultVals.clear();

     for (unsigned n = 0; n < m_layers.back().size() - 1; ++ n)
     {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
     }
}

void Net::backProp(const vector<double> &targetVals)
{
    // calculate overall net error (RMS of output neuron errors)
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for(unsigned n = 0; n < outputLayer.size() - 1; ++ n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement
    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
                            / (m_recentAverageSmoothingFactor + 1);
    
    // calculate output layer gradients 
    for (unsigned n = 0; n < outputLayer.size() - 1; ++ n) 
    {
        outputLayer[n].calOutputGradients(targetVals[n]);
    }

    // calculate gradients on hidden layers
    for(unsigned layerNum = m_layer.size() - 2; layerNum > 0; --layerNum) 
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
         
        for(unsigned n = 0; n < hiddenLayer; ++ n) 
        {
            hiddenLayer[n].calHiddenGradients(nextLayer);
        }
    }

    // for all layers from outputs to first hidden layers, update connection weights
    for (unsigned layerNum = m_layers.size()-1; layerNum > 0; --layerNum
    {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1]; 
        for(unsigned n = 0; n < Layer.size(); ++ n) 
        {
            layer[n].updateInputWeights(prevLayer); 
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{   
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++ i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    } 

    // Forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++ layerNum) 
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++ n) 
        {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        for(unsigned neuroNum = 0; neuroNum <= topology[layerNum]; ++ neuroNum)
        {
            m_layers.back().push_back(Neuron(numOutputs, neuroNum));
            cout << "Made a Neuron" << endl;
        }
    }
}

int main()
{
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);


    Net myNet(topology);

    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultVals;
    myNet.getResults(resultVals);   
}