
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron;

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;  
    static double alpha; 
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;  
double Neuron::alpha = 0.5;   


void Neuron::updateInputWeights(Layer &prevLayer)
{

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;


    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{

    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}


class Net
{
    public:
        Net()
        {
            cerr << "empty constructor";
        }
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVals) const;
        double getRecentAverageError(void) const { return m_recentAverageError; }
        void saveNet(vector<unsigned> &topology);

    private:    
        vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
        double m_error;
        double m_recentAverageError;
        static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0;

void Net::saveNet(vector<unsigned> &topology)
{
    ofstream outFile;
    outFile.open("net.dat");

    outFile << topology.size() << " ";
    for(int i = 0; i < topology.size(); ++ i)
        outFile << topology[i] << " ";

    for(int i = 0; i < m_layers.size() - 1; ++ i)
        for(int j = 0; j < m_layers[i].size(); ++ j)
            for(int k = 0; k < m_layers[i+1].size() - 1; ++ k)
                outFile << m_layers[i][j].outputWeight[k].weight << " ";

    outFile.close();
}

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        m_layers.back().back().setOutputVal(1.0);
    }
}


// ***************************************************** //

const int numTrainingTest = 6000;

int label[numTrainingTest + 1];
int image[numTrainingTest + 1][30][30];
int image1[numTrainingTest + 1][30][30];

void readTrainData();
void getNet(vector<unsigned> &topology, Net &myNet);

int n, rowCount, colCount;
const int REP = 60000;

#define NEWNET // 

int main()
{
    srand(time(NULL));

    readTrainData();

    Net myNet = Net();
    vector<unsigned> topology;

    #ifdef NEWNET 
        topology.push_back(rowCount * colCount);
        topology.push_back(20);
        topology.push_back(20);
        topology.push_back(10);
        myNet = Net(topology);
    #else
        getNet(topology, myNet);   
    #endif // NEWNET

    vector<double> inputVals;
    vector<double> targetVals;
    vector<double> outputVals;

    for(int x = 0; x < rowCount; ++ X)
        for(int y = 0; y < colCount; ++ y)
            inputVals.push_back(0);
    for(int i = 0; i < 10; ++ i)
        targetVals.push_back(0);

    for(int i = 0; i < REP; ++ i)
    {
        if(i % 1000 == 0) 
            cerr << i << " ";
        int id = rand() * rand() % numTrainingTest;

        // generalize the image
        for(int x = 0; x < rowCount; ++ x)
            for(int y = 0; y < colCount; ++ y)
            {
                inputVals[x*colCount + y] = 1.0*image[id][x][y]/255;
            }
        int rep = rand()% 40;
        for(int j = 0; j < rep; ++ j)
        {
            int pos = rand() % 784;
            int x = rand() % 256;
            inputVals[pos] = 1.0*x/255;
        }

        for(int j = 0; j < targetVals.size(); ++ j)
            targetVals[j] = 0;
        targetVals[label[id]] = 1;

        myNet.feedForward(inputVals);
        myNet.backProp(targetVals);
    }

    myNet.saveNet(topology);

    // testing 

    int numTest = 1000;
    int correct = 0;

    ofstream outFile;
    outFile.open("out.txt");

    for(int i = 0; i < numTest; ++ i)
    {
        int id = rand()*rand() % numTrainingTest;
        for(int x = 0; x < rowCount; ++ x)
            for(int y = 0; y < colCount; ++ y)
                inputVals[x*colCount+y] = 1.0*image[id][x][y]/255;

        myNet.feedForward(inputVals);
        int result = myNet.getResults();

        outFile << "Test #" << i << "\n";
        outFile << "Target: " << label[id] << "\n";
        outFile << "Output: " << result << "\n";
        
        correct += result == label[id];
    }

    outFile << "Accuracy: " << 100.0*correct/numTest << "%";
    outFile.close();
}

void readTrainData()
{
    ifstream inFile;
    inFile.open("training-set/training_input.dat");

    inFile >> n >> rowCount >> colCount;

    for(int i = 0; i < n; ++ i) 
    {
        inFile >> label[i];
        int dx = rowCount - 1, dy = colCount - 1;
        for(int x = 0; x < rowCount; ++ x)
            for(int y = 0; y < colCount; ++ y)
            {
                inFile >> image1[i][x][y];
                image[i][x][y] = 0;
                if(image1[i][x][y]) {
                    dx = min(x, dx);
                    dy = min(y, dy);
                }
            }
        
        for(int x = dx; x < rowCount; ++ x)
            for(int y = dy; y < colCount; ++ y)
                image[i][x-dx][y-dy] = image1[i][x][y];
    }
    inFile.close();
}

void getNet(vector<unsigned> &topology, Net &newNet)
{
    ifstream inFile;
    inFile.open("net.dat");
    topology.clear();
    int n;
    inFile >> n;
    
    for(int i = 0; i < n; ++ i)
    {
        int x;
        inFile >> x;
        topology.push_back(x);
    }

    newNet = Net(topology);
    vector<Layer> &m_layers = newNet.m_layers;
    for(int i = 0; i < m_layers.size()-1; ++ i)
        for(int j = 0; j < m_layers[i].size(); ++ j)
            for(int k = 0; k < m_layers[i+1].size()-1; ++ k)
                m_layers[i][j].outputWeight[k].weight;
    
    inFile.close();
}