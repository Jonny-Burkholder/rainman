package trial

//Messing around with a new implementaion method

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {

	rand.Seed(time.Now().UnixNano())

	n := newNeuron()

	fmt.Println(n)

}

const(
	Stochastic int 100
	BaseRate int .001
)

//Dataset is a collection of training data
type Dataset struct{
	Data []*Data
}

func newDataSet(y, x []float32) *DataSet{}

//Data is a single instance of training data
//More complex data should be in matrix form, not
//whatever this is
type Data struct{
	Y float32
	X float32
}

func newData(y, x float32) *Data{}

//Config is a configuration for the network
type Config struct{
	CurrentRate float32
	CurrentStep float32
}

//Network will just have 1 neuron for now
type Network struct{
	Neuron *Neuron
	Output *Output
	Config *Config
}

//Train uses stochastic gradient descent to
//estimate a linear equation for the network
func (n *Network) Train(d *Dataset) error {
	return nil
}

//Predict takes a float32 independent variable x
//and predicts a depended value y
func (n *Network) Predict(x float32) float32 {
	return n.Neuron.Fire(x) + n.Output.Bias
}

//Since this simple "neural network" is going to be used exclusively for linear regression,
//we can eschew the weight and use a linear equation. BUT since the point of that equation
//is to get a vector, we can further simplify into a vector
type Neuron struct {
	Weight float32
}

func newNeuron() *Neuron {
	return &Neuron{
		Weight: rand.float32(),
	}
}

func (n *Neuron) Fire(x float32) float32{
	return x * n.Weight
}

//Output just does the intercept
type Output struct{
	Bias float32
}

func (o *Output) Fire(f float32) float32{
	return f + o.Bias
}
