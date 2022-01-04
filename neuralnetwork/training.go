package neuralnetwork

import (
	"fmt"
	"math/rand"
	"time"
)

var NilTrainingSet = &TrainingSet{}

//TrainingInstance is a single instance of training data. So this might
//be a single photo of a cat, or a face, that is "labeled" with the expected
//network output
type TrainingInstance struct {
	Inputs   []float64
	Expected []float64 //expected is what we expect the output of the neural network to be
	//we could probably just give a single int for extpected, for the index of the output
	//neuron that we expect to be fully activated
}

//TrainingData is a collection of instances of a single type of data. So
//This would be a slice of pictures only of cats, or only of faces
type TrainingData struct {
	Data      []*TrainingInstance
	Cost      []float64
	CostPrime []float64
}

//TrainingSet is a collection of training data. This is the data type that will
//by in large be used for the training of the network, as it will hold all
//of the necessary data for forward and backward propogation. It's also possible
//this could be handled from file instead of in memory, since we'll more than
//likely be dealing with large datasets
type TrainingSet struct {
	Data []*TrainingData
}

//LoadTrainingSet loads training data from csv or json
//it's a method because the data must match the network's
//inputs and ouputs. Which, now that I think about it, is a
//pretty glaring weakness in the network
func (n *Network) LoadTrainingSet(path string) (*TrainingSet, error) {
	return NilTrainingSet, nil
}

//Train takes in a training set and forward feeds and
//backpropogates is stochastically through the network. At least,
//that's the default until I learn other gradient descent methods.
//Honestly I'm starting to feel like the distinction between forward
//feeding and backpropogation is largely academic
func (n *Network) Train(t *TrainingSet) {
	var avgErr float64
	var iteration int
	for avgErr > n.Config.TrainingCondition && iteration <= n.Config.MaxSteps { //whichever comes first
		//for each training data
		for _, data := range t.Data {
			//randomly select instances to feed forward into the network
			indexes := n.Stochastic(len(data.Data))
			for _, i := range indexes {
				//add up the cost and cost prime of each of those instances
				res, _ := n.Activate(data.Data[i].Inputs)
				for j := 0; j < len(res); j++ {
					diff := data.Data[i].Expected[j] - res[j]
					data.Cost[j] += diff * diff
					data.CostPrime[j] += diff * 2
				}
			}
			//regressively pass these values up through the network to make adjustments
			n.BackPropogate(data.Cost, data.CostPrime)
			//some may argue to use one trainingdata at a time, I guess we can play around with it
		}
	}
	fmt.Printf("Network successfully trained to %v over %d iterations\n", avgErr, iteration) //I don't remember how to control the precision of floats with printf lol
}

//Stochastic takes an integer l, and returns a slice of indeces bounded between zero and l
//The resultant slice is the fixed length of data points used for stochastic gradient descent,
//and is used to
func (n *Network) Stochastic(l int) []int {
	rand.Seed(time.Now().UnixNano())
	temp := make(map[int]bool)
	res := make([]int, n.Config.Stochastic)
	for i := 0; i < l; i++ {
		//if the number is already in use, loop until a unique number is reached
		for {
			num := rand.Intn(l)
			if _, ok := temp[num]; ok == !true {
				res[i] = num
				temp[num] = true
				break
			}
		}
	}
	return res
}
