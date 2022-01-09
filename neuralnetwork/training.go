package neuralnetwork

import (
	"math/rand"
	"time"
)

type TrainingInput struct {
	Inputs   []float64
	Expected []float64
}

type TrainingExample struct {
	Data []*TrainingInput
}

//yes, the names are getting confusing
type TrainingData struct {
	//for instance, if this is mnist, there should be 10 of these
	Examples []*TrainingExample
}

//Train feeds training data through the nn and calculates the cost,
//backpropogating that through the network. For now, I'm going to
//strictly be using stochastic gradient descent
func (n *Network) Train(t *TrainingData) {
	rand.Seed(time.Now().UnixNano())

	//now we nest the crap outta some loops, you know how I do
	for i := range t.Examples {
		for j := range t.Examples[i].Data {
			//now we're in the training example, so we'll set some
			//conditions and get to training
			//**	**		**		**
			//the epoch stores all the indices of the training inputs
			//in a discrete manner for pseudo-random traversal
			epoch := make([]int, len(t.Examples[i].Data[j].Inputs))
			for k, num := range rand.Perm(len(epoch)) {
				epoch[k] = num
			}
			//chunksize is how many training examples we'll train in each pass
			chunksize := n.Config.StochasticMax
			//let's check to make sure our chunk size is smaller than the epoch, shall we?
			if chunksize > len(epoch) {
				chunksize = len(epoch)
			}
			//now let's rip through some training data
			currentErr := 100.00 //*hopefully* this is bigger than everyone's training condition
			iter := 0
			l := 1
			for l*chunksize <= len(epoch) && iter <= n.Config.MaxSteps && currentErr > n.Config.TrainingCondition {
				//train the things!
			}
		}
	}

}
