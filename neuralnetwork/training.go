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
		example := t.Examples[i]
		//now we're in the training example, so we'll set some
		//conditions and get to training
		//**	**		**		**
		//the epoch stores all the indices of the training inputs
		//in a discrete manner for pseudo-random traversal

		epoch := make([]int, len(example.Data[0].Inputs)) //they should all be the same length... ideally
		for j, num := range rand.Perm(len(epoch)) {
			epoch[j] = num
		}
		//chunksize is how many training examples we'll train in each pass
		chunksize := n.Config.StochasticMax
		//let's check to make sure our chunk size is smaller than the epoch, shall we?
		if chunksize > len(epoch) {
			chunksize = len(epoch)
		}
		//now let's rip through some training data
		var avgCost float64
		cost := make([]float64, len(n.OutputLayer.Outputs))
		prime := make([]float64, len(cost))
		currentErr := 100.00 //*hopefully* this is bigger than everyone's training condition
		k := 0
		iter := 1
		for k*chunksize <= len(epoch) && k <= n.Config.MaxSteps && currentErr > n.Config.TrainingCondition {
			for m := k * chunksize; m < iter*chunksize; m++ {
				res := n.ForwardFeed(example.Data[epoch[m]].Inputs)
				c, p := n.CostFunction.Cost(res, example.Data[epoch[m]].Expected)
				currentErr = averageCost(cost)
				avgCost += currentErr
				for l := range c {
					cost[l] += c[l]
					prime[l] += p[l]
				}
			}
			n.Backpropagate(smush(float64(chunksize), prime))
			k++
			iter++
		}
		//something broke the loop, hurrah! Now that we've got our error data, let's backprop
		avgCost /= float64(iter)
	}

}

func smush(n float64, val []float64) []float64 {
	for i := 0; i < len(val); i++ {
		val[i] /= n
	}
	return val
}
