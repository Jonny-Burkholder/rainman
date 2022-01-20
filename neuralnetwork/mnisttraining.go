package neuralnetwork

import (
	"math/rand"
	"time"

	"github.com/Jonny-Burkholder/neural-network/mnist"
)

//TrainMnist specifically just trains the network to read mnist data
func (n *Network) TrainMnist() {
	data, err := mnist.ReadTrainSet("./mnist/dataset")
	if err != nil {
		panic(err)
	}

	//seed random so that we actually get random numbers
	rand.Seed(time.Now().UnixNano())

	//let's shuffle our index numbers
	randomIndex := make([]int, data.N)
	for i, num := range rand.Perm(data.N) {
		randomIndex[i] = num
	}

	//let's set some variables
	var avgErr float64 = 1000
	i := 0
	chunkSize := n.Config.StochasticMax

	//run until either we run out of data, or the config file tells us to stop
	for i*chunkSize < data.N && i < n.Config.MaxSteps && avgErr > n.Config.TrainingCondition {
		//let's grab the index numbers for this chunk
		startIndex := chunkSize * i
		endIndex := startIndex + chunkSize
		chunk := randomIndex[startIndex:endIndex]

		//let's train through a chunk of data using that random index slice
		for _, j := range chunk {
			//first, we have to convert the mnist data into something we can read
			input := unpackExample(data.Data[j].Image)
			out := n.ForwardFeed(input)
			expected := make([]float64, len(out))
			expected[data.Data[j].Digit] = 1.0
			cost, prime := n.CostFunction.cost(out, expected)
			n.Backpropagate(prime)
			avgErr += cost
		}

		//it is *average* error, after all
		avgErr /= float64(chunkSize)

	}

}

//this is going to be expensive. Whoops
func unpackExample(e [][]uint8) []float64 {
	res := make([]float64, len(e)*len(e[0]))

	k := 0

	for i := 0; i < len(e); i++ {
		for j := 0; j < len(e[0]); j++ {
			res[k] = float64(e[i][j]) //I should really bitshift this
			k++
		}
	}
	return res
}

//TestMnist will test a fully-trained network to see how much of the test
//data it can correctly recognize
func TestMnist() {
	//do stuff and print stuff here
	//good news is, there's no need to randomize anything
	//don't even need to return, we'll just track everything
	//and print from here
}
