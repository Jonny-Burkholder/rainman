package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func main() {

	nn, err := neuralnetwork.NewNetwork("test_network", config, 16, 8, 4)

	if err != nil {
		panic(err)
	}

	rand.Seed(time.Now().UnixNano())

	tdata.Data = make([]*neuralnetwork.TrainingInstance, 50)
	tdata.Cost = make([]float64, 1)
	tdata.CostPrime = make([]float64, 1)

	for i := 0; i < 50; i++ {
		tdata.Data[i] = &neuralnetwork.TrainingInstance{}
		tdata.Data[i].Inputs = make([]float64, 16)
		tdata.Data[i].Expected = make([]float64, 4)
		tdata.Data[i].Expected[2] = 1
		for j := range tdata.Data[i].Inputs {
			tdata.Data[i].Inputs[j] = rand.Float64()
		}
	}

	set.Data = make([]*neuralnetwork.TrainingData, 1)
	set.Data[0] = &tdata

	s, _ := nn.Activate(set.Data[0].Data[0].Inputs)

	fmt.Println(s)

	fmt.Println(nn.Train(&set))

}

var config = neuralnetwork.NewDefaultConfig()

var set = neuralnetwork.TrainingSet{}

var tdata = neuralnetwork.TrainingData{}
