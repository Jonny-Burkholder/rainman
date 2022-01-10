package mnist

import "github.com/Jonny-Burkholder/neural-network/neuralnetwork"

func main() {

	config := neuralnetwork.NewDefaultConfig()

	nn, err := neuralnetwork.NewNetwork(config, 28*28, 40, 10)

	if err != nil {
		panic(err)
	}

}
