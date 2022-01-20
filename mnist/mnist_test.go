package mnist

import (
	"fmt"
	"testing"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

func TestMnist(t *testing.T) {

	config := neuralnetwork.NewDefaultConfig()

	nn, err := neuralnetwork.NewNetwork(config, 28*28, 40, 10)

	if err != nil {
		t.Error(err)
	}

	_ = nn.String()

	data, err := ReadTrainSet("./dataset")
	if err != nil {
		t.Error(err)
	}

	fmt.Println(data.N)      // number of data
	fmt.Println(data.Width)  // image width [pixel]
	fmt.Println(data.Height) // image height [pixel]

	for i := 0; i < 5; i++ {
		printData(data, i)
	}

}

func printData(dataSet *DataSet, index int) {
	data := dataSet.Data[index]
	fmt.Println(data.Digit) // print Digit (label)
	PrintImage(data.Image)  // print Image
}

const (
	trainImages = "./dataset/train-images-idx3-ubyte.gz"
	trainLabels = "./dataset/train-labels-idx1-ubyte.gz"
	testImages  = "./dataset/t10k-images-idx3-ubyte.gz"
	testLabels  = "./dataset/t10k-labels-idx1-ubyte.gz"
)
