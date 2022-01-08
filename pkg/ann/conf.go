package ann

import (
	"time"
)

type Config struct {
	// Layers=[]int{8,4,4,2} will create a
	// neural network with an input layer
	// containing 8 neurons, two hidden
	// layers each containing 4 neurons,
	// and an output layer with 2 neurons.
	Layers       []int
	Seed         int64
	IsRegression bool
	Activation   ActivationFn
}

var defaultConfig = &Config{
	Layers:       []int{64, 32, 16, 8},
	Seed:         time.Now().UnixNano(),
	IsRegression: false,
	Activation:   ReLU,
}

func checkConfig(conf *Config) {
	if conf == nil {
		*conf = *defaultConfig
	}
	if len(conf.Layers) < 3 {
		panic("config: layers must contain a minimum of three layers [1 input, 1 hidden, and 1 output]")
	}
	if conf.Activation == nil {
		conf.Activation = ReLU
	}
}
