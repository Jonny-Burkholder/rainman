package neuralnetwork

var costFunctions = []string{"Mean Squared", "Cross-Entropy"} //for human reference

var DefaultConfig = &Config{
	DefaultActivationType: 1,
	OutputActivationType:  0,
	CostFunction:          0,
	DescentType:           1,
	MaxSteps:              1000, //no idea if that's a good size
	StochasticMax:         30,
	ReluCap:               6,    //pretty standard stuff
	LearningRate:          .001, //This will scale down with each iteration
	TrainingCondition:     .01,
	ReluLeak:              .01,
}

//eventually we'll have different gradient descent options like momentum and adam

type Config struct {
	DefaultActivationType int //we'll have a slice of activation types somewhere. I mean really the only point of having them though is for human benefit, a simple int works just fine
	OutputActivationType  int //default for this will be sigmoid
	CostFunction          int
	DescentType           int //algorithm used for gradient descent
	MaxSteps              int //Depending on the size of a given epoch, terminate after so many steps if the epoch has not been exhausted
	StochasticMax         int //maximum number of training values passed in a stochastic gradiant descent
	ReluCap               int //the capped output of the relu function, e.g. relu6
	LearningRate          float64
	TrainingCondition     float64 //the minimum value that the average error must reach to terminate training
	ReluLeak              float64
}

func NewConfig(a, o, c, d, m, s, rc int, l, b, r float64) *Config {
	return &Config{
		DefaultActivationType: a,
		OutputActivationType:  o,
		CostFunction:          c,
		DescentType:           d,
		MaxSteps:              m,
		StochasticMax:         s,
		ReluCap:               rc,
		LearningRate:          l,
		TrainingCondition:     b,
		ReluLeak:              r,
	}
}
