package neuralnetwork

var costFunctions = []string{"Mean Squared", "Cross-Entropy"} //for human reference

type Config struct {
	ActivationType       int //we'll have a slice of activation types somewhere. I mean really the only point of having them though is for human benefit, a simple int works just fine
	OutputActivationType int //default for this will be sigmoid
	CostFunction         int
	MaxSteps             int
	Stochastic           int
	LearningRate         float64
	TrainingCondition    float64 //the certainty that the network is aiming for to termincate the training condition
	ReluLeak             float64
}

func NewConfig(a, o, c, m, s int, l, b, r float64) *Config {
	return &Config{
		ActivationType:       a,
		OutputActivationType: o,
		CostFunction:         c,
		MaxSteps:             m,
		Stochastic:           s,
		LearningRate:         l,
		TrainingCondition:    b, //the certainty that the network is aiming for to termincate the training condition
		ReluLeak:             r,
	}
}

func NewDefaultConfig() *Config {
	return &Config{
		ActivationType:       0,
		OutputActivationType: 0,
		CostFunction:         0,
		MaxSteps:             100, //no idea if that's a good size, probably not nearly enough
		Stochastic:           100,
		LearningRate:         .001,
		TrainingCondition:    .01,  //the certainty that the network is aiming for to termincate the training condition
		ReluLeak:             .001, //this is probably stupid and I'll probably get rid of it
	}
}
