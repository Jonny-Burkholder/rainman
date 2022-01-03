package neuralnetwork

var costFunctions = []string{"Mean Squared", "Cross-Entropy"} //for human reference

type Config struct {
	ActivationType       int //we'll have a slice of activation types somewhere. I mean really the only point of having them though is for human benefit, a simple int works just fine
	OutputActivationType int //default will be the same as the main activation type
	CostFunction         int
	MaxSteps             int
	Stochastic           int
	LearningRate         float64
	BaseStepSize         float64
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
		BaseStepSize:         b,
		ReluLeak:             r,
	}
}

func NewDefaultConfig() *Config {
	return &Config{
		ActivationType:       0,
		OutputActivationType: 0,
		CostFunction:         0,
		MaxSteps:             100, //no idea if that's a good size
		Stochastic:           100,
		LearningRate:         .001,
		BaseStepSize:         .01,
		ReluLeak:             .001,
	}
}
