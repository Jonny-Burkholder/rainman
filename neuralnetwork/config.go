package neuralnetwork

type Config struct {
	ActivationType int //we'll have a slice of activation types somewhere. I mean really the only point of having them though is for human benefit, a simple int works just fine
	LearningRate   float64
	BaseStepSize   float64
	MaxSteps       int
	Stochastic     int
}

func NewConfig(a, m, s int, l, b float64) *Config {
	return &Config{
		ActivationType: a,
		LearningRate:   l,
		BaseStepSize:   b,
		MaxSteps:       m,
		Stochastic:     s,
	}
}
