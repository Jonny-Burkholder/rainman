package neuralnetwork

type Config struct {
	ActivationType int //we'll have a slice of activation types somewhere. I mean really the only point of having them though is for human benefit, a simple int works just fine
	LearningRate   float32
	MaxSteps       int
}

func NewConfig(a, m int, l float32) *Config {
	return &Config{
		ActivationType: a,
		LearningRate:   l,
		MaxSteps:       m,
	}
}
