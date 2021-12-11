package neuralnetwork

//This should probably be in a more general calculus package in pkg, but that's for a refactor

//Vector is a line with a length, a slope, and a velocity
//Maybe this should be an interface? But probably not
type Vector struct {
	Data []float32
}

func NewVector(data ...float32) *Vector {
	return &Vector{
		Data: data,
	}
}
