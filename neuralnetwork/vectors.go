package neuralnetwork

//This should probably be in a more general calculus package in pkg, but that's for a refactor

//Vector is a line with a length, a slope, and a velocity
//Maybe this should be an interface? But probably not
type Vector struct {
	Length   float32
	Slope    float32
	Velocity float32 //not actually sure how this is measured
}

func NewVector(length, slope, velocity float32) *Vector {
	return &Vector{
		Length:   length,
		Slope:    slope,
		Velocity: velocity,
	}
}
