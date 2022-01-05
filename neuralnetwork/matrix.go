package neuralnetwork

//this is a temporary page to hold matrices and stuff
//until we port over Scottie's code

type MatrixThing interface{} //empty because I'm lazy and this is being replaced

//matrix does what it says on the tin
type Matrix [][]float64

//vector is just a 1d matrix
type Vector []float64

func NewMatrix(a, b int) *Matrix {
	return &Matrix{}
}

func NewVector(length int) *Vector {
	return &Vector{}
}

func DotProduct(a *Vector, b *Matrix) *Vector {
	return &Vector{}
}

func AddMatrix(a, b MatrixThing) *Vector {
	return &Vector{}
}
