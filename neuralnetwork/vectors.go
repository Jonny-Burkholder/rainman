package neuralnetwork

import "errors"

//This should probably be in a more general calculus package in pkg, but that's for a refactor

var nilVector = &Vector{}
var errInsufficientVectors = errors.New("Insufficient number of vectors to perform this operation")
var errUnmatchedVectorDimensions = errors.New("Dimensions of vectors does not match")
var errNoCrossProduct = errors.New("Cross product of these vectors is unknown")

//Vector is an n dimensional line segment representing a movement throughout a real or imagined space
//It may be overkill to define a new type for what is basically a slice of floats, but I think it'll be
//worth it for the method wrapping
type Vector struct {
	Data []float32
}

func NewVector(data ...float32) *Vector {
	return &Vector{
		Data: data,
	}
}

//checkDimensions takes a slice of vectors and verifies that they are all operating in the same dimension
func checkDimensions(v []*Vector) (int, error) {
	l := len(v[0].Data)
	for i := 0; i < len(v); i++ {
		if len(v[i].Data) != l {
			return 0, errUnmatchedVectorDimensions
		}
	}
	return l, nil
}

//DotProduct multiplies two vectors, returning the dot product
func DotProduct(vectors ...*Vector) (float32, error) {
	//check that all vectors are in the same number of dimensions?
	if len(vectors) < 2 {
		return 0, errInsufficientVectors
	}

	l, err := checkDimensions(vectors)
	if err != nil {
		return 0, err
	}

	var res float32
	for i := 0; i < l; i++ {
		temp := vectors[0].Data[i]
		for j := 1; j < len(vectors); j++ {
			temp *= vectors[j].Data[i]
		}
		res += temp
	}
	return res, nil
}

//CrossProduct does some voodoo, I'm sure
func CrossProduct(a, b *Vector) (*Vector, error) {
	if len(a.Data) != 3 || len(b.Data) != 3 {
		return nilVector, errNoCrossProduct
	}
	res := make([]float32, len(a.Data))

	//find the determinant of the resulting matrix of stacking the vectors
	//I'll do this later

	return &Vector{
		Data: res,
	}, nil
}

//AddVectors returns the sum of two or more vectors
func AddVectors(vectors ...*Vector) (*Vector, error) {
	if len(vectors) < 1 {
		return nilVector, errInsufficientVectors
	} else if len(vectors) < 2 {
		return vectors[0], nil
	}
	//check that they're all in the same dimension
	l, err := checkDimensions(vectors)
	if err != nil {
		return nilVector, err
	}
	res := make([]float32, l)
	//Add each dimension together
	for i := 0; i < l; i++ {
		for j := 0; j < len(vectors); j++ {
			res[i] += vectors[j].Data[i]
		}
	}
	return &Vector{
		Data: res,
	}, nil
}
