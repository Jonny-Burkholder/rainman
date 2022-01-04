package tools

import "fmt"

const maxSize int = 1000000000
const minSize int = 1

//Compress takes input data that's too large for the network and compresses it into something
//more manageable. I'm assuming this will only be necessary for truly massive pieces of data,
//as the networks should, for the most part, be able to scale to their input
func Compress(inputs []float64, target int) ([]float64, error) {
	if target > maxSize || target < minSize {
		return []float64{}, fmt.Errorf("Try again joker")
	}
	if len(inputs) > target {
		res := make([]float64, target) //n.Size does *not* work here. We'll have to add another argument to the function
		//do stuff
		return res, nil
	}
	return []float64{}, fmt.Errorf("Data is already correct size")
}

//Upscale takes an input that's too small and interpolates intermediate values
func Upscale(inputs []float64, target int) ([]float64, error) {
	if target > maxSize || target < minSize {
		return []float64{}, fmt.Errorf("Try again joker")
	}
	if len(inputs) < target {
		res := make([]float64, target)
		//do magic
		return res, nil
	}
	return []float64{}, fmt.Errorf("Data is already correct size")
}

//QuantizeToInt takes float values and quantizes them to integers
func QuantizeToInt(matrix [][]float64) ([][]int, error) {
	//does the quantize
	return [][]int{}, nil
}
