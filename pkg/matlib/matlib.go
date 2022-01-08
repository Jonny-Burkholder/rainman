package matlib

import (
	"fmt"
	"math/rand"
	"time"
)

var (
	sampleMatA = [][]float64{ // shape 2,3
		{2, 2},
		{0, 3},
		{0, 4},
	}
	sampleMatB = [][]float64{ // shape 3,2
		{2, 1, 2},
		{3, 2, 4},
	}
	sampleDotAB = [][]float64{
		{10, 6, 12},
		{9, 6, 12},
		{12, 8, 16},
	}
)

func MatrixTranspose(m [][]float64) [][]float64 {
	r := make([][]float64, len(m[0]))
	for x := range r {
		r[x] = make([]float64, len(m))
	}
	for y, s := range m {
		for x, e := range s {
			r[x][y] = e
		}
	}
	return r
}

func MatrixDotProduct(a, b [][]float64) [][]float64 {
	if a == nil || b == nil {
		panic("got nil receiver!")
	}
	c := NewMatrix(len(a), len(b[0]))
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(b[0]); j++ {
			for k := 0; k < len(b); k++ {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return c
}

func NewMatrix(x, y int) [][]float64 {
	if x < 1 || y < 1 {
		panic("bad sizes!")
	}
	m := make([][]float64, x)
	for i := range m {
		m[i] = make([]float64, y)
	}
	return m
}

func MatrixApply(m [][]float64, fn func(x float64) float64) [][]float64 {
	for i := range m {
		for j := range m[i] {
			m[i][j] = fn(m[i][j])
		}
	}
	return m
}

func MatrixFillRand(m [][]float64, r *rand.Rand) [][]float64 {
	if r == nil {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	for i := range m {
		for j := range m[i] {
			m[i][j] = r.Float64()
		}
	}
	return m
}

func VectorDotProduct(a, b []float64) float64 {
	if len(a) != len(a) {
		panic("bad sizes!")
	}
	var r float64
	for i, ai := range a {
		r += ai * b[i]
	}
	return r
}

func MatrixToString(m [][]float64) string {
	ss := fmt.Sprintf("[%d][%d]float64{\n", len(m), len(m[0]))
	for i := 0; i < len(m); i++ {
		ss += fmt.Sprintf(" {")
		for j := 0; j < len(m[i]); j++ {
			ss += fmt.Sprintf("%.4f", m[i][j])
			if j < len(m[i])-1 {
				ss += fmt.Sprintf(", ")
			}
		}
		ss += "},\n"
	}
	ss += "}\n"
	return ss
}
