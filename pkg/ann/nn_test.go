package ann

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"testing"
	"time"
)

func print3DMatrix(m [][][]float64) {
	for i := 0; i < len(m); i++ {
		fmt.Printf("matrix #%d\n", i)
		print2DMatrix(m[i])
	}
}

func print2DMatrix(m [][]float64) {
	fmt.Printf("rows=%d, cols=%d\n", len(m), len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[i]); j++ {
			fmt.Printf("%.4f ", m[i][j])
		}
		fmt.Printf("\n")
	}
	fmt.Printf("\n")
}

func TestPrint(t *testing.T) {
	nn := NewANN(&Config{
		Layers:       []int{6, 12, 8, 4},
		Seed:         time.Now().UnixNano(),
		IsRegression: false,
		Activation:   Sigmoid,
	})
	print3DMatrix(nn.weights)
}

var samples = [][][]float64{
	{
		{
			0,
			0,
		},
		{
			0,
		},
	},
	{
		{
			0,
			1,
		},
		{
			1,
		},
	},
	{
		{
			1,
			0,
		},
		{
			1,
		},
	},
	{
		{
			1,
			1,
		},
		{
			0,
		},
	},
}

var samples2 = [][][]float64{
	{
		{
			0,
			0,
		},
		{
			10,
		},
	},
	{
		{
			0,
			1,
		},
		{
			20,
		},
	},
	{
		{
			1,
			0,
		},
		{
			20,
		},
	},
	{
		{
			1,
			1,
		},
		{
			10,
		},
	},
}

func TestNetwork(t *testing.T) {
	log.SetOutput(ioutil.Discard)

	tests := []struct {
		name               string
		samples            [][][]float64
		results            [][][]float64
		activationFunction ActivationFn
	}{
		{
			name:    "Output range 0-1 Sigmond",
			samples: samples,
			results: [][][]float64{
				{{0, 0}, {0}},
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
			},
			activationFunction: Sigmoid,
		},
		// {
		// 	name:    "Output range 0-1 Bent Identity",
		// 	samples: samples,
		// 	results: [][][]float64{
		// 		{{0, 0}, {0}},
		// 		{{0, 1}, {1}},
		// 		{{1, 0}, {1}},
		// 		{{1, 1}, {0}},
		// 	},
		// 	activationFunction: BentIdentity,
		// },
		{
			name:    "Output range 0-1 Rectified linear unit",
			samples: samples,
			results: [][][]float64{
				{{0, 0}, {0}},
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
			},
			activationFunction: ReLU,
		},
		// {
		// 	name:    "Output range 0-1 Leaky rectified linear unit",
		// 	samples: samples,
		// 	results: [][][]float64{
		// 		{{0, 0}, {0}},
		// 		{{0, 1}, {1}},
		// 		{{1, 0}, {1}},
		// 		{{1, 1}, {0}},
		// 	},
		// 	activationFunction: LeakyReLU,
		// },
		// {
		// 	name:    "Output range 0-1 ArSinH",
		// 	samples: samples,
		// 	results: [][][]float64{
		// 		{{0, 0}, {0}},
		// 		{{0, 1}, {1}},
		// 		{{1, 0}, {1}},
		// 		{{1, 1}, {0}},
		// 	},
		// 	activationFunction: ArSinH,
		// },
		// {
		// 	name:    "Output range 10-20",
		// 	samples: samples2,
		// 	results: [][][]float64{
		// 		{{0, 0}, {10}},
		// 		{{0, 1}, {20}},
		// 		{{1, 0}, {20}},
		// 		{{1, 1}, {10}},
		// 	},
		// 	activationFunction: ???,
		// },
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t1 *testing.T) {

			conf := &Config{
				Layers:       []int{2, 3, 1},
				Seed:         time.Now().UnixNano(),
				IsRegression: false,
				Activation:   tt.activationFunction,
			}

			nn := NewANN(conf)

			nn.Train(tt.samples, 10000, 1.01, true)
			for _, exp := range tt.results {
				if res := nn.Predict(exp[0])[0]; !percDiffLessThan(res, exp[1][0], 2) {
					t1.Errorf("Result is too different to be accurate; Using %s got: %.2f, expected: %.2f", tt.activationFunction, res, exp[1][0])
				}
			}

			// Export and re-load network to ensure those functions are tested.
			var buf bytes.Buffer
			if err := nn.SaveModel(&buf); err != nil {
				t1.Fatalf("Unable to export network: %+v", err)
			}

			// Using LoadFrom tests both the Load and LoadFrom functions.
			nn, err := LoadModel(bytes.NewReader(buf.Bytes()))
			if err != nil {
				t1.Fatalf("Could not load network: %+v", err)
			}

			for _, exp := range tt.results {
				if res := nn.Predict(exp[0])[0]; !percDiffLessThan(res, exp[1][0], 2) {
					t1.Errorf("Result is too different to be accurate; Using %s got: %.2f, expected: %.2f", tt.activationFunction, res, exp[1][0])
				}
			}
		})
	}
}

// percDiffLessThan returns whether v1 and v2 differ by the percentage.
func percDiffLessThan(v1, v2, perc float64) bool {
	absDiff := math.Abs(v1 - v2)
	// Prevent issues with divide by zero
	if absDiff == 0 || v1 == 0 || v2 == 0 {
		return true
	}

	decDiff := absDiff / math.Max(v1, v2)
	return decDiff*100.0 < perc
}
