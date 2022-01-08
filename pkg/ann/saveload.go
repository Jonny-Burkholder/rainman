package ann

import (
	"bytes"
	"encoding/json"
	"io"
	"reflect"
	"sync"
)

var lock sync.RWMutex

type aNNModel struct {
	Weights    [][][]float64 `json:"Weights"`
	Biases     [][]float64   `json:"Biases"`
	Deltas     [][]float64   `json:"Deltas"`
	Zvalues    [][]float64   `json:"Zvalues"`
	Layers     int           `json:"Layers"`
	Neurons    []int         `json:"Neurons"`
	Activation string        `json:"Activation"`
}

func (nn *aNN) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(aNNModel{
		Weights:    nn.weights,
		Biases:     nn.biases,
		Deltas:     nn.deltas,
		Zvalues:    nn.zvalues,
		Layers:     nn.layers,
		Neurons:    nn.neurons,
		Activation: reflect.TypeOf(nn.ActivationFn).Name(),
	})
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func (nn *aNN) UnmarshalJSON(data []byte) error {
	var nnMod aNNModel
	if err := json.Unmarshal(data, &nnMod); err != nil {
		return err
	}
	nn.weights = nnMod.Weights
	nn.biases = nnMod.Biases
	nn.deltas = nnMod.Deltas
	nn.zvalues = nnMod.Zvalues
	nn.layers = nnMod.Layers
	nn.neurons = nnMod.Neurons
	nn.ActivationFn = InitActivationFn(nnMod.Activation)
	return nil
}

func (nn *aNN) SaveModel(w io.Writer) error {
	lock.Lock()
	defer lock.Unlock()
	return json.NewEncoder(w).Encode(nn)
}

func LoadModel(r io.Reader) (*aNN, error) {
	lock.RLock()
	defer lock.RUnlock()
	var n aNN
	if err := json.NewDecoder(r).Decode(&n); err != nil {
		return nil, err
	}
	return &n, nil
}
