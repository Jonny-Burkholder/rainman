package tools

import (
	"encoding/json"
	"os"

	"github.com/Jonny-Burkholder/neural-network/neuralnetwork"
)

//LoadConfig reads a config.json file into memory
func LoadConfig(path string) (*neuralnetwork.Config, error) {
	c := &neuralnetwork.Config{}
	f, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}

	err = json.Unmarshal(f, c)

	return c, err
}

//SaveConfig persists a config file to disk
func SaveConfig(c *neuralnetwork.Config, path string) error {
	f, err := json.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(path, f, 0644)
}
