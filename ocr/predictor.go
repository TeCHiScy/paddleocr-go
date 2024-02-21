package ocr

import (
	"fmt"
	"path"

	pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

// Predictor is a wrapped paddle predictor.
type Predictor struct {
	predictor *pd.Predictor
	input     *pd.Tensor
	output    *pd.Tensor
	config    *pd.Config
}

// NewPredictor creates a new paddle predictor.
func NewPredictor(cfg *PredictorConfig, modelDir string) (*Predictor, error) {
	if !isPathExist(path.Join(modelDir, "inference.model")) ||
		!isPathExist(path.Join(modelDir, "inference.params")) {
		return nil, fmt.Errorf("model not found in %s.", modelDir)
	}

	config := pd.NewConfig()
	config.DisableGlogInfo()
	config.SetModel(path.Join(modelDir, "inference.model"), path.Join(modelDir, "inference.params"))

	if cfg.UseGPU {
		config.EnableUseGpu(cfg.GPUMem, cfg.GPUID)
	} else {
		// config.DisableGpu()
		config.SetCpuMathLibraryNumThreads(cfg.NumCPUThreads)
		if cfg.UseMKLDNN {
			config.EnableMKLDNN()
		}
	}

	// false for zero copy tensor
	// config.SwitchUseFeedFetchOps(false)
	// config.SwitchSpecifyInputNames(true)
	config.SwitchIrOptim(cfg.UseIROptim)
	config.EnableMemoryOptim(true)
	predictor := pd.NewPredictor(config)

	return &Predictor{
		input:     predictor.GetInputHandle(predictor.GetInputNames()[0]),
		output:    predictor.GetOutputHandle(predictor.GetOutputNames()[0]),
		config:    config,
		predictor: predictor,
	}, nil
}
