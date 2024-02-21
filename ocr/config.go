package ocr

import (
	"os"

	"gopkg.in/yaml.v3"
)

// PredictorConfig is the configuration for paddle predictor.
type PredictorConfig struct {
	UseGPU        bool   `yaml:"use_gpu"`
	UseMKLDNN     bool   `yaml:"use_mkldnn"`
	UseTensorrt   bool   `yaml:"use_tensorrt"`
	UseIROptim    bool   `yaml:"use_ir_optim"`
	NumCPUThreads int    `yaml:"num_cpu_threads"`
	GPUID         int32  `yaml:"gpu_id"`
	GPUMem        uint64 `yaml:"gpu_mem"`
}

// Config is the configuration for OCR engine.
// Refer: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/deploy/cpp_infer/src/args.cpp
type Config struct {
	Predictor PredictorConfig `yaml:"predictor"`

	Detector struct {
		ModelDir     string  `yaml:"model_dir"`
		LimitType    string  `yaml:"limit_type"`
		LimitSideLen int     `yaml:"limit_side_len"`
		Thresh       float32 `yaml:"thresh"`
		BoxThresh    float64 `yaml:"box_thresh"`
		UnclipRatio  float64 `yaml:"unclip_ratio"`
		ScoreMode    string  `yaml:"score_mode"`
		UseDilation  bool    `yaml:"use_dilation"`
	} `yaml:"detector"`

	Recognizer struct {
		ModelDir      string `yaml:"model_dir"`
		BatchNum      int    `yaml:"batch_num"`
		ImageShape    []int  `yaml:"image_shape"`
		CharDictPath  string `yaml:"char_dict_path"`
		MaxTextLength int    `yaml:"max_text_length"`
	} `yaml:"recognizer"`

	Classifier struct {
		Enabled    bool    `yaml:"enabled"`
		ModelDir   string  `yaml:"model_dir"`
		Thresh     float32 `yaml:"thresh"`
		BatchNum   int     `yaml:"batch_num"`
		ImageShape []int   `yaml:"image_shape"`
	} `yaml:"classifier"`
}

// ReadConfig reads the OCR engine configuration from .yaml file.
func ReadConfig(name string) (*Config, error) {
	data, err := os.ReadFile(name)
	if err != nil {
		return nil, err
	}

	cfg := &Config{}
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return cfg, nil
}
