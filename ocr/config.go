package ocr

import (
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	UseGPU        bool   `yaml:"use_gpu"`
	IROptim       bool   `yaml:"ir_optim"`
	EnableMkldnn  bool   `yaml:"enable_mkldnn"`
	UseTensorrt   bool   `yaml:"use_tensorrt"`
	NumCPUThreads int    `yaml:"num_cpu_threads"`
	GPUID         int32  `yaml:"gpu_id"`
	GPUMem        uint64 `yaml:"gpu_mem"`

	DetModelDir      string  `yaml:"det_model_dir"`
	DetMaxSideLen    int     `yaml:"det_max_side_len"`
	DetDBThresh      float64 `yaml:"det_db_thresh"`
	DetDBBoxThresh   float64 `yaml:"det_db_box_thresh"`
	DetDBUnclipRatio float64 `yaml:"det_db_unclip_ratio"`

	RecModelDir     string `yaml:"rec_model_dir"`
	RecImageShape   []int  `yaml:"rec_image_shape"`
	RecBatchNum     int    `yaml:"rec_batch_num"`
	RecCharDictPath string `yaml:"rec_char_dict_path"`
	MaxTextLength   int    `yaml:"max_text_length"`
	UseSpaceChar    bool   `yaml:"use_space_char"`

	UseAngleCls   bool    `yaml:"use_angle_cls"`
	ClsModelDir   string  `yaml:"cls_model_dir"`
	ClsImageShape []int   `yaml:"cls_image_shape"`
	ClsBatchNum   int     `yaml:"cls_batch_num"`
	ClsThresh     float64 `yaml:"cls_thresh"`

	Det bool `yaml:"det"`
	Rec bool `yaml:"rec"`
	Cls bool `yaml:"cls"`
}

func ReadYaml(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	body := &Config{}
	if err := yaml.Unmarshal(data, &body); err != nil {
		return nil, err
	}
	return body, nil
}
