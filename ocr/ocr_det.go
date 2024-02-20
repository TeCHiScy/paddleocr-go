package ocr

import (
	"log"
	"os"
	"time"

	"gocv.io/x/gocv"
)

type DBDetector struct {
	*PaddleModel
	preProcess  DetPreProcess
	postProcess DetPostProcess
}

func NewDBDetector(args *Config) *DBDetector {
	modelDir := args.DetModelDir
	detector := &DBDetector{
		PaddleModel: NewPaddleModel(args),
		preProcess:  NewDBProcess(make([]int, 0), args.DetMaxSideLen),
		postProcess: NewDBPostProcess(args.DetDBThresh, args.DetDBBoxThresh, args.DetDBUnclipRatio),
	}
	if checkModelExists(modelDir) {
		home, _ := os.UserHomeDir()
		modelDir, _ = downloadModel(home+"/.paddleocr/det", modelDir)
	} else {
		log.Panicf("det model path: %v not exist! Please check!", modelDir)
	}
	detector.LoadModel(modelDir)
	return detector
}

func (det *DBDetector) Run(img gocv.Mat) [][][]int {
	oriH := img.Rows()
	oriW := img.Cols()
	data, resizeH, resizeW := det.preProcess.Run(img)
	st := time.Now()
	det.input.Reshape([]int32{1, 3, int32(resizeH), int32(resizeW)})
	det.input.CopyFromCpu(data)

	det.predictor.Run()

	ratioH, ratioW := float64(resizeH)/float64(oriH), float64(resizeW)/float64(oriW)
	boxes := det.postProcess.Run(det.outputs[0], oriH, oriW, ratioH, ratioW)
	log.Println("det_box num: ", len(boxes), ", time elapse: ", time.Since(st))
	return boxes
}
