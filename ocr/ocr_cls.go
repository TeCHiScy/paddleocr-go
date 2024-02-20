package ocr

import (
	"log"
	"os"
	"time"

	"github.com/LKKlein/gocv"
)

type TextClassifier struct {
	*PaddleModel
	batchNum int
	thresh   float64
	shape    []int
	labels   []string
}

type ClsResult struct {
	Score float32
	Label int64
}

func NewTextClassifier(modelDir string, args map[string]any) *TextClassifier {
	shapes := []int{3, 48, 192}
	if v, ok := args["cls_image_shape"]; ok {
		for i, s := range v.([]any) {
			shapes[i] = s.(int)
		}
	}
	cls := &TextClassifier{
		PaddleModel: NewPaddleModel(args),
		batchNum:    getInt(args, "cls_batch_num", 30),
		thresh:      getFloat64(args, "cls_thresh", 0.9),
		shape:       shapes,
	}
	if checkModelExists(modelDir) {
		home, _ := os.UserHomeDir()
		modelDir, _ = downloadModel(home+"/.paddleocr/cls", modelDir)
	} else {
		log.Panicf("cls model path: %v not exist! Please check!", modelDir)
	}
	cls.LoadModel(modelDir)
	return cls
}

func (cls *TextClassifier) Run(imgs []gocv.Mat) []gocv.Mat {
	batch := cls.batchNum
	var clsTime int64 = 0
	clsout := make([]ClsResult, len(imgs))
	srcimgs := make([]gocv.Mat, len(imgs))
	c, h, w := cls.shape[0], cls.shape[1], cls.shape[2]
	for i := 0; i < len(imgs); i += batch {
		j := i + batch
		if len(imgs) < j {
			j = len(imgs)
		}

		normImgs := make([]float32, (j-i)*c*h*w)
		for k := i; k < j; k++ {
			tmp := gocv.NewMat()
			imgs[k].CopyTo(&tmp)
			srcimgs[k] = tmp
			img := clsResize(imgs[k], cls.shape)
			data := normPermute(img, []float32{0.5, 0.5, 0.5}, []float32{0.5, 0.5, 0.5}, 255.0)
			copy(normImgs[(k-i)*c*h*w:], data)
		}

		st := time.Now()
		cls.input.Reshape([]int32{int32(j - i), int32(c), int32(h), int32(w)})
		cls.input.CopyFromCpu(normImgs)

		cls.predictor.Run()

		predictShape := cls.outputs[0].Shape()
		predictBatch := make([]float32, numElements(predictShape))
		cls.outputs[0].CopyToCpu(predictBatch)

		for batchIdx := 0; batchIdx < int(predictShape[0]); batchIdx++ {
			l := batchIdx * int(predictShape[1])
			r := (batchIdx + 1) * int(predictShape[1])
			label, score := argmax(predictBatch[l:r])
			clsout[i+batchIdx] = ClsResult{Score: score, Label: int64(label)}
			if label%2 == 1 && float64(score) > cls.thresh {
				gocv.Rotate(srcimgs[i+batchIdx], &srcimgs[i+batchIdx], gocv.Rotate180Clockwise)
			}
		}

		clsTime += int64(time.Since(st).Milliseconds())
	}
	log.Println("cls num: ", len(clsout), ", cls time elapse: ", clsTime, "ms")
	return srcimgs
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
