package ocr

import (
	"log"
	"os"
	"time"

	"github.com/LKKlein/gocv"
)

type TextRecognizer struct {
	*PaddleModel
	batchNum int
	textLen  int
	shape    []int
	charType string
	labels   []string
}

func NewTextRecognizer(modelDir string, args map[string]any) *TextRecognizer {
	shapes := []int{3, 32, 320}
	if v, ok := args["rec_image_shape"]; ok {
		for i, s := range v.([]any) {
			shapes[i] = s.(int)
		}
	}
	labelpath := getString(args, "rec_char_dict_path", "./config/ppocr_keys_v1.txt")
	labels := readLines2StringSlice(labelpath)
	if getBool(args, "use_space_char", true) {
		labels = append(labels, " ")
	}
	rec := &TextRecognizer{
		PaddleModel: NewPaddleModel(args),
		batchNum:    getInt(args, "rec_batch_num", 30),
		textLen:     getInt(args, "max_text_length", 25),
		charType:    getString(args, "rec_char_type", "ch"),
		shape:       shapes,
		labels:      labels,
	}
	if checkModelExists(modelDir) {
		home, _ := os.UserHomeDir()
		modelDir, _ = downloadModel(home+"/.paddleocr/rec/ch", modelDir)
	} else {
		log.Panicf("rec model path: %v not exist! Please check!", modelDir)
	}
	rec.LoadModel(modelDir)
	return rec
}

func (rec *TextRecognizer) Run(imgs []gocv.Mat, bboxes [][][]int) []OCRText {
	recResult := make([]OCRText, 0, len(imgs))
	batch := rec.batchNum
	var recTime int64 = 0
	c, h, w := rec.shape[0], rec.shape[1], rec.shape[2]
	for i := 0; i < len(imgs); i += batch {
		j := i + batch
		if len(imgs) < j {
			j = len(imgs)
		}

		maxwhratio := 0.0
		for k := i; k < j; k++ {
			h, w := imgs[k].Rows(), imgs[k].Cols()
			ratio := float64(w) / float64(h)
			if ratio > maxwhratio {
				maxwhratio = ratio
			}
		}

		normimgs := make([]float32, (j-i)*c*h*w)
		for k := i; k < j; k++ {
			data := crnnPreprocess(imgs[k], rec.shape, []float32{0.5, 0.5, 0.5},
				[]float32{0.5, 0.5, 0.5}, 255.0, maxwhratio, rec.charType)
			copy(normimgs[(k-i)*c*h*w:], data)
		}

		st := time.Now()
		rec.input.Reshape([]int32{int32(j - i), int32(c), int32(h), int32(w)})
		rec.input.CopyFromCpu(normimgs)

		// rec.predictor.SetZeroCopyInput(rec.input)
		rec.predictor.Run()
		// rec.predictor.GetZeroCopyOutput(rec.outputs[0])
		// rec.predictor.GetZeroCopyOutput(rec.outputs[1])

		predictShape := rec.outputs[0].Shape()
		predictBatch := make([]float32, numElements(predictShape))
		rec.outputs[0].CopyToCpu(predictBatch)
		recTime += int64(time.Since(st).Milliseconds())

		for m := 0; m < int(predictShape[0]); m++ {
			var text string
			var count int
			var score float32
			var lastIndex int
			for n := 0; n < int(predictShape[1]); n++ {
				l := (m*int(predictShape[1]) + n) * int(predictShape[2])
				r := (m*int(predictShape[1]) + n + 1) * int(predictShape[2])
				// get argmax_idx & get score
				argmaxIdx, maxValue := argmax(predictBatch[l:r])
				if argmaxIdx > 0 && (!(n > 0 && argmaxIdx == lastIndex)) {
					score += maxValue
					count += 1
					text += rec.labels[argmaxIdx]
				}
				lastIndex = argmaxIdx
			}
			if score == 0.0 && count == 0 {
				continue
			}
			score /= float32(count)
			recResult = append(recResult, OCRText{
				BBox:  bboxes[i+m],
				Text:  text,
				Score: float64(score),
			})
		}
	}
	log.Println("rec num: ", len(recResult), ", rec time elapse: ", recTime, "ms")
	return recResult
}
