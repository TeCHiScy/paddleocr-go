package ocr

import (
	"image"
	"image/color"
	"log"
	"math"
	"slices"
	"sort"
	"time"

	clipper "github.com/ctessum/go.clipper"
	"gocv.io/x/gocv"
)

// detector is the text detector.
type detector struct {
	*Predictor
	limitType    string
	limitSideLen int

	thresh        float32
	boxThresh     float64
	maxCandidates int
	unClipRatio   float64
	minSize       float32
	useDilation   bool
	scoreMode     string

	mean    []float32
	scale   []float32
	isScale bool
}

// newDetector creates a new text detector.
func newDetector(cfg *Config) (*detector, error) {
	dcfg := cfg.Detector
	model, err := NewPredictor(&cfg.Predictor, dcfg.ModelDir)
	if err != nil {
		return nil, err
	}
	return &detector{
		Predictor:    model,
		limitType:    dcfg.LimitType,
		limitSideLen: dcfg.LimitSideLen,

		thresh:        dcfg.Thresh,
		boxThresh:     dcfg.BoxThresh,
		unClipRatio:   dcfg.UnclipRatio,
		scoreMode:     dcfg.ScoreMode,
		useDilation:   dcfg.UseDilation,
		minSize:       3,
		maxCandidates: 1000,

		mean:    []float32{0.485, 0.456, 0.406},
		scale:   []float32{1 / 0.229, 1 / 0.224, 1 / 0.225},
		isScale: true,
	}, nil
}

func (d *detector) Run(img gocv.Mat) [][][]int {
	t := time.Now()
	h, w := img.Rows(), img.Cols()
	resizeImg, ratioH, ratioW := d.Resize(img)
	defer resizeImg.Close()

	normalize(resizeImg, d.mean, d.scale, d.isScale)

	d.input.Reshape([]int32{1, 3, int32(resizeImg.Rows()), int32(resizeImg.Cols())})
	d.input.CopyFromCpu(permute(resizeImg))
	d.predictor.Run()

	boxes := d.postProcess(h, w, ratioH, ratioW)

	log.Printf("detector: box num: %d, elapsed: %dms\n", len(boxes), time.Since(t).Milliseconds())
	return boxes
}

func (d *detector) Resize(img gocv.Mat) (gocv.Mat, float64, float64) {
	w, h := img.Cols(), img.Rows()
	ratio := 1.0
	if d.limitType == "min" {
		minWh := min(h, w)
		if minWh < d.limitSideLen {
			if h < w {
				ratio = float64(d.limitSideLen) / float64(h)
			} else {
				ratio = float64(d.limitSideLen) / float64(w)
			}
		}
	} else {
		maxWh := max(h, w)
		if maxWh > d.limitSideLen {
			if h > w {
				ratio = float64(d.limitSideLen) / float64(h)
			} else {
				ratio = float64(d.limitSideLen) / float64(w)
			}
		}
	}

	resizeH := int(float64(h) * ratio)
	resizeW := int(float64(w) * ratio)

	resizeH = max(int(math.Round(float64(resizeH)/32)*32), 32)
	resizeW = max(int(math.Round(float64(resizeW)/32)*32), 32)

	resizeImg := gocv.NewMat()
	gocv.Resize(img, &resizeImg, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)
	ratioH := float64(resizeH) / float64(h)
	ratioW := float64(resizeW) / float64(w)
	return resizeImg, ratioH, ratioW
}

func getMinBoxes(box gocv.RotatedRect2f) ([][]float32, float32) {
	ssid := max(box.Width, box.Height)
	m := gocv.NewMat()
	defer m.Close()

	gocv.BoxPoints2f(box, &m)
	points := mat2Vec(m)
	sort.Sort(xFloatSortBy(points))

	point1, point2, point3, point4 := points[0], points[1], points[2], points[3]
	if points[3][1] <= points[2][1] {
		point2, point3 = points[3], points[2]
	} else {
		point2, point3 = points[2], points[3]
	}
	if points[1][1] <= points[0][1] {
		point1, point4 = points[1], points[0]
	} else {
		point1, point4 = points[0], points[1]
	}

	return [][]float32{point1, point2, point3, point4}, ssid
}

func mat2Vec(mat gocv.Mat) [][]float32 {
	data := make([][]float32, mat.Rows())
	for i := 0; i < mat.Rows(); i++ {
		data[i] = make([]float32, mat.Cols())
		for j := 0; j < mat.Cols(); j++ {
			data[i][j] = mat.GetFloatAt(i, j)
		}
	}
	return data
}

func boxScoreFast(points [][]float32, pred gocv.Mat) float64 {
	height, width := pred.Rows(), pred.Cols()
	boxX := []float32{points[0][0], points[1][0], points[2][0], points[3][0]}
	boxY := []float32{points[0][1], points[1][1], points[2][1], points[3][1]}

	xmin := clamp(int(math.Floor(float64(slices.Min(boxX)))), 0, width-1)
	xmax := clamp(int(math.Ceil(float64(slices.Max(boxX)))), 0, width-1)
	ymin := clamp(int(math.Floor(float64(slices.Min(boxY)))), 0, height-1)
	ymax := clamp(int(math.Ceil(float64(slices.Max(boxY)))), 0, height-1)

	mask := gocv.NewMatWithSize(ymax-ymin+1, xmax-xmin+1, gocv.MatTypeCV8UC1)
	defer mask.Close()

	ppt := [][]image.Point{{
		image.Pt(int(points[0][0])-xmin, int(points[0][1])-ymin),
		image.Pt(int(points[1][0])-xmin, int(points[1][1])-ymin),
		image.Pt(int(points[2][0])-xmin, int(points[2][1])-ymin),
		image.Pt(int(points[3][0])-xmin, int(points[3][1])-ymin),
	}}
	gocv.FillPoly(&mask, gocv.NewPointsVectorFromPoints(ppt), color.RGBA{0, 0, 1, 0})
	croppedImg := pred.Region(image.Rect(xmin, ymin, xmax+1, ymax+1))
	defer croppedImg.Close()
	s := croppedImg.MeanWithMask(mask)
	return s.Val1
}

func (d *detector) getContourArea(box [][]float32) float64 {
	var area, dist float64
	for i := 0; i < 4; i++ {
		area += float64(box[i][0]*box[(i+1)%4][1] - box[i][1]*box[(i+1)%4][0])
		dist += math.Sqrt(float64(
			(box[i][0]-box[(i+1)%4][0])*(box[i][0]-box[(i+1)%4][0]) +
				(box[i][1]-box[(i+1)%4][1])*(box[i][1]-box[(i+1)%4][1]),
		))
	}
	area = math.Abs(area / 2.0)
	return area * d.unClipRatio / dist
}

func (d *detector) UnClip(box [][]float32) gocv.RotatedRect2f {
	distance := d.getContourArea(box)
	offset := clipper.NewClipperOffset()
	path := []*clipper.IntPoint{
		&clipper.IntPoint{X: clipper.CInt(box[0][0]), Y: clipper.CInt(box[0][1])},
		&clipper.IntPoint{X: clipper.CInt(box[1][0]), Y: clipper.CInt(box[1][1])},
		&clipper.IntPoint{X: clipper.CInt(box[2][0]), Y: clipper.CInt(box[2][1])},
		&clipper.IntPoint{X: clipper.CInt(box[3][0]), Y: clipper.CInt(box[3][1])},
	}
	offset.AddPath(clipper.Path(path), clipper.JtRound, clipper.EtClosedPolygon)
	soln := offset.Execute(distance)

	points := make([]image.Point, 0, 4)
	for j := 0; j < len(soln); j++ {
		for i := 0; i < len(soln[len(soln)-1]); i++ {
			points = append(points, image.Point{int(soln[j][i].X), int(soln[j][i].Y)})
		}
	}

	var res gocv.RotatedRect2f
	if len(points) <= 0 {
		res = gocv.RotatedRect2f{Center: gocv.NewPoint2f(0, 0), Width: 1, Height: 1, Angle: 0}
	} else {
		res = gocv.MinAreaRect2f(gocv.NewPointVectorFromPoints(points))
	}
	return res
}

func (d *detector) boxesFromBitmap(pred gocv.Mat, bitmap gocv.Mat) [][][]int {
	w, h := bitmap.Cols(), bitmap.Rows()
	contours := gocv.FindContours(bitmap, gocv.RetrievalList, gocv.ChainApproxSimple)
	numContours := contours.Size()
	if numContours > d.maxCandidates {
		numContours = d.maxCandidates
	}

	boxes := make([][][]int, 0, numContours)
	for i := 0; i < numContours; i++ {
		contour := contours.At(i)
		if contour.Size() <= 2 {
			continue
		}
		box := gocv.MinAreaRect2f(contour)
		minBoxes, ssid := getMinBoxes(box)
		if ssid < d.minSize {
			continue
		}

		var score float64
		if d.scoreMode == "slow" { // compute using polygon
			score = polygonScoreAcc(contour.ToPoints(), pred)
		} else {
			score = boxScoreFast(minBoxes, pred)
		}
		if score < d.boxThresh {
			continue
		}

		points := d.UnClip(minBoxes)
		if points.Height < 1.001 || points.Width < 1.001 {
			continue
		}

		clipBoxes, ssid := getMinBoxes(points)
		if ssid < d.minSize+2 {
			continue
		}

		dstWidth, dstHeight := pred.Cols(), pred.Rows()
		intClipBoxes := make([][]int, 4)
		for i := 0; i < 4; i++ {
			intClipBoxes[i] = []int{
				clamp(int(math.Round(float64(clipBoxes[i][0]/float32(w)*float32(dstWidth)))), 0, dstWidth),
				clamp(int(math.Round(float64(clipBoxes[i][1]/float32(h)*float32(dstHeight)))), 0, dstHeight),
			}
		}
		boxes = append(boxes, intClipBoxes)
	}
	return boxes
}

func orderPointsClockwise(box [][]int) [][]int {
	sort.Sort(xIntSortBy(box))
	leftMost, rightMost := [][]int{box[0], box[1]}, [][]int{box[2], box[3]}
	if leftMost[0][1] > leftMost[1][1] {
		leftMost[0], leftMost[1] = leftMost[1], leftMost[0]
	}
	if rightMost[0][1] > rightMost[1][1] {
		rightMost[0], rightMost[1] = rightMost[1], rightMost[0]
	}
	return [][]int{leftMost[0], rightMost[0], rightMost[1], leftMost[1]}
}

func filterTagDetRes(boxes [][][]int, oriH, oriW int, ratioH, ratioW float64) [][][]int {
	points := make([][][]int, 0, len(boxes))
	for i := 0; i < len(boxes); i++ {
		boxes[i] = orderPointsClockwise(boxes[i])
		for j := 0; j < len(boxes[i]); j++ {
			boxes[i][j][0] = int(float64(boxes[i][j][0]) / ratioW)
			boxes[i][j][1] = int(float64(boxes[i][j][1]) / ratioH)
			boxes[i][j][0] = min(max(boxes[i][j][0], 0), oriW-1)
			boxes[i][j][1] = min(max(boxes[i][j][1], 0), oriH-1)
		}
	}

	for i := 0; i < len(boxes); i++ {
		rectW := int(math.Sqrt(math.Pow(float64(boxes[i][0][0]-boxes[i][1][0]), 2.0) + math.Pow(float64(boxes[i][0][1]-boxes[i][1][1]), 2.0)))
		rectH := int(math.Sqrt(math.Pow(float64(boxes[i][0][0]-boxes[i][3][0]), 2.0) + math.Pow(float64(boxes[i][0][1]-boxes[i][3][1]), 2.0)))
		if rectW <= 4 || rectH <= 4 {
			continue
		}
		points = append(points, boxes[i])
	}
	return points
}

func (d *detector) postProcess(oriH, oriW int, ratioH, ratioW float64) [][][]int {
	shape := d.output.Shape()
	h, w := int(shape[2]), int(shape[3])

	predicts := make([]float32, accumulate(shape))
	d.output.CopyToCpu(predicts)

	pred := gocv.NewMatWithSize(h, w, gocv.MatTypeCV32F)
	defer pred.Close()

	cbuf := gocv.NewMatWithSize(h, w, gocv.MatTypeCV8UC1)
	defer cbuf.Close()

	threshold := d.thresh * 255
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			pred.SetFloatAt(i, j, predicts[i*w+j])
			cbuf.SetUCharAt(i, j, uint8(predicts[i*w+j]*255))
		}
	}

	bitmap := gocv.NewMat()
	defer bitmap.Close()
	gocv.Threshold(cbuf, &bitmap, threshold, 255, gocv.ThresholdBinary)

	if d.useDilation {
		kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Point{2, 2})
		gocv.Dilate(bitmap, &bitmap, kernel)
	}

	boxes := d.boxesFromBitmap(pred, bitmap)
	return filterTagDetRes(boxes, oriH, oriW, ratioH, ratioW)
}

func polygonScoreAcc(points []image.Point, pred gocv.Mat) float64 {
	width, height := pred.Cols(), pred.Rows()
	boxX := make([]int, 0, len(points))
	boxY := make([]int, 0, len(points))
	for _, p := range points {
		boxX = append(boxX, p.X)
		boxY = append(boxY, p.Y)
	}

	xMin := clamp(int(math.Floor(float64(slices.Min(boxX)))), 0, width-1)
	xMax := clamp(int(math.Ceil(float64(slices.Max(boxX)))), 0, width-1)
	yMin := clamp(int(math.Floor(float64(slices.Min(boxY)))), 0, height-1)
	yMax := clamp(int(math.Ceil(float64(slices.Max(boxY)))), 0, height-1)

	mask := gocv.Zeros(yMax-yMin+1, xMax-xMin+1, gocv.MatTypeCV8UC1)
	defer mask.Close()

	rookPoint := make([]image.Point, len(points))
	for i := range points {
		rookPoint[i] = image.Pt(boxX[i]-xMin, boxY[i]-yMin)
	}
	ppt := [][]image.Point{rookPoint}
	gocv.FillPoly(&mask, gocv.NewPointsVectorFromPoints(ppt), color.RGBA{0, 0, 1, 0})

	croppedImg := pred.Region(image.Rect(xMin, yMin, xMax+1, yMax+1))
	defer croppedImg.Close()

	s := croppedImg.MeanWithMask(mask)
	return s.Val1
}
