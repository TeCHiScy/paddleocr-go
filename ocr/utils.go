package ocr

import (
	"archive/tar"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"
)

func ReadImage(image_path string) gocv.Mat {
	img := gocv.IMRead(image_path, gocv.IMReadColor)
	if img.Empty() {
		log.Printf("Could not read image %s\n", image_path)
		os.Exit(1)
	}
	return img
}

func clip(value, min, max int) int {
	if value <= min {
		return min
	} else if value >= max {
		return max
	}
	return value
}

func argmax(s []float32) (int, float32) {
	max, idx := s[0], 0
	for i, v := range s {
		if v > max {
			idx, max = i, v
		}
	}
	return idx, max
}

func checkModelExists(modelPath string) bool {
	if isPathExist(modelPath+"/model") && isPathExist(modelPath+"/params") {
		return true
	}
	if strings.HasPrefix(modelPath, "http://") ||
		strings.HasPrefix(modelPath, "ftp://") || strings.HasPrefix(modelPath, "https://") {
		return true
	}
	return false
}

func downloadFile(filepath, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	log.Println("[download_file] from:", url, " to:", filepath)
	return err
}

func isPathExist(path string) bool {
	if _, err := os.Stat(path); err == nil {
		return true
	} else if os.IsNotExist(err) {
		return false
	}
	return false
}

func downloadModel(modelDir, modelPath string) (string, error) {
	if modelPath != "" && (strings.HasPrefix(modelPath, "http://") ||
		strings.HasPrefix(modelPath, "ftp://") || strings.HasPrefix(modelPath, "https://")) {
		if checkModelExists(modelDir) {
			return modelDir, nil
		}
		_, suffix := path.Split(modelPath)
		outPath := filepath.Join(modelDir, suffix)
		outDir := filepath.Dir(outPath)
		if !isPathExist(outDir) {
			os.MkdirAll(outDir, os.ModePerm)
		}

		if !isPathExist(outPath) {
			err := downloadFile(outPath, modelPath)
			if err != nil {
				return "", err
			}
		}

		if strings.HasSuffix(outPath, ".tar") && !checkModelExists(modelDir) {
			unTar(modelDir, outPath)
			os.Remove(outPath)
			return modelDir, nil
		}
		return modelDir, nil
	}
	return modelPath, nil
}

func unTar(dst, src string) (err error) {
	fr, err := os.Open(src)
	if err != nil {
		return err
	}
	defer fr.Close()

	tr := tar.NewReader(fr)
	for {
		hdr, err := tr.Next()

		switch {
		case err == io.EOF:
			return nil
		case err != nil:
			return err
		case hdr == nil:
			continue
		}

		var dstFileDir string
		if strings.Contains(hdr.Name, "model") {
			dstFileDir = filepath.Join(dst, "model")
		} else if strings.Contains(hdr.Name, "params") {
			dstFileDir = filepath.Join(dst, "params")
		}

		switch hdr.Typeflag {
		case tar.TypeDir:
			continue
		case tar.TypeReg:
			file, err := os.OpenFile(dstFileDir, os.O_CREATE|os.O_RDWR, os.FileMode(hdr.Mode))
			if err != nil {
				return err
			}
			_, err2 := io.Copy(file, tr)
			if err2 != nil {
				return err2
			}
			file.Close()
		}
	}

	return nil
}

func readLines2StringSlice(filepath string) []string {
	if strings.HasPrefix(filepath, "http://") || strings.HasPrefix(filepath, "https://") {
		home, _ := os.UserHomeDir()
		dir := home + "/.paddleocr/rec/"
		_, suffix := path.Split(filepath)
		f := dir + suffix
		if !isPathExist(f) {
			err := downloadFile(f, filepath)
			if err != nil {
				log.Println("download ppocr key file error!")
				return nil
			}
		}
		filepath = f
	}
	content, err := os.ReadFile(filepath)
	if err != nil {
		log.Println("read ppocr key file error!")
		return nil
	}
	lines := strings.Split(string(content), "\n")
	return lines
}
