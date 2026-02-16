package main

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	tf "github.com/galeone/tensorflow/tensorflow/go"

	"golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

type ImageType string

const (
	IMAGE_OTHER    ImageType = "UNKNOWN"
	IMAGE_WEBP     ImageType = "WEBP"
	IMAGE_JPEG     ImageType = "JPG"
	IMAGE_PNG      ImageType = "PNG"
	IMAGE_GIF      ImageType = "GIF"
	HTTP_ADDRESS             = "0.0.0.0:9000"
	MODEL_TRESHOLD           = 0.7
	MODEL_SIZE               = 224
)

type ClassifyResults struct {
	Drawing float32
	Hentai  float32
	Neutral float32
	Porn    float32
	Sexy    float32
}

var tfModel *tf.SavedModel

func main() {
	time.Local = time.UTC

	// Startup Services
	var stopCtx, stop = context.WithCancel(context.Background())
	var stopWg sync.WaitGroup

	StartupModel(stopCtx, &stopWg)
	go StartupHTTP(stopCtx, &stopWg)

	// Await Shutdown Signal
	cancel := make(chan os.Signal, 1)
	signal.Notify(cancel, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)
	<-cancel
	stop()

	// Begin Shutdown Process
	timeout, finish := context.WithTimeout(context.Background(), time.Minute)
	defer finish()
	go func() {
		<-timeout.Done()
		if timeout.Err() == context.DeadlineExceeded {
			log.Fatalln("[MAIN] Shutdown Deadline Exceeded")
		}
	}()
	stopWg.Wait()
	os.Exit(0)
}

func StartupHTTP(stop context.Context, await *sync.WaitGroup) {

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		var elapsedProbe, elapsedDecode, elapsedClassify int64
		t := time.Now()

		// Request Validation
		if r.Method != http.MethodPost {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}

		var d []byte
		if raw, err := io.ReadAll(r.Body); err != nil {
			http.Error(w, "Body Error", http.StatusBadRequest)
			return
		} else {
			d = raw
		}

		// Image Validation
		var imageType ImageType
		var imageInfo image.Config
		var imageData image.Image
		var imageErr error

		switch { // Sniffing
		case len(d) > 3 && // JPEG
			d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF:
			imageType = IMAGE_JPEG

		case len(d) > 8 && // PNG
			d[0] == 0x89 && d[1] == 0x50 && d[2] == 0x4E && d[3] == 0x47 &&
			d[4] == 0x0D && d[5] == 0x0A && d[6] == 0x1A && d[7] == 0x0A:
			imageType = IMAGE_PNG

		case len(d) > 4 && // GIF
			d[0] == 0x47 && d[1] == 0x49 && d[2] == 0x46 && d[3] == 0x38:
			imageType = IMAGE_GIF

		case len(d) > 12 && // WEBP
			d[0] == 0x52 && d[1] == 0x49 && d[2] == 0x46 && d[3] == 0x46 &&
			d[8] == 0x57 && d[9] == 0x45 && d[10] == 0x42 && d[11] == 0x50:
			imageType = IMAGE_WEBP

		default:
			imageType = IMAGE_OTHER
		}

		switch imageType { // Decode Header
		case IMAGE_WEBP:
			imageInfo, imageErr = webp.DecodeConfig(bytes.NewReader(d))
		case IMAGE_JPEG:
			imageInfo, imageErr = jpeg.DecodeConfig(bytes.NewReader(d))
		case IMAGE_PNG:
			imageInfo, imageErr = png.DecodeConfig(bytes.NewReader(d))
		case IMAGE_GIF:
			imageInfo, imageErr = gif.DecodeConfig(bytes.NewReader(d))
		default:
			http.Error(w, "Unsupported Image Format", http.StatusBadRequest)
			return
		}
		elapsedProbe = time.Since(t).Nanoseconds()
		t = time.Now()

		if imageErr != nil {
			http.Error(w, "Invalid Image Data", http.StatusBadRequest)
			return
		}
		if imageInfo.Height > 2048 || imageInfo.Width > 2048 {
			http.Error(w, "Image dimension cannot be larger than 2048 pixels", http.StatusBadRequest)
			return
		}
		if imageInfo.Height < 32 || imageInfo.Width < 32 {
			http.Error(w, "Image dimension cannot be smaller than 32 pixels", http.StatusBadRequest)
			return
		}

		switch imageType { // Decode Pixels
		case IMAGE_WEBP:
			imageData, imageErr = webp.Decode(bytes.NewReader(d))
		case IMAGE_JPEG:
			imageData, imageErr = jpeg.Decode(bytes.NewReader(d))
		case IMAGE_PNG:
			imageData, imageErr = png.Decode(bytes.NewReader(d))
		case IMAGE_GIF:
			imageData, imageErr = gif.Decode(bytes.NewReader(d))
		default:
			http.Error(w, "Invalid Image Type", http.StatusBadRequest)
			return
		}
		if imageErr != nil {
			http.Error(w, "Invalid Image Data", http.StatusBadRequest)
			return
		}
		elapsedDecode = time.Since(t).Nanoseconds()
		t = time.Now()

		// Image Classification
		results, err := ModelClassifyImage(imageData)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		elapsedClassify = time.Since(t).Nanoseconds()
		t = time.Now()

		enc := json.NewEncoder(w)
		enc.SetEscapeHTML(false)
		enc.Encode(map[string]any{
			"allowed": (results.Hentai + results.Porn + (results.Sexy * 0.9)) < MODEL_TRESHOLD,
			"timings": map[string]any{
				"probe":    elapsedProbe,
				"decode":   elapsedDecode,
				"classify": elapsedClassify,
			},
			"image": map[string]any{
				"height": imageInfo.Height,
				"width":  imageInfo.Width,
			},
			"logits": map[string]any{
				"drawing": results.Drawing,
				"hentai":  results.Hentai,
				"neutral": results.Neutral,
				"porn":    results.Porn,
				"sexy":    results.Sexy,
			},
		})

	})

	svr := http.Server{
		Handler:           mux,
		Addr:              HTTP_ADDRESS,
		MaxHeaderBytes:    4096,
		IdleTimeout:       10 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      30 * time.Second,
		ReadTimeout:       30 * time.Second,
	}

	// Shutdown Logic
	await.Add(1)
	go func() {
		defer await.Done()
		<-stop.Done()

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := svr.Shutdown(shutdownCtx); err != nil {
			log.Println("[HTTP] Shutdown Failed:", err)
		}

		log.Println("[HTTP] Server Closed")
	}()

	// Server Startup
	log.Println("[HTTP] Listening @", HTTP_ADDRESS)
	if err := svr.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalln("[HTTP] Startup Failed:", err)
	}
}

func StartupModel(stop context.Context, await *sync.WaitGroup) {
	t := time.Now()

	// Read Model from Disk
	model, err := tf.LoadSavedModel("model", []string{"serve"}, nil)
	if err != nil {
		log.Fatalf("[TFGO] Unable to Load Model (%s)\n", err)
	}
	tfModel = model

	// Test Model using Dummy Tensor
	dummy, _ := tf.NewTensor([1][MODEL_SIZE][MODEL_SIZE][3]float32{})
	if _, err := ModelClassifyTensor(dummy); err != nil {
		log.Fatalf("[TFGO] Failed to Initialize Model (%s)\n", err)
	}

	// Shutdown Logic
	await.Add(1)
	go func() {
		defer await.Done()
		<-stop.Done()
		tfModel.Session.Close()
		log.Println("[TFGO] Model Closed")
	}()

	log.Printf("[TFGO] Model Ready (%s)\n", time.Since(t))
}

// Cast Predictions on a Tensor using the NSFW Model
func ModelClassifyTensor(tensor *tf.Tensor) ([]float32, error) {
	results, err := tfModel.Session.Run(
		map[tf.Output]*tf.Tensor{
			tfModel.Graph.Operation("serving_default_input").Output(0): tensor,
		},
		[]tf.Output{
			tfModel.Graph.Operation("StatefulPartitionedCall").Output(0),
		},
		[]*tf.Operation{},
	)
	if err != nil {
		return []float32{}, err
	}
	// cursed...
	return results[0].Value().([][]float32)[0], err
}

// Classify an Image returning true if it's considered safe
func ModelClassifyImage(someImage image.Image) (ClassifyResults, error) {

	// Resize Image to Usable Size
	resized := image.NewRGBA(image.Rect(0, 0, MODEL_SIZE, MODEL_SIZE))
	draw.NearestNeighbor.Scale(resized, resized.Rect, someImage, someImage.Bounds(), draw.Over, nil)

	// Convert Pixel Data into Normalized Floats
	var tensorCap = MODEL_SIZE * MODEL_SIZE * 3
	var tensorData = make([]float32, 0, tensorCap)
	var tensorShape = []int64{1, MODEL_SIZE, MODEL_SIZE, 3}
	for x := 0; x < MODEL_SIZE; x++ {
		for y := 0; y < MODEL_SIZE; y++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			tensorData = append(tensorData, float32(r)/65535, float32(g)/65535, float32(b)/65535)
		}
	}

	// Create Tensor, reshape it, then classify
	tensor, err := tf.NewTensor(tensorData)
	if err != nil {
		return ClassifyResults{}, err
	}
	if err := tensor.Reshape(tensorShape); err != nil {
		return ClassifyResults{}, err
	}
	results, err := ModelClassifyTensor(tensor)
	if err != nil {
		return ClassifyResults{}, err
	}

	// Return Results
	// Drawing[0], Hentai[1], Neutral[2], Porn[3], Sexy[4]
	return ClassifyResults{
		Drawing: results[0],
		Hentai:  results[1],
		Neutral: results[2],
		Porn:    results[3],
		Sexy:    results[4],
	}, nil
}
