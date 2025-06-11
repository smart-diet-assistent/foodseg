#ifndef FOOD_SEGMENTATION_H
#define FOOD_SEGMENTATION_H

#include "Arduino.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Food category definitions (33 classes including background)
#define NUM_FOOD_CLASSES 33
#define BACKGROUND_CLASS 0

// Model input/output dimensions
#define MODEL_INPUT_WIDTH 320
#define MODEL_INPUT_HEIGHT 240
#define MODEL_INPUT_CHANNELS 3
#define MODEL_INPUT_SIZE (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS)
#define MODEL_OUTPUT_SIZE (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * NUM_FOOD_CLASSES)

// Memory allocation for TensorFlow Lite
#define TENSOR_ARENA_SIZE (600 * 1024)  // 600KB for model execution

// Food category names
extern const char* FOOD_NAMES[NUM_FOOD_CLASSES];

// Model data from model_data.cc
extern const unsigned char g_model[];
extern const int g_model_len;

class FoodSegmentation {
public:
    FoodSegmentation();
    ~FoodSegmentation();
    
    // Initialize the model
    bool begin();
    
    // Run inference on input image
    bool predict(float* input_image, uint8_t* output_mask);
    
    // Helper functions
    bool preprocessImage(uint8_t* rgb_image, float* normalized_output);
    int getClassWithMaxArea(uint8_t* mask, float* percentage = nullptr);
    void calculateClassAreas(uint8_t* mask, float* areas);
    const char* getClassName(int class_id);
    
    // Status functions
    bool isInitialized() { return initialized_; }
    const char* getLastError() { return error_message_; }
    
private:
    // TensorFlow Lite components
    tflite::MicroErrorReporter* error_reporter_;
    const tflite::Model* model_;
    tflite::MicroInterpreter* interpreter_;
    tflite::AllOpsResolver* resolver_;
    
    // Memory management
    uint8_t* tensor_arena_;
    
    // Model tensors
    TfLiteTensor* input_tensor_;
    TfLiteTensor* output_tensor_;
    
    // Status
    bool initialized_;
    char error_message_[256];
    
    // Private helper functions
    void setError(const char* message);
    bool allocateMemory();
    void freeMemory();
};

// Utility functions for image processing
namespace FoodSegmentationUtils {
    // Convert RGB888 to normalized float array
    void rgbToFloat(uint8_t* rgb_data, int width, int height, float* output);
    
    // Resize image using nearest neighbor interpolation
    void resizeImage(uint8_t* input, int input_w, int input_h, 
                    uint8_t* output, int output_w, int output_h, int channels);
    
    // Apply color map to segmentation mask for visualization
    void applyColorMap(uint8_t* mask, int width, int height, uint8_t* colored_output);
    
    // Calculate IoU between two masks
    float calculateIoU(uint8_t* mask1, uint8_t* mask2, int size, int class_id);
}

#endif // FOOD_SEGMENTATION_H
