#include "FoodSegmentation.h"
#include "model_data.cc"
#include <cmath>

// Food category names corresponding to your inference_demo.py
const char* FOOD_NAMES[NUM_FOOD_CLASSES] = {
    "Background",
    "Chocolate",
    "Cheese/Butter", 
    "Milk",
    "Egg",
    "Apple",
    "Banana",
    "Strawberry",
    "Mango",
    "Orange",
    "Beef Steak",
    "Pork",
    "Chicken/Duck",
    "Sauce",
    "Bread",
    "Corn",
    "Pasta",
    "Noodles",
    "Rice",
    "Tofu",
    "Eggplant",
    "Potato",
    "Garlic",
    "Tomato",
    "Scallion",
    "Ginger",
    "Lettuce",
    "Cucumber",
    "Carrot",
    "Cabbage",
    "Green Bean",
    "King Oyster Mushroom",
    "Shiitake Mushroom"
};

FoodSegmentation::FoodSegmentation() 
    : error_reporter_(nullptr)
    , model_(nullptr)
    , interpreter_(nullptr)
    , resolver_(nullptr)
    , tensor_arena_(nullptr)
    , input_tensor_(nullptr)
    , output_tensor_(nullptr)
    , initialized_(false) {
    memset(error_message_, 0, sizeof(error_message_));
}

FoodSegmentation::~FoodSegmentation() {
    freeMemory();
}

bool FoodSegmentation::begin() {
    // Initialize error reporter
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter_ = &micro_error_reporter;
    
    // Load model
    model_ = tflite::GetModel(g_model);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        setError("Model schema version mismatch");
        return false;
    }
    
    // Allocate memory
    if (!allocateMemory()) {
        return false;
    }
    
    // Create resolver
    static tflite::AllOpsResolver resolver;
    resolver_ = &resolver;
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model_, *resolver_, tensor_arena_, TENSOR_ARENA_SIZE, error_reporter_);
    interpreter_ = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter_->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        setError("Failed to allocate tensors");
        return false;
    }
    
    // Get input and output tensors
    input_tensor_ = interpreter_->input(0);
    output_tensor_ = interpreter_->output(0);
    
    // Validate tensor dimensions
    if (input_tensor_->dims->size != 4 || 
        input_tensor_->dims->data[1] != MODEL_INPUT_CHANNELS ||
        input_tensor_->dims->data[2] != MODEL_INPUT_HEIGHT ||
        input_tensor_->dims->data[3] != MODEL_INPUT_WIDTH) {
        setError("Input tensor dimensions mismatch");
        return false;
    }
    
    if (output_tensor_->dims->size != 4 ||
        output_tensor_->dims->data[1] != NUM_FOOD_CLASSES ||
        output_tensor_->dims->data[2] != MODEL_INPUT_HEIGHT ||
        output_tensor_->dims->data[3] != MODEL_INPUT_WIDTH) {
        setError("Output tensor dimensions mismatch");
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool FoodSegmentation::predict(float* input_image, uint8_t* output_mask) {
    if (!initialized_) {
        setError("Model not initialized");
        return false;
    }
    
    // Copy input data to input tensor
    float* input_data = input_tensor_->data.f;
    memcpy(input_data, input_image, MODEL_INPUT_SIZE * sizeof(float));
    
    // Run inference
    TfLiteStatus invoke_status = interpreter_->Invoke();
    if (invoke_status != kTfLiteOk) {
        setError("Inference failed");
        return false;
    }
    
    // Process output - find class with maximum probability for each pixel
    float* output_data = output_tensor_->data.f;
    int pixel_count = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT;
    
    for (int i = 0; i < pixel_count; i++) {
        int max_class = 0;
        float max_prob = output_data[i * NUM_FOOD_CLASSES];
        
        for (int c = 1; c < NUM_FOOD_CLASSES; c++) {
            float prob = output_data[i * NUM_FOOD_CLASSES + c];
            if (prob > max_prob) {
                max_prob = prob;
                max_class = c;
            }
        }
        
        output_mask[i] = (uint8_t)max_class;
    }
    
    return true;
}

bool FoodSegmentation::preprocessImage(uint8_t* rgb_image, float* normalized_output) {
    // Normalization parameters (same as training)
    const float mean[3] = {0.485f, 0.456f, 0.406f};  // ImageNet means
    const float std[3] = {0.229f, 0.224f, 0.225f};   // ImageNet stds
    
    int pixel_count = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT;
    
    // Convert RGB to normalized float and rearrange to CHW format
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < pixel_count; i++) {
            float pixel_value = rgb_image[i * 3 + c] / 255.0f;
            normalized_output[c * pixel_count + i] = (pixel_value - mean[c]) / std[c];
        }
    }
    
    return true;
}

int FoodSegmentation::getClassWithMaxArea(uint8_t* mask, float* percentage) {
    int class_counts[NUM_FOOD_CLASSES] = {0};
    int total_pixels = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT;
    
    // Count pixels for each class
    for (int i = 0; i < total_pixels; i++) {
        if (mask[i] < NUM_FOOD_CLASSES) {
            class_counts[mask[i]]++;
        }
    }
    
    // Find class with maximum area (excluding background)
    int max_class = 1;
    int max_count = class_counts[1];
    
    for (int c = 2; c < NUM_FOOD_CLASSES; c++) {
        if (class_counts[c] > max_count) {
            max_count = class_counts[c];
            max_class = c;
        }
    }
    
    if (percentage != nullptr) {
        *percentage = (float)max_count / (float)total_pixels * 100.0f;
    }
    
    return max_class;
}

void FoodSegmentation::calculateClassAreas(uint8_t* mask, float* areas) {
    int class_counts[NUM_FOOD_CLASSES] = {0};
    int total_pixels = MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT;
    
    // Count pixels for each class
    for (int i = 0; i < total_pixels; i++) {
        if (mask[i] < NUM_FOOD_CLASSES) {
            class_counts[mask[i]]++;
        }
    }
    
    // Convert to percentages
    for (int c = 0; c < NUM_FOOD_CLASSES; c++) {
        areas[c] = (float)class_counts[c] / (float)total_pixels * 100.0f;
    }
}

const char* FoodSegmentation::getClassName(int class_id) {
    if (class_id >= 0 && class_id < NUM_FOOD_CLASSES) {
        return FOOD_NAMES[class_id];
    }
    return "Unknown";
}

void FoodSegmentation::setError(const char* message) {
    strncpy(error_message_, message, sizeof(error_message_) - 1);
    error_message_[sizeof(error_message_) - 1] = '\0';
}

bool FoodSegmentation::allocateMemory() {
    tensor_arena_ = (uint8_t*)malloc(TENSOR_ARENA_SIZE);
    if (tensor_arena_ == nullptr) {
        setError("Failed to allocate tensor arena");
        return false;
    }
    return true;
}

void FoodSegmentation::freeMemory() {
    if (tensor_arena_ != nullptr) {
        free(tensor_arena_);
        tensor_arena_ = nullptr;
    }
}

// Utility functions implementation
namespace FoodSegmentationUtils {
    
    void rgbToFloat(uint8_t* rgb_data, int width, int height, float* output) {
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};
        
        int pixel_count = width * height;
        
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < pixel_count; i++) {
                float pixel_value = rgb_data[i * 3 + c] / 255.0f;
                output[c * pixel_count + i] = (pixel_value - mean[c]) / std[c];
            }
        }
    }
    
    void resizeImage(uint8_t* input, int input_w, int input_h, 
                    uint8_t* output, int output_w, int output_h, int channels) {
        float x_ratio = (float)input_w / output_w;
        float y_ratio = (float)input_h / output_h;
        
        for (int y = 0; y < output_h; y++) {
            for (int x = 0; x < output_w; x++) {
                int src_x = (int)(x * x_ratio);
                int src_y = (int)(y * y_ratio);
                
                // Clamp to bounds
                src_x = src_x < input_w ? src_x : input_w - 1;
                src_y = src_y < input_h ? src_y : input_h - 1;
                
                for (int c = 0; c < channels; c++) {
                    int src_idx = (src_y * input_w + src_x) * channels + c;
                    int dst_idx = (y * output_w + x) * channels + c;
                    output[dst_idx] = input[src_idx];
                }
            }
        }
    }
    
    void applyColorMap(uint8_t* mask, int width, int height, uint8_t* colored_output) {
        // Simple color map for visualization
        const uint8_t colors[NUM_FOOD_CLASSES][3] = {
            {0, 0, 0},       // Background - Black
            {139, 69, 19},   // Chocolate - Brown
            {255, 255, 0},   // Cheese/Butter - Yellow
            {255, 255, 255}, // Milk - White
            {255, 255, 224}, // Egg - Light Yellow
            {255, 0, 0},     // Apple - Red
            {255, 255, 0},   // Banana - Yellow
            {255, 20, 147},  // Strawberry - Pink
            {255, 165, 0},   // Mango - Orange
            {255, 140, 0},   // Orange - Dark Orange
            {139, 0, 0},     // Beef Steak - Dark Red
            {255, 192, 203}, // Pork - Pink
            {255, 228, 181}, // Chicken/Duck - Beige
            {128, 0, 0},     // Sauce - Maroon
            {210, 180, 140}, // Bread - Tan
            {255, 215, 0},   // Corn - Gold
            {255, 228, 196}, // Pasta - Bisque
            {255, 228, 196}, // Noodles - Bisque
            {255, 255, 255}, // Rice - White
            {245, 245, 220}, // Tofu - Beige
            {128, 0, 128},   // Eggplant - Purple
            {160, 82, 45},   // Potato - Brown
            {255, 255, 255}, // Garlic - White
            {255, 99, 71},   // Tomato - Tomato Red
            {0, 128, 0},     // Scallion - Green
            {255, 215, 0},   // Ginger - Gold
            {0, 255, 0},     // Lettuce - Green
            {0, 255, 0},     // Cucumber - Green
            {255, 140, 0},   // Carrot - Orange
            {0, 128, 0},     // Cabbage - Green
            {0, 128, 0},     // Green Bean - Green
            {255, 248, 220}, // King Oyster Mushroom - Cornsilk
            {139, 69, 19}    // Shiitake Mushroom - Brown
        };
        
        for (int i = 0; i < width * height; i++) {
            uint8_t class_id = mask[i];
            if (class_id < NUM_FOOD_CLASSES) {
                colored_output[i * 3] = colors[class_id][0];     // R
                colored_output[i * 3 + 1] = colors[class_id][1]; // G
                colored_output[i * 3 + 2] = colors[class_id][2]; // B
            } else {
                colored_output[i * 3] = 0;     // R
                colored_output[i * 3 + 1] = 0; // G
                colored_output[i * 3 + 2] = 0; // B
            }
        }
    }
    
    float calculateIoU(uint8_t* mask1, uint8_t* mask2, int size, int class_id) {
        int intersection = 0;
        int union_count = 0;
        
        for (int i = 0; i < size; i++) {
            bool m1_has_class = (mask1[i] == class_id);
            bool m2_has_class = (mask2[i] == class_id);
            
            if (m1_has_class && m2_has_class) {
                intersection++;
            }
            if (m1_has_class || m2_has_class) {
                union_count++;
            }
        }
        
        return union_count > 0 ? (float)intersection / union_count : 0.0f;
    }
}
