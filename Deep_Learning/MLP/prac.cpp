#include <iostream>

struct Tensor {
    int rows;
    int cols;
    float* data;  // The raw memory pointer

    // Constructor: Allocate memory dynamically
    Tensor(int r, int c) {
        rows = r;
        cols = c;
        data = new float[rows * cols]; // Raw contiguous block in RAM
    }

    // Destructor: Free memory to avoid memory leaks
    ~Tensor() {
        delete[] data;
    }
};