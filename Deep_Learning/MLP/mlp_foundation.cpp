#include <iostream>
#include <cassert>
#include <random>

struct Tensor {
    int rows;
    int cols;
    float* data;

    //constructor : Allocate Memory Dynamically
    Tensor(int r, int c){
            rows = r;
            cols = c;
            data = new float[rows * cols];
    }

    //destructor : Free memory to avoid memory leaks
    ~Tensor(){
        delete[] data;
    }

    float& operator()(int row_idx, int col_idx){
        return data[(row_idx * cols) + col_idx];
    }

    const float& operator()(int row_idx, int col_idx) const{
        return data[(row_idx * cols) + col_idx];
    }

    //weight initializaton
    void randomize(float mean = 0.0f, float stddev = 0.1f){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, stddev);

        int total_elements = rows * cols;
        for (int i = 0; i < total_elements; ++i) {
            data[i] = dist(gen);
        }
    }
};

// 1. Matrix Multiplication (Standalone Verb)
Tensor matmul(const Tensor& a, const Tensor& b){
    assert(a.cols == b.rows && "Matrix Dimensions do not match for multiplication!");

    Tensor result(a.rows, b.cols);

    for (int i=0; i < a.rows; ++i){
        for (int j = 0; j< b.cols; ++j){
            float sum = 0.0f;

            for (int k = 0; k< a.cols; ++k){
                sum += a(i,k) * b(k,j);
            }
            result(i,j) = sum;
        }
    }
    return result;
} // <--- matmul ends cleanly here!

// 2. Add Bias (Standalone Verb - Outside matmul)
void add_bias(Tensor& mat, const Tensor& bias){
    assert(mat.cols == bias.cols && "Bias columns must match in order to add the bias to the matrix");

    for (int i = 0; i < mat.rows; ++i){
        for (int j = 0; j < mat.cols; ++j){
            mat(i, j) += bias(0, j);
        }
    }
}

// 3. ReLU Activation (Standalone Verb - Outside matmul)
void relu(Tensor& mat){
    int total_elements = mat.rows * mat.cols;
    
    for (int i = 0; i < total_elements; ++i){
        if (mat.data[i] < 0.0f){
            mat.data[i] = 0.0f; // kills negative values
        }
    }
}


struct LinearLayer{
    Tensor weights;
    Tensor bias;

    //constructor: Sets up weights and biases based on dimensions
    LinearLayer(int input_dim, int output_dim)
        :  weights(input_dim, output_dim), bias(1,output_dim){
            weights.randomize(0.0f, 0.1f);

            for (int j = 0; j < output_dim; ++j){
                bias(0,j) = 0.0f;
            }
        }


        Tensor forward(const Tensor& input){
            
            Tensor output = matmul(input, weights);
            add_bias(output, bias);
            relu(output);
            return output;
        }
};

int main(){
    Tensor input(2,3);
    input(0, 0) =  1.0f; input(0, 1) = -2.0f; input(0, 2) =  1.5f;
    input(1, 0) =  0.5f; input(1, 1) =  1.0f; input(1, 2) = -0.5f;


    //initialize a linear layer
    LinearLayer layer(3,4);

    Tensor output  = layer.forward(input);

    //printing the results
    std::cout<< "Forward pass successfull! output shape "<<output.rows<<" X "<<output.cols<<"\n";

    return 0;
}