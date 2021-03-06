#ifndef TESTS_H
#define TESTS_H

// unit tests
int add_test_1(bool run_on_gpu = false);
int add_test_2(bool run_on_gpu = false);
int sub_test_1(bool run_on_gpu = false);
int sub_test_2(bool run_on_gpu = false);
int sub_test_1_uint8();
int sub_test_2_uint8();
int mul_test_1(bool run_on_gpu = false);
int mul_test_2(bool run_on_gpu = false);
int mul_test_1_int32();
int div_test_1(bool run_on_gpu = false);
int div_test_2(bool run_on_gpu = false);
int matmul_test(bool run_on_gpu = false);
int matmul_test_uint8();
int exp_test(bool run_on_gpu = false);
int log_test(bool run_on_gpu = false);
int sig_test(bool run_on_gpu = false);
int tanh_test(bool run_on_gpu = false);
int scalar_mul(bool run_on_gpu = false);
int sum_test(bool run_on_gpu = false);
int fc_test(bool run_on_gpu = false);


// neural network tests
int neural_network_test();
int MNIST_test(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels);
int MNIST_test_gpu(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels);
int quantized_MNIST_test(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels);

#endif // TESTS_H