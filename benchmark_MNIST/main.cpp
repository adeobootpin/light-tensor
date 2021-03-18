#include <stdio.h>
#include <torch/torch.h>


int MNIST_libtorch(const char* MNIST_training_images, const char* MNIST_training_labels, const char* MNIST_test_images, const char* MNIST_test_labels, int epochs);
int MNIST_lten(const char* MNIST_images, const char* MNIST_labels, const char* MNIST_test_images, const char* MNIST_test_labels, int epochs);


int main(int argc, char* argv[])
{
	int epochs = 5;

	const char* MNIST_training_images;
	const char* MNIST_training_labels;
	const char* MNIST_test_images;
	const char* MNIST_test_labels;

	if (argc < 5)
	{
		printf("Usage: benchmarks path_to_MNIST_training_images path_to_MNIST_training_labels path_to_MNIST_test_images path_to_MNIST_test_labels [number_of_epochs]\n");
		return -1;
	}
	else
	{
		MNIST_training_images = argv[1];
		MNIST_training_labels = argv[2];
		MNIST_test_images = argv[3];
		MNIST_test_labels = argv[4];

		if (argc > 5)
		{
			epochs = atoi(argv[5]);
		}

	}




	std::cout << "MNIST training using libtorch library (" << epochs << " epoch(s))..." << std::endl;
	MNIST_libtorch(MNIST_training_images, MNIST_training_labels, MNIST_test_images, MNIST_test_labels, epochs);

	std::cout << "MNIST training using l-ten library (" << epochs << " epoch(s))..." << std::endl;
	MNIST_lten(MNIST_training_images, MNIST_training_labels, MNIST_test_images, MNIST_test_labels, epochs);
}


int ReadDataFromFile(const char* file_name, void** pp_data, size_t* data_size);

void ReverseBytes(unsigned char* bytes, int byte_count)
{
	int i;
	unsigned char* temp;

	temp = new unsigned char[byte_count];
	memcpy(temp, bytes, byte_count);

	for (i = 0; i < byte_count; i++)
	{
		bytes[byte_count - i - 1] = temp[i];
	}

	delete temp;
}


int LoadMNISTImages(const char* filename, float** pp_images, int* total_images)
{
	int ret;
	int i;
	int j;
	int k;
	unsigned char* data;
	size_t data_size;
	int magic;
	int image_count;
	int height;
	int width;
	unsigned char* char_index;
	unsigned char* char_images;
	float* float_images;
	int index;

	ret = ReadDataFromFile(filename, (void**)&data, &data_size);
	if (ret)
	{
		printf("Failed to load MNIST images [%s]\n", filename);
		return ret;
	}

	char_index = data;
	ReverseBytes(char_index, sizeof(int));
	memcpy(&magic, char_index, sizeof(int));

	char_index += sizeof(int);
	ReverseBytes(char_index, sizeof(int));
	memcpy(&image_count, char_index, sizeof(int));

	char_index += sizeof(int);
	ReverseBytes(char_index, sizeof(int));
	memcpy(&height, char_index, sizeof(int));

	char_index += sizeof(int);
	ReverseBytes(char_index, sizeof(int));
	memcpy(&width, char_index, sizeof(int));

	char_index += sizeof(int);

	char_images = new unsigned char[height * width];
	float_images = new float[image_count * height * width];

	index = 0;
	for (i = 0; i < image_count; i++)
	{
		for (j = 0; j < height; j++)
		{
			for (k = 0; k < width; k++)
			{
				float_images[index++] = *char_index++;
			}
		}
	}

	delete data;

	*total_images = image_count;
	*pp_images = float_images;

	return 0;
}


int LoadMNISTLabels(const char* filename, char** pp_labels)
{
	int ret;
	int i;
	unsigned char* data;
	size_t data_size;
	int magic;
	int image_count;
	unsigned char* pch;
	char* labels;

	ret = ReadDataFromFile(filename, (void**)&data, &data_size);
	if (ret)
	{
		printf("Failed to load MNIST labels [%s]\n", filename);
		return ret;
	}

	pch = data;
	ReverseBytes(pch, sizeof(int));
	memcpy(&magic, pch, sizeof(int));

	pch += sizeof(int);
	ReverseBytes(pch, sizeof(int));
	memcpy(&image_count, pch, sizeof(int));

	pch += sizeof(int);

	labels = new char[image_count];

	for (i = 0; i < image_count; i++)
	{
		double dd = (double)(*pch);
		labels[i] = (*pch++);
	}

	delete data;

	*pp_labels = labels;

	return 0;
}





