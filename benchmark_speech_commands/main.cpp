#include <stdio.h>
#include <torch/torch.h>
#include "benchmark_speech_commands.h"




int main(int argc, char* argv[])
{
	int ret;
	DATASET_DATA training_set;
	DATASET_DATA testing_set;
	int epochs = 30;
	const char* speech_commands_training_list;
	const char* speech_commands_testing_list;
	const char* speech_commands_training_dir;
	const char* speech_commands_testing_dir;


	if (argc < 5)
	{
		printf("Usage: benchmarks-speech-cmd path_to_training_file_list path_to_testing_file_list path_to_training_dir path_to_testing_dir [number_of_epochs]\n");
		return -1;
	}
	else
	{
		speech_commands_training_list = argv[1];
		speech_commands_testing_list = argv[2];
		speech_commands_training_dir = argv[3];
		speech_commands_testing_dir = argv[4];

		if (argc > 5)
		{
			epochs = atoi(argv[5]);
		}

	}


	std::cout << "loading speech commands data set..." << std::endl;;

	ret = LoadDataset(speech_commands_training_list, speech_commands_training_dir, &training_set);
	if(ret)
	{
		return -1;
	}
	ret = LoadDataset(speech_commands_testing_list, speech_commands_testing_dir, &testing_set);
	if(ret)
	{
		return -1;
	}

	std::cout << "done [trainig examples: " << training_set.total_examples << " test examples: " << testing_set.total_examples << "]\n" << std::endl;

	std::cout << "speech commands training using libtorch library (" << epochs << " epoch(s))..." << std::endl;
	speech_commands_libtorch(&training_set, &testing_set, epochs);

	std::cout << "speech commands training using l-ten library (" << epochs << " epoch(s))..." << std::endl;
	speech_commands_lten(&training_set, &testing_set, epochs);

	return 0;
}


void Normalize(float* data, int len)
{
	double mu;
	double sd;
	double sd_inv;
	double val;
	int i;
	double epsilon = 1.0e-8;

	mu = 0;

	for (i = 0; i < len; i++)
	{
		mu += data[i];
	}

	mu /= len;

	sd = 0;
	for (i = 0; i < len; i++)
	{
		val = data[i] - mu;
		sd += (val * val);
	}

	sd /= len;

	sd = sqrt(sd);
	sd_inv = 1.0 / (sd + epsilon);

	for (i = 0; i < len; i++)
	{
		data[i] = (data[i] - mu) * sd_inv;
	}
}


int LoadDataset(const char* dataset_file_list, const char* file_dir, DATASET_DATA* data_set)
{
	int ret;
	size_t bytes_read;
	int index;
	DATASET_DATA training_data_set;
	DATASET_DATA testing_data_set;
	int total_classes;
	DATASET_FILES_NAMES* dataset_file_names;
	int i;
	int j;
	char file_name[100];

	ret = LoadDataSetFileNames(dataset_file_list, &dataset_file_names, &total_classes);
	if (ret)
	{
		std::cout << "LoadDataSetFileNames failed" << std::endl;
		return ret;
	}

	data_set->total_examples = 0;

	for (i = 0; i < total_classes; i++)
	{
		data_set->total_examples += dataset_file_names[i].total_files;
	}


	data_set->audio_data = new float*[data_set->total_examples];
	if (data_set->audio_data)
	{
		memset(data_set->audio_data, 0, sizeof(float*) * data_set->total_examples);
	}

	data_set->labels = new int64_t[data_set->total_examples];

	index = 0;
	for (i = 0; i < total_classes; i++)
	{
		for (j = 0; j < dataset_file_names[i].total_files; j++)
		{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
			sprintf_s(file_name, sizeof(file_name), "%s%s/%s.bin", file_dir, dataset_file_names[i].class_name, dataset_file_names[i].file_names[j]);
#else
			snprintf(file_name, sizeof(file_name), "%s%s/%s.bin", file_dir, dataset_file_names[i].class_name, dataset_file_names[i].file_names[j]);
#endif
			ret = ReadDataFromFile(file_name, (void**)&data_set->audio_data[index], &bytes_read);
			if (ret)
			{
				std::cout << "Failed to read " << file_name << std::endl;
			}
			Normalize(data_set->audio_data[index], bytes_read / sizeof(float));
			data_set->labels[index] = i;
			index++;
		}
	}

	data_set->file_dir = file_dir;

	return 0;
}



char* get_file_name(char* path)
{
	char* file_name;
	char* ch;
	char* start;

	start = path;

	while (true)
	{
		if (*start == '/' || *start == '\\')
		{
			start++;
			break;
		}
		start++;
	}

	ch = start + strlen(start);

	if (*ch == ' ') // avoid trailing spaces
	{
		while (*ch == ' ')
		{
			ch--;
			if (ch < path)
			{
				assert(0);
			}
		}
	}

	file_name = new char[ch - start + 2];
	memcpy(file_name, start, ch - start + 1);
	file_name[ch - start + 1] = '\0';

	return file_name;
}

void get_class(char* path, char* class_name)
{
	size_t len;
	int i;

	len = strlen(path);
	for (i = 0; i < len; i++)
	{
		if (path[i] == '/' || path[i] == '\\')
		{
			class_name[i] = '\0';
			break;
		}
		class_name[i] = path[i];
	}
}

int LoadDataSetFileNames(const char* dataset_file_list, DATASET_FILES_NAMES** dataset_file_names, int* total_classes)
{
	int ret;
	size_t size;
	size_t len;
	char* data;
	char* scratch = nullptr;
	char* ch;
	int class_count;
	char tokens[] = "\r\n";
	char temp[MAX_PATH_LEN];
	char className[MAX_PATH_LEN];
	DATASET_FILES_NAMES* d_f_n;
	int index;

	ret = ReadDataFromFile(dataset_file_list, (void**)&data, &size);
	if (ret)
	{
		return ret;
	}

	memset(className, 0, sizeof(className));
	scratch = new char[size + 1];
	ch = scratch;
	memcpy(ch, data, size);
	ch[size] = '\0';

	class_count = 0;

	ch = strtok(ch, tokens);
	while (ch)
	{
		get_class(ch, temp);

		if (strcmp(className, temp))
		{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
			strcpy_s(className, sizeof(className), temp);
#else
			strncpy(className, temp, sizeof(className));
#endif
			class_count++;
		}
		ch = strtok(NULL, tokens);
	}

	if (!class_count)
	{
		return -1;
	}

	d_f_n = new DATASET_FILES_NAMES[class_count];
	memset(d_f_n, 0, sizeof(DATASET_FILES_NAMES)* class_count);

	ch = scratch;

	memcpy(ch, data, size);
	memset(className, 0, sizeof(className));
	index = 0;

	ch[size] = '\0';
	ch = strtok(ch, tokens);
	while (ch)
	{
		get_class(ch, temp);
		if (strcmp(className, temp))
		{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
			strcpy_s(className, sizeof(className), temp);
#else
			strncpy(className, temp, sizeof(className));
#endif

			d_f_n[index].class_name = new char[strlen(className) + 1];
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
			strcpy_s(d_f_n[index].class_name, strlen(temp) + 1, temp);
#else
			strncpy(d_f_n[index].class_name, temp, strlen(temp) + 1);
#endif
			d_f_n[index].total_files = 0;
			d_f_n[index].file_names = 0;

			index++;
		}

		assert(index > 0);

		d_f_n[index - 1].file_names = (char**)BlockRealloc(d_f_n[index - 1].file_names, sizeof(char*)* d_f_n[index - 1].total_files, sizeof(char*)* (d_f_n[index - 1].total_files + 1));
		d_f_n[index - 1].file_names[d_f_n[index - 1].total_files] = get_file_name(ch);

		d_f_n[index - 1].total_files++;

		ch = strtok(NULL, tokens);
	}

	*dataset_file_names = d_f_n;
	*total_classes = class_count;
	return 0;
}
