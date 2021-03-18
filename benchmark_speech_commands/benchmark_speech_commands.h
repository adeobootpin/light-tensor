#ifndef BENCHMARK_SPEECH_COMMANDS
#define BENCHMARK_SPEECH_COMMANDS

typedef struct TAG_DATASET_FILES_NAMES
{
	char* class_name;
	int total_files;
	char** file_names;
}DATASET_FILES_NAMES;

typedef struct TAG_DATASET_DATA
{
	int total_examples;
	float** audio_data;
	int64_t* labels;
	const char* file_dir;
}DATASET_DATA;

const int MAX_PATH_LEN = 260; // windows MAX_PATH

int ReadDataFromFile(const char* file_name, void** pp_data, size_t* data_size);
int LoadDataset(const char* dataset_file_list, const char* file_dir, DATASET_DATA* data_set);
int LoadDataSetFileNames(const char* dataset_file_list, DATASET_FILES_NAMES** dataset_file_names, int* total_classes);
int LoadDataSetFileNames(const char* dataset_file_list, DATASET_FILES_NAMES** dataset_file_names, int* total_classes);
void* BlockRealloc(void* current_block_ptr, int current_size, int new_size);

int speech_commands_libtorch(DATASET_DATA* training_set, DATASET_DATA* test_set, int epochs);
int speech_commands_lten(DATASET_DATA* training_set, DATASET_DATA* test_set, int epochs);


#endif //BENCHMARK_SPEECH_COMMANDS