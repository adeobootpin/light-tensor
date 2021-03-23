#ifndef THREADPOOL2_H
#define THREADPOOL2_H
/*
MIT License

Copyright (c) 2021 adeobootpin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
semaphore class from: Maxim Egorushkin https://stackoverflow.com/questions/4792449/c0x-has-no-semaphores-how-to-synchronize-threads
*/

#include <thread>
#include <queue>
#include <condition_variable>


class semaphore
{
private:
	std::mutex mutex_;
	std::condition_variable condition_;
	unsigned long count_ = 0;

public:
	void notify()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		++count_;
		condition_.notify_one();
	}

	void wait()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		while (!count_)
		{
			condition_.wait(lock);
		}
		--count_;
	}
};


class wait_multiple
{
private:
	std::mutex mutex_;
	std::condition_variable condition_;
	unsigned long count_ = 0;
	unsigned long desired_count_;

public:

	void set_desired_count(int desired_count)
	{
		desired_count_ = desired_count;
	}

	void notify()
	{
		std::lock_guard<std::mutex> lock(mutex_);
		++count_;
		condition_.notify_one();
	}

	void wait() // only one thread gets released
	{
		std::unique_lock<std::mutex> lock(mutex_);
		while (count_ != desired_count_)
		{
			condition_.wait(lock);
		}
		count_ = 0;
	}
};

typedef struct _Job
{
	void* param_ptr; // pointer to data to pass to function
	int block_index; // specifies index of a particular job 0, 1, 2, ... etc.  For example if a job is scheduled for 3 threads, then 3 job structs will be enqueued with indices 0, 1, 2.
	int total_blocks; // total number of threads requested for the job
	int* return_value;
	wait_multiple* completion_notifyer;
	int(*thread_proc)(void*, int, int);
}Job;

typedef struct _ThreadProcData
{
	std::queue<Job>* job_queue_ptr;
	std::mutex* job_queue_mutex_ptr;
	int pool_thread_id;
	semaphore* execute_gate_ptr;
}ThreadProcData;

class ThreadPool
{
public:
	ThreadPool();
	~ThreadPool();

	bool Init(unsigned int thread_count);
	void WaitForTaskCompletion(int** retvals);
	int Execute(int(*fn)(void*, int block_index, int total_blocks), void* param, int total_blocks);
	int get_thread_count() { return thread_count_; }

private:
	unsigned int thread_count_;
	std::queue<Job> job_queue_;
	std::mutex job_queue_mutex_;
	int* return_values_;

	semaphore execute_gate_;
	wait_multiple completion_notifyer_;
};

#endif // THREADPOOL2_H