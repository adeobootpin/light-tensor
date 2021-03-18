#include <assert.h>
#include "threadpool2.h"

void WorkerThreadProc(ThreadProcData* tpd)
{
	Job job;

	while (true)
	{
		tpd->execute_gate_ptr->wait();

		while (true)
		{
			std::unique_lock<std::mutex> lock(*tpd->job_queue_mutex_ptr);
			if (!tpd->job_queue_ptr->empty())
			{
				job = tpd->job_queue_ptr->front();
				tpd->job_queue_ptr->pop();
				lock.unlock();

				*job.return_value = job.thread_proc(job.param_ptr, job.block_index, job.total_blocks);
				job.completion_notifyer->notify();
			}
			else
			{
				lock.unlock();
				break;
			}

		}
	}
}

ThreadPool::ThreadPool()
{
	thread_count_ = 0;
	return_values_ = nullptr;
}

ThreadPool::~ThreadPool()
{
}

bool ThreadPool::Init(unsigned int thread_count)
{
	ThreadProcData* tpd;
	unsigned int i;

	if (thread_count == 0)
	{
		thread_count_ = std::thread::hardware_concurrency();
	}
	else
	{
		thread_count_ = thread_count;
	}

	return_values_ = new int[thread_count_];
	if (!return_values_)
	{
		return false;
	}

	for (i = 0; i < thread_count_; i++)
	{
		tpd = new ThreadProcData;

		tpd->job_queue_ptr = &job_queue_;
		tpd->job_queue_mutex_ptr = &job_queue_mutex_;
		tpd->execute_gate_ptr = &execute_gate_;
		tpd->pool_thread_id = i;

		std::thread(WorkerThreadProc, tpd).detach();
	}

	return true;
}


int ThreadPool::Execute(int(*fn)(void*, int block_index, int total_blocks), void* param, int total_blocks)
{
	Job job;
	int i;

	completion_notifyer_.set_desired_count(total_blocks);

	for (i = 0; i < total_blocks; i++)
	{
		job.block_index = i;
		job.total_blocks = total_blocks;
		job.return_value = &return_values_[i];
		job.param_ptr = param;
		job.thread_proc = fn;
		job.completion_notifyer = &completion_notifyer_;

		std::unique_lock<std::mutex> lock(job_queue_mutex_);
		job_queue_.push(job);
		lock.unlock();

		execute_gate_.notify();
	}


	return 0;
}

void ThreadPool::WaitForTaskCompletion(int** retvals)
{
	completion_notifyer_.wait();
	*retvals = return_values_;
}