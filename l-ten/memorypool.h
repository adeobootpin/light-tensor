#ifndef MEMORY_POOL
#define MEMORY_POOL

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

#include <unordered_map>

struct block_info
{
	uint64_t block_size;
	void* block_address;
};

class block_list
{
public:
	block_list() : head_(0), tail_(0), full_(false) {}
	~block_list() {}

	bool init(unsigned int total_blocks)
	{
		blocks_ = new block_info[total_blocks];
		if (!blocks_)
		{
			return false;
		}

		total_blocks_ = total_blocks;

		return true;
	}

	int AddBlock(void* block, uint64_t size)
	{
		if (full_)
		{
			return -1;
		}

		blocks_[tail_].block_size = size;
		blocks_[tail_].block_address = block;
		tail_++;

		if (tail_ == total_blocks_)
		{
			tail_ = 0;
		}

		if (tail_ == head_)
		{
			full_ = true;
		}

		return 0;
	}

	void* RemoveBlock()
	{
		void* block;

		if (tail_ == head_ && !full_)
		{
			return nullptr; // empty
		}

		block = blocks_[head_].block_address;

		head_++;
		if (head_ == total_blocks_)
		{
			head_ = 0;
		}

		full_ = false;

		return block;
	}

private:
	unsigned int head_;
	unsigned int tail_;
	unsigned int total_blocks_;
	bool full_;
	block_info* blocks_;
};


// creates total_pools memory pools (each pool is a list of total_blocks_per_pool blocks)
// each pool caches blocks of < pow(pool_index + 1, 2) byte blocks
// pool_[0] caches blocks of size < 2 bytes
// pool_[1] caches blocks of size < 4 bytes
class MemoryPool
{
public:
	MemoryPool(unsigned int total_pools, unsigned int total_blocks_per_pool) : total_pools_(total_pools), total_blocks_per_pool_(total_blocks_per_pool) { Init(); }
	~MemoryPool() {}

	bool Init()
	{
		unsigned int i;
		pool_ = new block_list[total_pools_];

		for (i = 0; i < total_pools_; i++)
		{
			if (!pool_[i].init(total_blocks_per_pool_))
			{
				return false;
			}
		}

		return true;
	}

	void SetMemoryAllocator(void*(*alloc_fn)(uint64_t size), void(*free_fn)(void*))
	{
		alloc_fn_ = alloc_fn;
		free_fn_ = free_fn;
	}

	void* AllocateMemory(uint64_t size)
	{
		void* memory = nullptr;
		unsigned long pool_index;

		if (!size)
		{
			return memory;
		}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		_BitScanReverse64(&pool_index, size); //__builtin_clz for gcc ?
#else    
		pool_index = 63 - __builtin_clzll(size);
#endif

		if (pool_index < total_pools_)
		{
			memory = pool_[pool_index].RemoveBlock();
			if (!memory)
			{
				memory = alloc_fn_(static_cast<uint64_t>(0x1) << (pool_index + 1));
			}
		}

		if (memory)
		{
			memory_map_[reinterpret_cast<uint64_t>(memory)] = size;
		}

		return memory;
	}

	void FreeMemory(void* memory)
	{
		uint64_t size;
		unsigned long pool_index;
		int ret;

		auto it = memory_map_.find(reinterpret_cast<uint64_t>(memory));
		if (it == memory_map_.end())
		{
			assert(0);
		}

		size = it->second;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		_BitScanReverse64(&pool_index, size); //__builtin_clz for gcc ?
#else
    pool_index = 63 - __builtin_clzll(size);
#endif

		ret = pool_[pool_index].AddBlock(memory, size);
		if (ret)
		{
			memory_map_.erase(reinterpret_cast<uint64_t>(memory));
			free_fn_(memory);
		}
	}

private:
	unsigned int total_pools_;
	unsigned int total_blocks_per_pool_;

	block_list* pool_;
	void* (*alloc_fn_)(uint64_t size);
	void(*free_fn_)(void*);
	std::unordered_map<uint64_t, uint64_t> memory_map_;
};


extern MemoryPool memory_pool;

#endif // ! MEMORY_POOL
