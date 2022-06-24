#ifndef OFFSET_CALC_H
#define OFFSET_CALC_H


#ifdef __NVCC__
#define LTEN_HOST_DEVICE __host__ __device__
#else
#define LTEN_HOST_DEVICE
#endif

struct Divider
{
	uint32_t magic;
	uint32_t shift;
	uint32_t divisor;
};

struct OffsetCalc
{
	OffsetCalc(const int num_dims, const uint64_t* strides_dst, const uint64_t* strides_src)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;

		ndims = num_dims;

		for (i = 0; i < ndims; i++)
		{
			divisor = (uint32_t)strides_src[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div[i].magic = (uint32_t)magic;
			div[i].shift = shift;
			div[i].divisor = divisor;

			strides_dst_[i] = (uint32_t)strides_dst[i];
		}
	}

	LTEN_HOST_DEVICE uint32_t GetOffset(uint32_t index)
	{
		uint32_t offs;

		int i;
		uint32_t coordinate;

		offs = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < MAX_DIMS; ++i)
		{
			if (i == ndims)
			{
				break;
			}

			coordinate = ((((uint64_t)index * div[i].magic) >> 32) + index) >> div[i].shift;
			index = index - coordinate * div[i].divisor;

			offs += coordinate * strides_dst_[i];
		}

		return offs;

	}

	LTEN_HOST_DEVICE void GetOffset(uint32_t index, uint32_t* offset)
	{
		uint32_t offs;

		int i;
		uint32_t coordinate;

		offs = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < MAX_DIMS; ++i)
		{
			if (i == ndims)
			{
				break;
			}

			coordinate = ((((uint64_t)index * div[i].magic) >> 32) + index) >> div[i].shift;
			index = index - coordinate * div[i].divisor;

			offs += coordinate * strides_dst_[i];
		}

		*offset = offs;
	}

	uint32_t strides_dst_[MAX_DIMS];
	Divider div[MAX_DIMS];
	int ndims;

};

#endif // OFFSET_CALC_H