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


struct OffsetCalc_generic
{
	OffsetCalc_generic(const int num_dims, const uint64_t* strides )
	{
		uint32_t divisor;
		uint32_t shift;
		int i;

		ndims = num_dims;

		for (i = 0; i < ndims; i++)
		{
			divisor = (uint32_t)strides[i];

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

			strides_[i] = (uint32_t)strides[i];
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
		for (i = 0; i < ndims; ++i)
		{
			coordinate = ((((uint64_t)index * div[i].magic) >> 32) + index) >> div[i].shift;
			index = index - coordinate * div[i].divisor;

			offs += coordinate * strides_[i];
		}

		return offs;
	}

	uint32_t strides_[MAX_DIMS];
	Divider div[MAX_DIMS];
	int ndims;

};






struct OffsetCalc_mean_var
{
	OffsetCalc_mean_var(const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;
		int index_1;
		int index_2;
		int naxes;
		uint64_t numels;

		naxes = ndims_src - ndims_dst;

		bitmask_ = 0;

		for (i = 0; i < naxes; i++)
		{
			bitmask_ |= (1 << axes[i]);
		}

		index_1 = 0;
		index_2 = 0;

		for (i = 0; i < ndims_src; i++)
		{
			if ((bitmask_ & (1 << i)))
			{
				axes_strides_src[index_1] = strides_src[i];				
				axes_dims[index_1] = dims_src[i];
				index_1++;
			}
			else
			{
				non_axis_strides_src[index_2++] = strides_src[i];
			}
		}

		assert(index_1 == naxes);
		assert(index_2 == ndims_dst);


		for (i = 0; i < ndims_dst; i++)
		{
			divisor = (uint32_t)strides_dst[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_dst_[i].magic = (uint32_t)magic;
			div_dst_[i].shift = shift;
			div_dst_[i].divisor = divisor;

			strides_dst_[i] = (uint32_t)strides_dst[i];
		}


		numels = 1;
		for (i = naxes - 1; i >= 0; i--)
		{
			axes_srides[i] = numels;
			numels *= axes_dims[i];
		}


		for (i = 0; i < naxes; i++)
		{
			divisor = (uint32_t)axes_srides[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_axes_[i].magic = (uint32_t)magic;
			div_axes_[i].shift = shift;
			div_axes_[i].divisor = divisor;
		}


		ndims_dst_ = ndims_dst;
		ndims_src_ = ndims_src;
		naxes_ = naxes;

	}

	LTEN_HOST_DEVICE uint32_t GetSrcOffset(uint32_t dst_index, uint32_t index) // dst index and worker (thread) index
	{
		return 0;
	}

	LTEN_HOST_DEVICE uint32_t GetPartialSrcOffset_d(uint32_t dst_index ) // compute portion of src offset contributed by non axis coordinates
	{
		uint32_t offs;

		int i;
		uint32_t coordinate;

		offs = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < 8; ++i)
		{
			if (i == ndims_dst_)
			{
				break;
			}

			coordinate = ((((uint64_t)dst_index * div_dst_[i].magic) >> 32) + dst_index) >> div_dst_[i].shift;
			dst_index = dst_index - coordinate * div_dst_[i].divisor;

			offs += coordinate * non_axis_strides_src[i];
		}

		return offs;
	}


	LTEN_HOST_DEVICE uint32_t GetPartialSrcOffset_s(uint32_t src_index)
	{
		uint32_t offs;

		int i;
		uint32_t coordinate;

		offs = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < 8; ++i)
		{
			if (i == naxes_)
			{
				break;
			}

			coordinate = ((((uint64_t)src_index * div_axes_[i].magic) >> 32) + src_index) >> div_axes_[i].shift;
			src_index = src_index - coordinate * div_axes_[i].divisor;

			offs += coordinate * axes_strides_src[i];
		}

		return offs;
	}

	uint32_t non_axis_strides_src[MAX_DIMS];
	uint32_t axes_strides_src[MAX_DIMS];
	uint32_t axes_dims[MAX_DIMS];
	uint32_t axes_srides[MAX_DIMS];


	uint32_t strides_dst_[MAX_DIMS];
	uint32_t strides_src_[MAX_DIMS];
	uint32_t squashed_strides_src_[MAX_DIMS];
	Divider div_dst_[MAX_DIMS];
	Divider div_src_[MAX_DIMS];
	Divider div_axes_[MAX_DIMS];
	int ndims_dst_;
	int ndims_src_;
	int naxes_;

	uint16_t bitmask_;

};



struct OffsetCalc_mean_var_old
{
	OffsetCalc_mean_var_old(const uint64_t* strides_dst, const uint64_t* strides_src, int ndims_dst, int ndims_src, const uint64_t* dims_src, const uint32_t* axes)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;
		int naxes;
		uint32_t squashed_dims[MAX_DIMS];
		uint64_t numels;


		ndims_dst_ = ndims_dst;

		for (i = 0; i < ndims_dst_; i++)
		{
			divisor = (uint32_t)strides_dst[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_dst_[i].magic = (uint32_t)magic;
			div_dst_[i].shift = shift;
			div_dst_[i].divisor = divisor;

			strides_dst_[i] = (uint32_t)strides_dst[i];
		}


		ndims_src_ = ndims_src;

		for (i = 0; i < ndims_src_; i++)
		{
			strides_src_[i] = (uint32_t)strides_src[i];
		}


		int index = 0;
		assert(MAX_DIMS <= sizeof(uint16_t) * 8); // increase size of bitmask if this changes
		bitmask_ = 0;
		naxes = ndims_src - ndims_dst;

		for (i = 0; i < naxes; i++)
		{
			bitmask_ |= (1 << axes[i]);
		}

		index = 0;
		for (i = 0; i < ndims_src; i++)
		{
			if ((bitmask_ & (1 << i)))
			{
				squashed_dims[index++] = dims_src[i];
			}
		}
		assert(index == naxes); // sanity check

		numels = 1;
		for (i = naxes - 1; i >= 0; i--)
		{
			squashed_strides_src_[i] = numels;
			numels *= squashed_dims[i];
		}
		

		for (i = 0; i < naxes; i++)
		{
			divisor = (uint32_t)squashed_strides_src_[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_squashed_src_[i].magic = (uint32_t)magic;
			div_squashed_src_[i].shift = shift;
			div_squashed_src_[i].divisor = divisor;
		}
	}

	LTEN_HOST_DEVICE uint32_t GetSrcOffset(uint32_t dst_index, uint32_t index) // dst index and worker (thread) index
	{
		uint32_t src_offset;

		int i;
		int index_dst;
		int index_squ_src;
		uint32_t coordinate;

		src_offset = 0;
		index_dst = 0;
		index_squ_src = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < ndims_src_; ++i)
		{
			if ((bitmask_ & (1 << i)))
			{
				coordinate = ((((uint64_t)index * div_squashed_src_[index_squ_src].magic) >> 32) + index) >> div_squashed_src_[index_squ_src].shift; // coordinate of index
				index = index - coordinate * div_squashed_src_[index_squ_src].divisor;
				index_squ_src++;
			}
			else
			{
				coordinate = ((((uint64_t)dst_index * div_dst_[index_dst].magic) >> 32) + dst_index) >> div_dst_[index_dst].shift; // coordinate of dst index
				dst_index = dst_index - coordinate * div_dst_[index_dst].divisor;
				index_dst++;
			}

			src_offset += coordinate * strides_src_[i];
		}

		return src_offset;

	}

	uint32_t strides_dst_[MAX_DIMS];
	uint32_t strides_src_[MAX_DIMS];
	uint32_t squashed_strides_src_[MAX_DIMS]; // source strides with axes dims removed
	Divider div_dst_[MAX_DIMS];
	Divider div_src_[MAX_DIMS];
	Divider div_squashed_src_[MAX_DIMS];
	int ndims_dst_;
	int ndims_src_;

	uint16_t bitmask_;
};

struct OffsetCalc_mean_std_simple
{
	OffsetCalc_mean_std_simple(const uint64_t* __restrict strides_dst, const uint64_t* __restrict strides_src, int ndims_dst, int ndims_src, const uint64_t* __restrict dims_src, const uint32_t* __restrict axes)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;
		int index;
		int index2;
		uint16_t bitmask;
		int naxes;
		uint32_t workspace_dims[MAX_DIMS];
		uint32_t workspace_pseudo_strides[MAX_DIMS];
		uint64_t numels;

		for (i = 0; i < ndims_dst; i++)
		{
			divisor = (uint32_t)strides_dst[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_dst_[i].magic = (uint32_t)magic;
			div_dst_[i].shift = shift;
			div_dst_[i].divisor = divisor;
		}

		assert(MAX_DIMS <= sizeof(uint16_t) * 8); // increase size of bitmask if this changes
		bitmask = 0;
		naxes = ndims_src - ndims_dst;

		for (i = 0; i < naxes; i++)
		{
			bitmask |= (1 << axes[i]);
		}

		//
		// If ndims_src = 5 and axes are { 0, 2 } for example, then for dst coordinates a, b, c
		// return src offset of 0, a, 0, b, c
		// Create squashed strides array of len 3 (i.e. ndims_dst) for quick evaluation
		//
		index = 0;
		index2 = 0;
		for (i = 0; i < ndims_src; i++)
		{
			if ((bitmask & (1 << i)))
			{
				workspace_strides_[index2++] = strides_src[i];
			}
			else
			{
				squashed_strides_src_[index++] = strides_src[i];
			}
		}

		assert(index == ndims_dst); // sanity check
		ndims_dst_ = ndims_dst;
		ndims_src_ = ndims_src;


		index = 0;
		for (i = 0; i < ndims_src; i++)
		{
			if ((bitmask & (1 << i)))
			{
				workspace_dims[index++] = dims_src[i];
			}
		}

		numels = 1;
		for (i = naxes - 1; i >= 0; i--)
		{
			workspace_pseudo_strides[i] = numels;
			numels *= workspace_dims[i];
		}

		for (i = 0; i < naxes; i++)
		{
			divisor = (uint32_t)workspace_pseudo_strides[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_workspace_[i].magic = (uint32_t)magic;
			div_workspace_[i].shift = shift;
			div_workspace_[i].divisor = divisor;
		}

		naxes_ = naxes;
	}




	LTEN_HOST_DEVICE uint32_t GetOffsets(uint32_t index)
	{
		uint32_t src_offset;

		int i;
		uint32_t coordinate;

		src_offset = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < ndims_dst_; ++i)
		{
			coordinate = ((((uint64_t)index * div_dst_[i].magic) >> 32) + index) >> div_dst_[i].shift;
			index = index - coordinate * div_dst_[i].divisor;

			src_offset += coordinate * squashed_strides_src_[i];
		}

		return src_offset;
	}

	LTEN_HOST_DEVICE uint32_t GetWorkspaceOffsets(uint32_t index)
	{
		uint32_t src_offset;

		int i;
		uint32_t coordinate;

		src_offset = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < naxes_; ++i)
		{
			coordinate = ((((uint64_t)index * div_workspace_[i].magic) >> 32) + index) >> div_workspace_[i].shift;
			index = index - coordinate * div_workspace_[i].divisor;

			src_offset += coordinate * workspace_strides_[i];
		}

		return src_offset;

	}


	Divider div_dst_[MAX_DIMS];
	Divider div_workspace_[MAX_DIMS];
	uint32_t squashed_strides_src_[MAX_DIMS];
	uint32_t workspace_strides_[MAX_DIMS];
	int ndims_dst_;
	int ndims_src_;
	int naxes_;
};


struct OffsetCalc_repeat
{
	OffsetCalc_repeat(const uint64_t* strides_dst, const uint64_t* strides_src, const uint64_t* dims_src, int ndims)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;

		for (i = 0; i < ndims; i++)
		{
			divisor = (uint32_t)strides_dst[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_dst_[i].magic = (uint32_t)magic;
			div_dst_[i].shift = shift;
			div_dst_[i].divisor = divisor;


			strides_src_[i] = static_cast<uint32_t>(strides_src[i]);
		}

		for (i = 0; i < ndims; i++)
		{
			divisor = (uint32_t)dims_src[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_dims_[i].magic = (uint32_t)magic;
			div_dims_[i].shift = shift;
			div_dims_[i].divisor = divisor;
		}


		ndims_ = ndims;
	}

	LTEN_HOST_DEVICE uint32_t GetOffsets(uint32_t index)
	{
		uint32_t src_offset;

		int i;
		uint32_t coordinate;
		uint32_t mod;

		src_offset = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < ndims_; ++i)
		{
			coordinate = ((((uint64_t)index * div_dst_[i].magic) >> 32) + index) >> div_dst_[i].shift;
			index = index - coordinate * div_dst_[i].divisor;


			mod = ((((uint64_t)coordinate * div_dims_[i].magic) >> 32) + coordinate) >> div_dims_[i].shift;
			mod = coordinate - mod * div_dims_[i].divisor;
			src_offset += mod * strides_src_[i]; //src_offset += (coordinate % dims_src[j]) * strides_src[j]; // convert from dst coordinate to src coordinate, then compute src offset

		}

		return src_offset;
	}


	Divider div_dst_[MAX_DIMS];
	Divider div_dims_[MAX_DIMS];
	uint32_t strides_src_[MAX_DIMS];
	int ndims_;
};


struct OffsetCalc_repeat_interleave
{
	OffsetCalc_repeat_interleave(const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* cummulative_times, int ndims, int nrepeats, int dim)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;

		for (i = 0; i < ndims; i++)
		{
			divisor = (uint32_t)strides_dst[i];

			for (shift = 0; shift < 32; shift++)
			{
				if ((1U << shift) >= divisor)
				{
					break;
				}
			}

			uint64_t one = 1;
			uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;

			div_dst_[i].magic = (uint32_t)magic;
			div_dst_[i].shift = shift;
			div_dst_[i].divisor = divisor;

			strides_array_[i] = static_cast<uint32_t>(strides_src[i]);
		}

		memcpy(cummulative_times_, cummulative_times, sizeof(uint32_t) * (nrepeats + 1));

		ndims_ = ndims;
		nrepeats_ = nrepeats;
		dim_ = dim;
	}

	LTEN_HOST_DEVICE uint32_t GetOffsets(uint32_t index)
	{
		uint32_t src_offset;

		int i;
		int j;
		uint32_t coordinate;

		src_offset = 0;

#ifdef __NVCC__
#pragma unroll
#endif
		for (i = 0; i < ndims_; ++i)
		{
			coordinate = ((((uint64_t)index * div_dst_[i].magic) >> 32) + index) >> div_dst_[i].shift;
			index = index - coordinate * div_dst_[i].divisor;
			if (i == dim_)
			{
				for (j = 0; j < nrepeats_; j++)
				{
					if (coordinate < cummulative_times_[j + 1]) // find correct window for coordinate
					{
						break;
					}
				}
				src_offset += j * strides_array_[i]; // coordinate is j
			}
			else
			{
				src_offset += coordinate * strides_array_[i];
			}
		}

		return src_offset;
	}


	Divider div_dst_[MAX_DIMS];
	uint32_t strides_array_[MAX_DIMS];
	uint32_t cummulative_times_[MAX_DIMS];
	int ndims_;
	int nrepeats_;
	int dim_;
};


struct OffsetCalc_permutaion
{
	OffsetCalc_permutaion(const uint64_t* strides_dst, const uint64_t* strides_src, const uint32_t* permutaions, int ndims)
	{
		uint32_t divisor;
		uint32_t shift;
		int i;

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

			div_[i].magic = (uint32_t)magic;
			div_[i].shift = shift;
			div_[i].divisor = divisor;

			strides_dst_[i] = (uint32_t)strides_dst[i];

			for (int j = 0; j < ndims; j++)
			{
				if (i == permutaions[j])
				{
					permutaions_[i] = j;
				}
			}
			
		}

		ndims_ = ndims;
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
			if (i == ndims_)
			{
				break;
			}

			coordinate = ((((uint64_t)index * div_[i].magic) >> 32) + index) >> div_[i].shift;
			index = index - coordinate * div_[i].divisor;

			offs += coordinate * strides_dst_[permutaions_[i]];
		}

		return offs;

	}

	Divider div_[MAX_DIMS];
	uint32_t strides_dst_[MAX_DIMS];
	uint32_t permutaions_[MAX_DIMS];
	int ndims_;

};


#endif // OFFSET_CALC_H