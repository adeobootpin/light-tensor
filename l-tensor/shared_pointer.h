#ifndef SHARED_POINTER_H
#define SHARED_POINTER_H

#include <assert.h>
#include <atomic>

class intrusive_ptr_type
{
public:
	intrusive_ptr_type() : ref_count_(0) {}

	virtual ~intrusive_ptr_type()
	{
		assert(ref_count_.load() == 0);
	}
	
	int add_ref() 
	{
		ref_count_++;
		return ref_count_;
	}

	int release()
	{
		ref_count_--;
		return ref_count_;
	}

	virtual void release_resources() {}

private:
	mutable std::atomic<int> ref_count_;

};


template <class RealObject>
class intrusive_ptr
{
public:
	intrusive_ptr()
	{
		real_object_ = nullptr;
	}

	intrusive_ptr(const intrusive_ptr& rhs) : real_object_(rhs.real_object_)
	{
		if (real_object_)
		{
			real_object_->add_ref();
		}
	}

	~intrusive_ptr()
	{
		int ref_count;

		if (real_object_)
		{
			ref_count = real_object_->release();
			assert(ref_count >= 0);
			if (!ref_count)
			{
				real_object_->release_resources();
				delete real_object_;
				real_object_ = nullptr;
			}
		}
	}

	intrusive_ptr(RealObject* real_object)
	{
		real_object_ = real_object;
		real_object_->add_ref();
	}


	intrusive_ptr& operator=(const intrusive_ptr& rhs) &
	{
		intrusive_ptr tmp = rhs;
		RealObject* swap;

		swap = real_object_;
		real_object_ = rhs.real_object_;
		tmp.real_object_ = swap;

		return *this;
	}

	
	RealObject* operator->() const
	{
		return real_object_;
	}
	
	RealObject* get_real_object() const
	{	
		return real_object_;
	}

private:
	RealObject* real_object_;
};

#endif //SHARED_POINTER_H