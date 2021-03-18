#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace lten {
	class Optimizer
	{
	public:
		Optimizer() : num_params_(0), network_params_ptr_(0) {}
		~Optimizer() {}

		void attach_network(NeuralNetwork& net)
		{
			network_params_ptr_ = net.get_parameters(&num_params_);
			setup_optimizer();
		}

		virtual void setup_optimizer() = 0;
		virtual void step() = 0;

		void zero_grad()
		{
			int i;

			for (i = 0; i < num_params_; i++)
			{
				network_params_ptr_[i].param_->clear_gradients();
			}

		}

	protected:
		int num_params_;
		NetworkParms* network_params_ptr_;
	};


	class SGDOptimizer : public Optimizer
	{
	public:
		SGDOptimizer() : lr_(0.01f), mo_(0.9f), wd_(0.0005f) {}
		~SGDOptimizer() {}

		void setup_optimizer();
		void step();

		void set_learning_rate(float lr) { lr_ = lr; }
		float get_learning_rate() { return lr_; }

		void set_momentum(float mo) { mo_ = mo; }
		float get_momentum() { return mo_; }

		void set_weight_decay(float wd) { wd_ = wd; }
		float get_weight_decay() { return wd_; }

	private:
		float lr_;
		float wd_;
		float mo_;

	};


	class AdamOptimizer : public Optimizer
	{
	public:
		AdamOptimizer() : lr_(0.01f), beta1_(0.9f), beta2_(0.999f), iteration_(0) {}
		~AdamOptimizer() {}

		void setup_optimizer();
		void step();

		void set_learning_rate(float lr) { lr_ = lr; }
		float get_learning_rate() { return lr_; }

		void set_beta1(float beta1) { beta1_ = beta1; }
		float get_beta1() { return beta1_; }

		void set_beta2(float beta2) { beta2_ = beta2; }
		float get_beta2() { return beta2_; }

	private:
		uint64_t iteration_;
		float lr_;
		float beta1_;
		float beta2_;
	};

} // namespace lten

#endif //OPTIMIZER_H

