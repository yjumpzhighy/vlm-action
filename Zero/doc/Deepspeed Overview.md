Tutorial / Deepspeed General Overview


## Simple config example
    config = {
      "train_batch_size": 16,
      "steps_per_print": 2000,
      "optimizer": {
        "type": "Adam",
        "params": {
          "lr": 0.001,
          "betas": [0.8,0.999],
          "eps": 1e-8,
          "weight_decay": 3e-7
        }
      },
      "scheduler": {
        "type": "WarmupLR",
        "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 0.001,
          "warmup_num_steps": 1000
        }
      },
      "gradient_clipping": 1.0,
      "prescale_gradients": False,
      "bf16": {
          "enabled": args.dtype == "bf16"
      },
      "fp16": {
          "enabled": args.dtype == "fp16",
          "fp16_master_weights_and_grads": False,
          "loss_scale": 0,
          "loss_scale_window": 500,
          "hysteresis": 2,
          "min_loss_scale": 1,
          "initial_scale_power": 15
      },
      "wall_clock_breakdown": False,
      "zero_optimization": {
          "stage": 2,
          "allgather_partitions": True,
          "reduce_scatter": True,
          "allgather_bucket_size": 50000000,
          "reduce_bucket_size": 50000000,
          "overlap_comm": True,
          "contiguous_gradients": True,
          "cpu_offload": False
      }
    }


## Start with building the engine
    def initialize(args, model, optimizer, model_parameters, training_data, lr_scheduler, 
      mpu, dist_init_required, collate_fn, config, config_params):
      //build deepspeed engine in one of three patterns 
      //1. build pipeline engine, if model instance PipelineModule
      engine = PipelineEngine(...)
      //2. build hybrid engine, able to train/inference simulatiously, fit for RLHF
      engine = DeepSpeedHybridEngine(...)
      //3. build normal engine
      engine = DeepSpeedEngine(...)

      //lr_scheduler will be stepped automatically during training.
      return [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]

## Let's dig deeper with deepspeed.runtime.engine.DeepSpeedEngine
    class DeepSpeedEngine(torch.nn.Module)
      //Optimzier optimization is the main concept of ZeRO based options
      //1. basic optimizer optimization if optimizer passed in, don't recommend
      //2. create a optimizer based on config (recommended), if stage > 1
      optimizer = DeepSpeedZeroOptimizer(...), if stage==2
      optimizer = DeepSpeedZeRoOffload(), if stage==3, seems stage 3 defaulted with offload
  
   
  
## For details, like stage2 DeepSpeedZeroOptimizer, notes made in corresponding docs.
