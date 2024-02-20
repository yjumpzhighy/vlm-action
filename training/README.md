# Deepspeed
  ## 1.optimizer states.
      #USE a linear layer, diff the sgd/adam states parameters:    
      
      #SGD optimizer
      '''  
      defaultdict(<class 'dict'>, {Parameter containing:
            tensor([[ 0.2335, -0.4083,  0.1899, -0.1726,  0.2455],
            [ 0.0609,  0.0835,  0.2872, -0.2975, -0.3320],
            [-0.0181, -0.2002, -0.3916,  0.0904,  0.2664]], device='cuda:0',
            requires_grad=True): {'momentum_buffer': None}})
      '''

      #Adam optimizer
      '''
      defaultdict(<class 'dict'>, {Parameter containing:
          tensor([[ 0.1231, -0.2155,  0.3782,  0.0744,  0.0497],
          [ 0.1089,  0.2391,  0.4422, -0.2361,  0.0235],
          [ 0.4218, -0.0264, -0.4429,  0.0436,  0.1418]], device='cuda:0',
          requires_grad=True): {'step': 1, 'exp_avg': tensor([[-0.0054, -0.0046, -0.0041, -0.0057, -0.0046],
          [ 0.0036,  0.0030,  0.0027,  0.0037,  0.0030],
          [-0.0092, -0.0079, -0.0069, -0.0097, -0.0078]], device='cuda:0'), 'exp_avg_sq': tensor(
          [[2.9528e-06, 2.1347e-06, 1.6599e-06, 3.2591e-06, 2.1137e-06],
          [1.2711e-06, 9.1896e-07, 7.1447e-07, 1.4029e-06, 9.0980e-07],
          [8.5557e-06, 6.1855e-06, 4.8096e-06, 9.4437e-06, 6.1243e-06]],
          device='cuda:0')}})
      '''
      #Compare
      As shown above, Adam keeps weight, exp_avg, exp_avg_sq as states. In another word, 3x memory usage.



  ## 2. deepspeed overview 
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

      def initialize(args, model, optimizer, model_parameters, training_data, lr_scheduler, 
          mpu, dist_init_required, collate_fn, config, config_params):
          # build pipeline engine, if model instance PipelineModule
          engine = PipelineEngine(...)
          # build hybrid engine, able to train/inference simulatiously, fit for RLHF
          engine = DeepSpeedHybridEngine(...)
          # build normal engine
          engine = DeepSpeedEngine(...)

      class DeepSpeedEngine(torch.nn.Module)
        # Optimzier optimization is the main concept of ZeRO based options
        # 1. basic optimizer optimization if optimizer passed in, don't recommend
        # 2. create a optimizer based on config (recommended), if stage > 1
        if stage==2：
          optimizer = DeepSpeedZeroOptimizer(...)
        elif stage==3:
          optimizer = DeepSpeedZeRoOffload()

        
  ## 3.deepspeed stage2
  Stage 1,2 focus on segment model gradients and optimizer states. Idealy, each process (GPU) keeps partial of the  
  gradients and optimizer states, and do broadcast when necessary, to reduce the GPU memory usage.  
  Also, cpu offload is supported as well.   
      
  In the optimizer, params_group will be segmented, thus each process keeps current belonging parameters and delete   
  the rest. This strategy ensure only current part of gradient will be calcuated and stored, same with the optimzier   
  states.    

        #in each optimizer.param_groups, perform round_robin distribute. for examaple, to partition into n parts:
        #usually n is the world_size (usually gpus) in the group

        #get all trainable param in optimizer group
        tensor_list_in_group = []
        for param in optimizer.param_groups[0]['params']:
              if param.requires_grad:
                tensor_list_in_group.append(param)

        #round robin to reorder weights tensors with ranks
        def _round_robin_reorder(tensor_list_in_group, num_partitions): 
          #Put all param tensors into num_partitions bucket
          partition_tensors = {}
          for i, tensor in enumerate(tensor_list_in_group):
              j = i % num_partitions
              if not j in partition_tensors:
                  partition_tensors[j] = []
              partition_tensors[j].append((i, tensor))
          reordered_tensors = []
          reordered_indices = {}
          for partition_index in partition_tensors.keys():
              for i, (original_index, tensor) in enumerate(partition_tensors[partition_index]):
                  reordered_indices[original_index] = len(reordered_tensors)
                  reordered_tensors.append(tensor)

        #flatten weights tensors into 1d tensor
        def flatten_dense_tensors_aligned(reordered_tensors):
          #flatten all tensors into one 1d tensor 
          group_flat = torch.flatten_dense_tensor(reordered_tensors)

        # partition the flatten tensor to parts
        def get_data_parallel_partitions(group_flat, num_partitions):
          #divide the flat weights into near num_partitions equally
          #each process will compute on a different part of the partition 
          partitions = []
          total_num_elements = tensor.numel()
          base_size = total_num_elements // num_partitions
          remaining = total_num_elements % num_partitions

          start = 0
          for id in range(num_partitions):
              partition_size = base_size
              if id < remaining:
                  partition_size = partition_size + 1
              partitions.append(tensor.narrow(0, start, partition_size))
              start = start + partition_size
          # Note partitioned equally, means a tensor maybe divided to two parts. 

        # copy each partition into device.
        partition_id = dist.get_rank()
        optimizer.param_groups[0]['params'] = [partitions[partition_id].to(device).clone().float().detach()]
        optimizer.param_groups[0]['params'].requires_grad = True

  ## 4.deepspeed stage3
  Stage 3 focus on segment model gradients, optimizer states, as well as model weights. Idealy, each process (GPU) keeps    
  partial  of the gradients and optimizer states and model parameters, and do broadcast when necessary, to reduce the GPU    
  memory usage.     
  Also, cpu offload is supported as well.     
      
  In the model, weights will be segmented, thus each process keeps current belonging weights and delete the rest.    
  This strategy ensure only current model weights will be calcuated and stored.    
      
  Since model weights get partitioned, thus in each process, its optimzier states "partitioned" natually without extra    
  operations.     

        #model weights convert
        param_list = module.parameters(recurse=True)  
        def _convert_to_zero_parameters(param_list):
            for param in param_list:
              #copy weights to local rank device in group. usually device 0
              param.data = param.data.to(local_rank)
              #broadcast to process groupm thus all ranks in group have it.
              dist.broadcast(param, 0, self.get_dp_process_group())

              #flatten the param tensor to 1d tensor
              one_dim_param = param.contiguous().view(-1)

              #copy partitioned model weights to current rank
              tensor_size = self._aligned_size(param)
              partition_size = tensor_size // self.num_partitions
              start = partition_size * get_partition_rank()
              end = start + partition_size
              src_tensor = one_dim_param.narrow(0, start, partition_size)
              param.ds_tensor.copy_(src_tensor)

        #FWD/BWD
        def setup_zero_stage3_hooks():
            #do register hooks for restore/release weights

