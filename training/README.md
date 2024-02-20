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



# Memory quantitative analysis
  ## 1.Basic model parameters
  A model usually consist of (embedding, encoder/decoder, lm_head).
  ### embedding 
  embedding layer is a look-up table, with parameters:

    N = n_vocab * d_model
    (n_vocab is the size of vocab dict, d_model is the embedding size)

  In some cases, embedding layer got frozen and un-trainable.

  ### encoder/decoder
  with each multi-head attention block, it can be represented as:

    N(Q,K,V) = 3*(d_model*d_head*n_head) + d_head*n_head*d_model 
    (d_model is the block input embedding size, d_head is dimension of QKV, n_head is number of heads)

  after multi_head attention block, followed by 2-linear layers mlp:

    N(mlp) = d_model * d_ffn + d_ffn * d_model
    (d_model is mlp input embedding size, d_ffn is the projected ffn embedding size)

  ### lm_head
  simple linear layer to project feature space into probability space:

    N(lm_head) = d_model * n_vocab
    (d_model is the lm_head input embedding size, n_vocab is vocab dict size)

  ### model
  Thus, the total model parameters size becomes:

    N(model) = n_vocab * d_model + n_blocks*(4(d_model*d_head*n_head) + 2(d_model*d_ffn)) + d_model * n_vocab

  Use llama2-7B as example, where n_vocab=32000, d_model=4096, d_model=d_head*n_head,
  d_ffn=11008, n_blocks=32. 

    N(llama2-7b) = 6738149376, almost 7b parameters.


  ## 2. Basic model memory usage
  memory usage during model training can be divided into 2 parts: fixed model parameters occupancy and dynamic
  intermediate activations occupancy. 
  ### parameters occupancy
  In training, due to Adam widely used, the memory usage is actually far more than model parameters size.

    1) use fp32 datatype
      O(parameters) = (N_parameters + N_grad + N_adamm + N_adamv) * 4 [bytes]
      (N_paramters/N_grad/N_adamm/N_adamv=N, N is the number of model parameters)

    total memory usage is 16N (bytes). With a llama2-7b, it becomes 104G

    2) use mixed datatype
      O(parameters) = N_parameters*2 + N_grad*2 + N_adamm*4 + N_adamv*4 + N_parameters*4 [bytes]
      (N_paramters/N_grad/N_adamm/N_adamv=N, N is the number of model parameters)

    total memory usage is 16N (bytes). With a llama2-7b, it becomes 104G

  ### activations occupancy
  PyTorch describe deep learning model as computation graph. Operation receive input tensor, do computation, 
  output activation tensor to downstream operations. 
  Those activation tensor will be remained, and occupy memory. 
  (Note inside an operation, some temp activations get created but released quickly, thus becomes negligible.)
  (Note we assume mixed precision mode below)

  #### Forward

    1.embedding layer is one lookup operation
        O(embedding) = (n_batch * n_seq * d_model) * 2 [bytes]
        (n_batch is batch size, n_seq is input token length, d_model is embedding size)

    2.encoder/decoder
    with each multi-head attention block, the operations include:
    2.1 QKV projection
        O(qkv_proj) = 3*(n_batch * n_seq * n_head * d_head) * 2 [bytes]
    2.2 QK cross attention
        O(qk_atten) = (n_batch * n_head * n_seq * n_seq) * 2 [bytes]
    2.3 Query
        O(v) = (n_batch * n_seq * n_head * d_head) * 2 [bytes]
    2.4 Linear project 
        O(l) = (n_batch * n_seq * d_model) * 2 [bytes]
    2.5 Mlp layer
        O(mlp) = n_batch * n_seq * d_ffn + n_batch * n_seq * d_ffn + n_batch *  n_seq * d_model 
        (refers to first linear, relu, and second linear)         
    2.5 Norm
        O(norm) = n_norms * (n_batch * n_seq * d_model) * 2 [bytes]

    (n_batch is batch size, n_seq is input token length, d_head is dimension of QKV, 
     n_head is number of heads, d_model is embedding size, n_norms is number of norm layers)

    3. lm_head
        O(lm_head) = (n_batch * n_seq * n_vocab) * 2 [bytes]

    O(forward) = (n_batch * n_seq * d_model +\
                  n_layers*(n_batch * n_seq * (4 * n_head * d_head + n_head * n_seq + 2*d_model + 2*d_ffn) +\
                    n_norms * (n_batch * n_seq * d_model)) +\
                  (n_batch * n_seq * 2*n_vocab)) * 2 [bytes]

  #### Backward   

    O(backward) = O(forward)

  Use llama2-7b as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
  n_blocks=32, n_batch=12, n_seq=64
    
    O(llama2) = O(parameters) + O(forward) + O(backword) = 104G + 3G + 3G
    (However, n_seq and n_batch now get larger and larger! for 128 batch and 512 sequence length, 
     O(forward) surges to 240G and O(llama2) surges to 580G!)



  ## 3. Lora model memory usage          
  With Lora, the trainable parameters significantly reduced, usually <0.1% of model parameters.
  ### parameters occupancy

    since few trainable parameters, the gradients/adam states parameters are negligible.

    O(parameters) = N_parameters*2 + N_grad*2 + N_adamm*4 + N_adamv*4 + N_lora*4 [bytes]
    (N_paramters=N, N_grad/N_adamm/N_adamv/N_lora=N*0.1%, N is the number of model parameters)

    total memory usage is ~2N (bytes). With a llama2-7b, it becomes 14G


  ### activations occupancy
  #### Forward

    O(forward) = (n_batch * n_seq * d_model +\
                  n_layers*(n_batch * n_seq * (4 * n_head * d_head + n_head * n_seq + 2*d_model + 2*d_ffn) +\
                    n_norms * (n_batch * n_seq * d_model)) +\
                  (n_batch * n_seq * 2*n_vocab)) * 2 [bytes]

  #### Backward

    O(forward) ~=0 as few trainable parameters.

  Use llama2-7b-lora as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
  n_blocks=32, n_batch=12, n_seq=64
    
    O(llama2_lora) = O(parameters) + O(forward) + O(backword) ~= 14G + 3G, about 17G.


  ## 4. Deepspeed+lora model memory usage
  Deepspeed is a deep learning optimization library for distributed training. For straight forward quantitative
  analyze on memory usage, we utilze zero3 + lora as example. 
  ### single gpu
  As known, deepspeed is for distributed training. However, it would also optimize the single GPU training pipeline.
  1) The deepspeed init would optimize the model, mainly cast fp32 to fp16 to reduce parameters size.
  2) deepspeed would also integrate memory fragmentations, boost memory efficiency.

  Use llama2-7b-lora as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
  n_blocks=32, n_batch=12, n_seq=64. As lora leaves few optimizer states, ignore.
   
    Raw model: 26G (fp32)
    Training MA peak: 38G

    After deepspeed init: 13G (fp16)
    Training MA peak: 21G

  ### multi gpu (one node)
  zero3 would partition model weights as well as optimizer states onto gpus, ensuring each gpu keeps only a section.
  1) partition linearly reduce memory usage for the model parameters and optimizer state, because ZeRO-3 partitions 
     those across the available GPUs.
  2) usage at other points represents consumption by input data, activations, gradients, intermediate buffers, and 
     fragmentation resulting from PyTorch computation. ZeRO3 does not affect the memory consumed by these buffers.
  3) collect partitioned model parameters and optimizer states take extra memory cost.

  Use llama2-7b-lora as example, where n_vocab=32000, d_model=4096, n_head=32, d_model=d_head*n_head, d_ffn=11008, 
  n_blocks=32, n_batch=12, n_seq=64. As lora leaves few optimizer states, ignore. On 4 GPUs.

    Raw model: 26G (fp32)
    After deepspeed init: 6G per gpu (fp16)
    Training MA peak: 26G
    (Note: above example shows 20G intermediate memory usage. compared to zero2, conclude that extra 10G is used for
     collect process)




