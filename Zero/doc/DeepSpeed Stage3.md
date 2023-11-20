# Overview
Stage 3 focus on segment model gradients, optimizer states, as well as model weights. Idealy, each process (GPU) keeps partial
of the gradients and optimizer states and model parameters, and do broadcast when necessary, to reduce the GPU memory usage. 
Also, cpu offload is supported as well.

In the model, weights will be segmented, thus each process keeps current belonging weights and delete the rest. 
This strategy ensure only current model weights will be calcuated and stored.

Since model weights get partitioned, thus in each process, its optimzier states "partitioned" natually without extra operations.


# DeepSpeedZeRoOffload
## model weights convert
    param_list = module.parameters(recurse=True)  
    def _convert_to_zero_parameters(param_list):
        for param in param_list:
          // copy weights to local rank device in group. usually device 0
          param.data = param.data.to(local_rank)
          // broadcast to process groupm thus all ranks in group have it.
            dist.broadcast(param, 0, self.get_dp_process_group())

          // flatten the param tensor to 1d tensor
          one_dim_param = param.contiguous().view(-1)

          // copy partitioned model weights to current rank
          tensor_size = self._aligned_size(param)
          partition_size = tensor_size // self.num_partitions
          start = partition_size * get_partition_rank()
          end = start + partition_size
          src_tensor = one_dim_param.narrow(0, start, partition_size)
          param.ds_tensor.copy_(src_tensor)

## FWD/BWD
    def setup_zero_stage3_hooks():
        // register hooks for restore/release weights
