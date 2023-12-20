# Overview
Stage 1,2 focus on segment model gradients and optimizer states. Idealy, each process (GPU) keeps partial of the
gradients and optimizer states, and do broadcast when necessary, to reduce the GPU memory usage.
Also, cpu offload is supported as well.

In the optimizer, params_group will be segmented, thus each process keeps current belonging parameters and delete the
rest. This strategy ensure only current part of gradient will be calcuated and stored, same with the optimzier states.

# DeepSpeedZeroOptimizer
in each optimizer.param_groups, perform round_robin distribute. for examaple, to partition into n parts:
usually n is the world_size (usually gpus) in the group

## get all trainable param in optimizer group
      tensor_list_in_group = []
      for param in optimizer.param_groups[0]['params']:
            if param.requires_grad:
               tensor_list_in_group.append(param)

## round robin to reorder weights tensors with ranks
      def _round_robin_reorder(tensor_list_in_group, num_partitions): 
        // Put all param tensors into num_partitions bucket
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

## flatten weights tensors into 1d tensor
      def flatten_dense_tensors_aligned(reordered_tensors):
        // flatten all tensors into one 1d tensor 
        group_flat = torch.flatten_dense_tensor(reordered_tensors)

## partition the flatten tensor to parts
      def get_data_parallel_partitions(group_flat, num_partitions):
        // divide the flat weights into near num_partitions equally
        // each process will compute on a different part of the partition 
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
        // Note partitioned equally, means a tensor maybe divided to
        // two parts. 

## copy each partition into device.
      partition_id = dist.get_rank()
      optimizer.param_groups[0]['params'] = [partitions[partition_id].to(device).clone().float().detach()]
      optimizer.param_groups[0]['params'].requires_grad = True
