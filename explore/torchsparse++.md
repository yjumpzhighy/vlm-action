# TorchSparse++: Efficient Point Cloud Engine

![image](https://github.com/user-attachments/assets/a827fe83-9515-4666-a09f-357266dd8e35)

## gather-gemm-scatter strategy:
<img src="https://github.com/user-attachments/assets/81ab120c-01cb-4f48-b95b-2b0346c67982" width="300" />  <img src="https://github.com/user-attachments/assets/d7f20316-d341-4ba6-8389-907862db137a" width="300" />
The pipeline would be proceed as:
1) gather input features based on input-output map for each cell weight
2) perform GEMM between input features and kernel weights
3) scatter the results back to output locations based on input-output map
4) repeat 1~3 for kernel volume times

The v2 pipeline would be proceed as:
1) call gather input features based on input-output map for each cell weight, for kernel volume times
2) perform bmm
3) call scatter the results back to output locations based on input-output map, for kernel volume times


## implicit gemm
<img src="https://github.com/user-attachments/assets/c64d28ff-40a6-4622-a271-a149205056f5" width="300" />
