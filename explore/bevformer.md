# BevFormer


## BEV map 
<img src="https://github.com/user-attachments/assets/2ba210b5-39d9-43d2-83eb-eb15a70825c1" width="400" height="600"> 

```
bev_queries = Parameter((H_bev * W_bev, embed_dim))
bev_coords[H_bev*W_bev, 3] #xyz grid from range(x_min, x_max, y_min, y_max, z=0)

# Project BEV coordinates to image plane
for cam_idx in range(self.num_cameras):
  projected_img_coords = project_bev_to_image(bev_coords, camera_poses)
  reference_points.append(img_coords)
reference_points = stack(reference_points)  #[B, N_cam, N_bev, 2], N_bev=H_bev*W_bev

# Bilinear sampling image features at reference points
for cam_idx in range(self.num_cameras):
    sampled_feat = F.grid_sample(image_features, reference_points)
    sampled_features.append(sampled_feat)
sampled_features = stack(sampled_features).flatten() #[B, N_bev*N_cam, C]

# cross-attn
bev_features = cross_attention(
    query=bev_queries,
    key=sampled_features,
    value=sampled_features
)

# Self-attn within BEV space for spatial reasoning
bev_features = self_attention(
    query=bev_features,
    key=bev_features,
    value=bev_features
)
```


## Model
<img src="https://github.com/user-attachments/assets/c0d2b4f0-50e7-4c97-8871-570f7e15ccdf" width="400" height="600">


