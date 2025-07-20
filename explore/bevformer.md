# BevFormer


## BEV feats
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
flatten_features = stack(sampled_features).flatten() #[B, N_bev*N_cam, C]

# cross-attn
bev_features = cross_attention(
    query=bev_queries,
    key=flatten_features,
    value=flatten_features
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

1. images feed to resnet50-backbone and ffn-neck, get 5 multi-lvl feats [[B,N_cam,H/8,W/8],..., [B,N_cam,H/128,W/128]]
2. bev features
```
bev_quiries = Parameter((H_bev*W_bev,256)
object_query_embedding = Parameter((900,256*2))

bev_positional_embedding = nn.Embedding(..)((B,H_bev,W_bev)) #[B,256,H_bev,W_bev], learned positional embedding
bev_quiries = bev_quiries.expand(..) #[H_bev*W_bev,B,256]
bev_positional_embedding = bev_positional_embedding.flatten(..) #[H_bev*W_bev,B,256]


```
