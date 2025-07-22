# BevFormer


## BEV feats
<img src="https://github.com/user-attachments/assets/20c4571c-cc04-4c56-9043-aea0d2db3d3f" width="400" height="600"> 


## Encoder
<img src="https://github.com/user-attachments/assets/c0d2b4f0-50e7-4c97-8871-570f7e15ccdf" width="400" height="600">

1. images feed to resnet50-backbone and ffn-neck, get 5 lvls mlvl_feats [[B,N_cam,H/8,W/8],..., [B,N_cam,H/128,W/128]]       
```
lvl_embeds = Parameter((5, 256)) #5 levels feats
# add lvl embedding and concat all features
for lvl, feat in mlvl_feats:
  feat += lvl_embeds[lvl]
mlvl_feats.cat()  #[N_cam,M,B,256], M=H/8*W/8+..+H/128*W/128
```
2. query setup  
```
cams_embeds = Parameter((N_cam, 256)) #each camera individual embedding
bev_quiries = Parameter((H_bev*W_bev,256)

bev_positional_embedding = nn.Embedding(..)((B,H_bev,W_bev)) #[B,256,H_bev,W_bev], learned positional embedding
bev_positional_embedding = bev_positional_embedding.flatten(..) #[H_bev*W_bev,B,256]
```

3. TSA
```
#1. get 2d bev plane reference location
ref_2d = stack(linspace(0.5,H_bev-0.5,H_bev)/H_bev,
               linspace(0.5,W_bev-0.5,W_bev)/W_bev).expand() #[B,H_bev*W_bev,1,2] and 2 refers uv

#2. get historical bev features
bev_quiries = bev_quiries.expand(..) #[B,H_bev*W_bev,256]
bev_pos = bev_positional_embedding.flatten(..) #[B,H_bev*W_bev,256]
bev_quiries += bev_pos  #[B,H_bev*W_bev,256]

if prev_bev is not None:  
    prev_bev = stack(prev_bev, bev_quiries) #[2B,H_bev*W_bev,256]
else: #the first frame
    prev_bev = stack(bev_quiries,bev_quiries) #[2B,H_bev*W_bev,256]

k = prev_bev
v = proj(prev_bev).view(..)  #[B*Bev_queue,H_bev*W_bev,Heads,-1]
q = cat(v[:B], bev_quiries) #[B,H_bev*W_bev,512]  #prev bev_feats and current bev_query

#3.Predict sampling offsets directly from current BEV query.
#  and no ego motion input needed, the network learns to predict where to sample!
#Num_lvls: feature maps used in attention
#Num_pts: number of sampling points for each query in each head
#Bev_queue: 2 refers to one history bev and one current bev
sampling_offsets = Linear(512,128)(q) #[B,H_bev*W_bev,128]
sampling_offsets = sampling_offsets.view(B*Bev_queue,H_bev*W_bev,Heads,Num_lvls,Num_pts,2)

#4. Predict attention weights from query
attention_weights = Linear(512, 64)(q) #[B,H_bev*W_bev,64]
attention_weights = attention_weights.view(B,H_bev*W_bev,Heads,Bev_queue,Num_lvls*Num_pts)
attention_weights = softmax(attention_weights).view(B*Bev_queue,H_bev*W_bev,Heads,Num_lvls,Num_pts)

#5. Compute sampling locations
sampling_loc = ref_2d + sampling_offsets / [H_bev, W_bev]  #[B*Bev_queue,H_bev*W_bev,Heads,Num_lvls,Num_pts,2]

#6. Sample 2d features from v
query = MultiScaleDeformAttention(v, sampling_loc)   #[B*Bev_queue,H_bev*W_bev,256]
query = query.permute().view(..).mean()   #[B,H_bev*W_bev,256]
```
  

4. SCA
```
#1. get 3d ref xyz in bev voxel
xs = linspace(0.5,W_bev-0.5,W_bev).expand(N_pts_per_pillar,H_bev,W_bev) / W_bev
ys = linspace(0.5,H_bev-0.5,H_bev).expand(N_pts_per_pillar,H_bev,W_bev) / H_bev
Z = pc_zmax - pc_zmin
zs = linspace(0.5,Z-0.5,Z).expand(N_pts_in_pillar,H_bev,W_bev) / Z
ref_3d = stack(xs,ys,zs).expand()  #[B,N_pts_per_pillar,H_bev*W_bev,3], indicates xyz of each voxel

#2. apply camera pos (extrinsics & intrinsics) and project to 2d image coord
lidar2img = tensor(..) #[B,N_cam,4,4]
lidar2img = lidar2img.repeat(..) #[N_pts_per_pillar,B,N_cam,H_bev*W_bev,4,4]
ref_3d = ref_3d.unsqueeze(-1) #[N_pts_per_pillar,B,N_cam,H_bev*W_bev,4,1]
ref_points_cam = matmul(lidar2img, ref_3d) #[N_pts_per_pillar,B,N_cam,H_bev*W_bev,4]
ref_points_cam = (ref_points_cam[...,0:2] / ref_points_cam[...,2]).view()  #[N_cam,B,H_bev*W_bev,N_pts_per_pillar,2], uv coord

#3.Predict sampling offsets directly from current BEV query.
#  and no ego motion input needed, the network learns to predict where to sample!
sampling_offsets = Linear()(query).view() #(B,H_bev*W_bev,Heads,Num_lvls,Num_pts,2)
attention_weights = Linear()(query).view() #(B,H_bev*W_bev,Heads,Num_lvls*Num_pts)
attention_weights = softmax(attention_weights).view() #B,H_bev*W_bev,Heads,Num_lvls,Num_pts)

#4. Compute sampling locations
sampling_loc = ref_points_cam + sampling_offsets / [H_bev, W_bev]  #[B,H_bev*W_bev,Heads,Num_lvls,Num_pts,2]


#5. sample bev 3d features from image features
k = v = mlvl_feats  #[N_cam,M,B,256]
query = MultiScaleDeformAttention(v, sampling_loc)   #[B*Bev_queue,H_bev*W_bev,256]
query = query.permute().view(..).mean()   #[B,H_bev*W_bev,256]
```


## Decoder
bev_feats: [B, H_bev*W_bev, 256]              
object_query_embedding = Parameter((900,256*2))                   
query_pos, query = split(object_query_embedding).expand  #[B,900,256], [B,900,256]                   
refence_points = sigmoid(Linear(256,3)(query_pos))[:2].unsqueeze() #[B,900,1,2]                            

```
v = bev_feats

#1. standard mha self attn
query = MHA(q=query,k=query,v=query,query_pos=query_pos,key_pos=query_pos)

#2. Predict sampling offsets directly from current BEV query.
#   and no ego motion input needed, the network learns to predict where to sample!
query += query_pos
sampling_offsets = Linear()(query).view() #(B,900,Heads,Num_lvls,Num_pts,2)
attention_weights = Linear()(query).view() #(B,900,Heads,Num_lvls*Num_pts)
attention_weights = softmax(attention_weights).view() #B,900,Heads,Num_lvls,Num_pts)

#3. Compute sampling locations
sampling_loc = refence_points + sampling_offsets / [H_bev, W_bev]  #[B,H_bev*W_bev,Heads,Num_lvls,Num_pts,2]

#4. get object query from bev features
query = MultiScaleDeformAttention(v, sampling_loc)  #[B,900,256]

```

## Head
stack query from 6 decoder blocks and get [6,B,900,256]                               
then linear project to cls and reg on each lvl














