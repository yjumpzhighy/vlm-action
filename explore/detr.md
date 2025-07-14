# detection transformer

<img src="https://github.com/user-attachments/assets/9a815499-be1b-41ef-b9b9-4acd96274228" width="400" height="600">   


# model  
1. (b,h,w,3)用resnet backbone (fronzen batchnorm) 预训练模型，得到src (b,h/32,w/32,256)            
   除padding外的位置都为1得到 mask(b,h,w), 通过F.interpolate得到 mask(b,h/32,w/32)           
2. Position encoding               
   a.对图片mask(b,h/32,w/32), 在x,y 累加，得到x_embed, y_embed (b,h/32)
   
      <img src="https://github.com/user-attachments/assets/99ccd76a-0c79-49e8-a5e7-7c6cfa87a3c9" width="200" height="300">  

   b. 对x, 第i维(0~127)，偶数sin(x_embed/10000^(2i/128))，奇数cos(x_embed/10000^(2i/128)),    
      叠加后有pos_x (b,h/32,w/32,128)
      
   c. 对y, 第i维(0~127)，偶数sin(y_embed/10000^(2i/128))，奇数cos(y_embed/10000^(2i/128)),             
      叠加后有pos_y (b,h/32,w/32,128)
   
      <img src="https://github.com/user-attachments/assets/186ac7dc-8be6-4c30-8bcf-a012304d80bb" width="200" height="300">    

   d. 最后在叠加到一起pos, (b,h/32,w/32,256)               

4. encoder            
   H=h/32, W=w/32               
   src(b,H,W,256)->(b, H*W, 256), pos(b,H,W,256)->(b, H*W, 256), mask(b,H,W)->(b, H*W)            
   a. q=k=src+pos,. 但v=src，不计算pos。 （因为q和k用来计算特征图中各个位置的相关性，
      加上位置编码能加强全局相关性. v只代表原图不需要加位置编码）
   
   b. 多头(8)注意力机制，直接使用nn.MultiheadAttention，最终结果和输入保持一致，
      仍然是src(b,H*W,256)
   
   c. 再将输入src和注意力机制后的src相加(类似residual), 再进入全连接网络层，最终
             输入仍然保持(b,H*W,256)

   d. 6个编码器重复上面步骤，终输src仍是(b,H*W,256). 每次的src就是上一个
             编码器的特征输出，但pos不变

6. decoder                
   a. encoder的输出特征图(b,H*W,256) + pos(b, H*W, 256)                  
      object queries 是预查询编码，最开始随机初始化为(b, 100, 256), 后续通过学习进行更                    
      新, 类似于pre-defined anchors.                               
   b. 第一个self-attention, 和encoder无关, 建模物体和物体之间的关系,找到图像中哪些位置存在物体                              
      q = k = object_queries, v = object_queries. 输出仍为object_queries (b, 100, 256)              
   c. 第二个cross-attention, 其中k和v来自encoder, q来自decoder自身, 建模图形特征和物体特征之间     
      的关系. 即根据object_query不断去从encoder的输出特征中询问(q,k计算相似度) 图像中的物体特征.               
      q = object_queries, k = src + pos, val = src 输出仍为object_queries(b, 100, 256)               
   d. 将两次attention得到的object_queries相加作为最终object_queries. 因为第一次代表物体信息缺乏图               
      像特征(class不准),第二次代表图像特征但却物体信息不足(bbox不准)                    
   e. 6个编码器重复上面步骤, 但与encoder不同6个输出都需要记录, 最终得到(6, b, 100, 256)         
8. loss         
   a. 预测              
      (6, b, 100, 256) 通过MLP投射为(6, b, 100, num_classes)               
      (6, b, 100, 256) 通过MLP投射为(6, b, 100, 4) 并sigmoid为(0,1)               
      注: 分别为centerx, centery, w, l 想对于原图的比例               
   b. HM匹配                 
      从100个预测框和N个gt框中找到一一对应,剩余都变为背景                 
      比如有3个预测框匹配成功,则剩余的97个预测框都是背景              
   c. 计算损失             
      class cross entropy. 上面的情况即正样本(3)+负样本(97)=100, 正样本正常计算, 负样本则强              
      制为背景类             
      bbox L1+GIOU,只计算正样本
   ```
   # pred_batch: [100, classes + 4]
   # gt_batch: [N_gt, classes + 4]
   
   matched_indices = hungarian_matching(pred_batch, gt_batch)
   # matched_indices: [(query_idx, gt_idx), (query_idx, gt_idx), ...]

   for query_idx, gt_idx in matched_indices:
      # Classification loss for matched pairs
      pred_class = pred_batch[query_idx][:num_classes]
      gt_class = gt_batch[gt_idx][:num_classes]
      class_loss += cross_entropy_loss(pred_class, gt_class)
      
      # Bbox loss for matched pairs (only if not background)
      if gt_class != background_class:
          pred_bbox = pred_batch[query_idx][num_classes:]
          gt_bbox = gt_batch[gt_idx][num_classes:]
          bbox_loss += l1_loss(pred_bbox, gt_bbox) + giou_loss(pred_bbox, gt_bbox)
        
   # Unmatched queries → background class
   unmatched_queries = set(range(num_queries)) - set([idx[0] for idx in matched_indices])
   for query_idx in unmatched_queries:
      pred_class = pred_batch[query_idx][:num_classes]
      class_loss += cross_entropy_loss(pred_class, background_class)
  
  total_loss += class_loss + bbox_loss
  ```       

# conclude
1.encoder的每个输入token代表特征图的一个像素点,encoder遍历图中剩余的所有像素点（包括它自己），来学习自己应该特别关注哪些像素点,目的是掌握全局信息. 
  那么decoder的object query同样可以理解为特征图中的某一个像素点，只不过它是随机的,类似于检测框的质心，在训练的过程中，这些随机点不断去学习自己
  应该关注的部分，然后调整自己的位置。                
2.each object query doesn't represent a specific object type or instance. Instead, they function more like "detection slots" 
  or "attention patterns" rather than object identities, able detect ANY object through the cross-attention mechanism.
