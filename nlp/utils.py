import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def pad_zero(seqs, max_len, pad_id):
    padded = np.full((len(seqs), max_len), fill_value=pad_id, dtype=np.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded

def set_soft_gpu(soft_gpu):
    import tensorflow as tf
    if soft_gpu:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            
class AvgMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg, self.sum, self.count = [0] * 3
    def update(self,val,count=1):
        self.count += count
        self.sum += val*count
        self.avg = self.sum / self.count
    def __repr__(self) -> str:
        text = f"Metrics:{self.avg:.5f}"
        return text
    
class BeamHypotheses(object):
    def __init__(self, num_hyp=1, max_length=1024, 
                 length_penalty=0.6, early_stopping=False):
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_hyp = num_hyp
        self.hyp = []
        self.worst_score = 1e9
    def _length_norm(self,length):
        # return length ** self.length_penalty
        return (5+length) ** self.length_penalty / (5+1) ** self.length_penalty
    def add(self, hyp, sum_logprobs):
        # add new hypothesis to list
        score = sum_logprobs / self._length_norm(len(self.hyp))
        if (len(self.hyp) < self.num_hyp or score > self.worst_score):
            self.hyp.append((score, hyp))
            if len(self.hyp) > self.num_hyp:
                # delete hyp with lowest score
                sorted_scores = sorted([(s,idx) for idx, (s,_) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    def is_done(self, best_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if (len(self.hyp) < self.num_hyp):
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_logprobs / self.max_length ** self.length_penalty
        
     
class GeneratorWithBeamSearch(nn.Module):
    def __init__(self, eos_index, max_length=1024, beam_size=4, 
                 length_penalty=0.6, per_node_beam_size=2,
                 repetition_penalty=1.0, temperature=1.0):
        super().__init__()
        self.eos_index = eos_index
        self.max_length = max_length
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
    
    def search(self, decode_step, input_ids, num_keep_best=1,
               do_sample=False, top_k=None, top_p=None,
               num_return_sequences=1):
        input_ids = input_ids.long()
        # [batch, cur_len], ie. input_ids=[[101]]
        batch_size, token_len = input_ids.shape 
        # [batch, beam, cur_len]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, self.beam_size, token_len)
        # [batch*beam, cur_len]
        input_ids = input_ids.contiguous().view(batch_size*self.beam_size, token_len)

        generated_hyps = [BeamHypotheses(num_keep_best,self.max_length,self.length_penalty,
                                         early_stopping=False) for _ in range(batch_size)]
        
        beam_scores = torch.zeros((batch_size, self.beam_size), dtype=torch.float, 
                                 device=input_ids.device)
        beam_scores[:,1:] = -1e9
        beam_scores = beam_scores.view(-1) #[batch*beam,]
        
        done = [False for _ in range(batch_size)]
        cur_token_len = token_len
        
        # decode token by token. each time process one token in each batch
        while (cur_token_len < self.max_length):
            # [batch*beam, vocab_size]
            # 取出最后一个时间步的各token概率，即当前条件概率
            scores = decode_step(input_ids)[:,-1,:]
            vocab_size = scores.shape[-1]
            
            # [batch*beam, vocab_size]
            scores = F.log_softmax(scores, dim=-1)
            _scores = scores + beam_scores[:,None].expand_as(scores)
            # [batch, beam*vocab_size]
            _scores = _scores.view(batch_size, self.beam_size * vocab_size)
            # get first largest k elements from each batch's beam*vocab_size elments, 
            # get [batch, 2*beam]
            next_scores, next_idx = torch.topk(_scores, 
                                                 2 * self.beam_size,
                                                 dim=1,
                                                 largest=True,
                                                 sorted=True)

            # a list of tuple(score,word_id,beam_id)
            next_batch_beam = []
            for batch_ex in range(batch_size):
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(
                    next_scores[batch_ex].max().item())
                
                if done[batch_ex]:
                    # just pad for next batch
                    next_batch_beam.extend([(0,self.eos_index,0)] * self.beam_size)
                    continue
                
                # next sentence beam content
                next_sentence_beam = []
                for idx, score in zip(next_idx[batch_ex], next_scores[batch_ex]):
                    # 因为是所有beam混合一起取topk, 可能出现一个beam里取出多个的情况
                    # beam_id是一样的，但score和word_id肯定不一样
                    beam_id = idx / vocab_size
                    word_id = idx % vocab_size
                    
                    idx_in_inputids = int(batch_ex * self.beam_size + beam_id)
                    # end of sentence? or next word?/mail/u/1/
                    if word_id == self.eos_index or cur_token_len+1 == self.max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[idx_in_inputids, :cur_token_len].clone(),
                            score.item()
                        )
                    else:
                        next_sentence_beam.append((score,word_id,idx_in_inputids))
                                                     
                    # the beam for next step is full
                    if len(next_sentence_beam) == self.beam_size:
                        break                                 
                    
                if cur_token_len+1 == self.max_length:
                    assert len(next_sentence_beam) == 0
                else:
                    assert len(next_sentence_beam) == self.beam_size
                    
                if len(next_sentence_beam) == 0:
                    next_sentence_beam = [(0, self.eos_index, 0)] * self.beam_size
                    
                next_batch_beam.extend(next_sentence_beam)
                
            
            assert len(next_batch_beam) == batch_size * self.beam_size
            
            # 把三元组列表再还原成三个独立列表 
            # [batch_size * beam_size,]
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            
            # generate captions for next step, ie, cur caption + next word
            # 当前input_ids, [batch_size * beam_size, cur_len]
            input_ids = input_ids[beam_idx, :]
            # 下一步input_ids, [batch_size * beam_size, cur_len + 1]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)
            
            cur_token_len += 1
            
            if all(done):
                break
            
        # select the best hypothesis and generate answer
        tgt_len = torch.ones(batch_size, num_keep_best, dtype=torch.long)
        logprobs = torch.zeros(batch_size, num_keep_best, dtype=torch.float).fill_(
                        -1e5).to(input_ids.device)
        all_best = []
        
        for batch_i, hypothese in enumerate(generated_hyps):
            best = []
            # [num_keep_best, ]
            hyp_scores = torch.tensor([x[0] for x in hypothese.hyp])
            
            _, best_indices = torch.topk(hyp_scores,
                                         min(num_keep_best, len(hyp_scores)),
                                         largest=True)
            for best_i, hpy_i in enumerate(best_indices):
                score, best_hyp = hypothese.hyp[hpy_i]
                best.append(best_hyp)
                logprobs[batch_i, best_i] = score
                
                tgt_len[batch_i, best_i] = len(best_hyp)+1 #+1 for <eos>
                
            all_best.append(best)
            
        # generate answer
        decoded = input_ids.new(batch_size,num_keep_best,self.max_length).fill_(self.eos_index)
        for batch_i, best in enumerate(all_best):
            for best_i, hypo in enumerate(best):
                decoded[batch_i,best_i,:tgt_len[batch_i,best_i]-1] = hypo
                decoded[batch_i,best_i,tgt_len[batch_i,best_i]-1] = self.eos_index
        if num_keep_best==1:
            decoded = decoded.squeeze(dim=1)
        return decoded, _  
    