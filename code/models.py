import torch.nn as nn
import torch
from utils import pad
import math
   
class MedVidNet(nn.Module):
    def __init__(self, encoder, num_heads, num_out=1, pooling_method='attn', drop_rate=0.0, debug=False):
        super(MedVidNet, self).__init__()
        self.num_features = encoder.num_features
        assert self.num_features % num_heads == 0, "The number of encoder features must be divisble by the number of attention heads."
        self.num_heads = num_heads
        self.subspace_size = self.num_features // num_heads
        self._scale = math.sqrt(self.subspace_size)
        self.num_out = num_out
        self.drop_rate = drop_rate
        self.debug=debug
        self.pool=pooling_method
        
        # frame encoder
        self.encoder = encoder
        
        # dot-prod attention of frame embeddings with global query vectors
        self.attn_query_vecs = nn.Parameter(torch.randn(self.num_heads, self.subspace_size))
        
        #module to compute network output from pooled projections
        self.fc_out = nn.Sequential(
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.num_features, self.num_out)
        )
        
        if self.pool == 'attn':
            self.pool_func = self.attention_pool
        elif self.pool == 'max':
            self.pool_func = self.max_pool
        elif self.pool == 'avg':
            self.pool_func = self.avg_pool
        else:
            raise NotImplementedError(f"{self.pool} pooling method has not been implemented. Use one of 'attn', 'max', or 'avg'")
    
    def forward(self, x, num_frames):
        # frame representations
        h = self.encoder(x)
        print("h shape:", h.shape) if self.debug else None
        # expect [L*N, h_dim]
        
        h_vid, attn = self.pool_func(h, num_frames)
        # expect [N, num heads, subspace_size], [L, N, num_heads]
        
        # compute the output
        output = self.fc_out(h_vid)
        print("output shape:", output.shape) if self.debug else None
        # expect [N, output size]
        
        return output, attn
    
    def attention_pool(self, h, num_frames):
        # attention logits
        h_query = h.view(-1, self.num_heads, self.subspace_size)
        print("h_query shape:", h_query.shape) if self.debug else None
        # expect [L*N, num_heads, subspace_size]
        print("query vector shape:", self.attn_query_vecs.shape) if self.debug else None
        # expect [num_heads, subspace_size]
        alpha = (h_query * self.attn_query_vecs).sum(axis=-1)
        print("alpha shape:", alpha.shape) if self.debug else None
        # expect [L*N, num_heads]
        
        # normalized attention
        alpha = pad(alpha, num_frames)
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-50
        attn = torch.softmax(alpha, axis=0)
        print("attn shape:", attn.shape) if self.debug else None
        # expect [L, N, num_heads]
        
        # pool within subspaces
        h_query_pad = pad(h_query, num_frames)
        print("h_query_pad shape:", h_query_pad.shape) if self.debug else None
        # expect [L, N, num heads, subspace_size]
        h_vid = torch.sum(h_query_pad * attn[...,None] / self._scale, axis=0)
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, num heads, subspace_size]
        
        h_vid_wide = h_vid.view(-1, self.num_features)
        print("h_vid_wide shape:", h_vid_wide.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_vid_wide, attn
    
    def max_pool(self, h, num_frames):
        h_pad = pad(h, num_frames)
        print("h_pad shape:", h_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take max
        h_vid = h_pad.max(0).values
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_vid, None
    
    def avg_pool(self, h, num_frames):
        h_pad = pad(h, num_frames)
        print("h_pad shape:", h_pad.shape) if self.debug else None
        # expect [L, N, h_dim]
        
        # take avg
        h_vid = h_pad.mean(0)
        print("h_vid shape:", h_vid.shape) if self.debug else None
        # expect [N, h_dim]
        
        return h_vid, None