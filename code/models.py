import torch.nn as nn
import torch
import copy

class VideoClassifier(nn.Module):
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifier, self).__init__()
        
        self.fc_pda = copy.deepcopy(frame_classifier.fc)
        # copy linear layer to match dimensions
        self.fc_attention = copy.deepcopy(frame_classifier.fc)
        
        # make the new classifier "at least as good" as averaging over frame scores
        # by initializing to uniform attention
        self.fc_attention.weight = nn.Parameter(torch.zeros_like(self.fc_attention.weight))
        self.fc_attention.bias = nn.Parameter(torch.ones_like(self.fc_attention.bias))
    
        self.encoder = frame_classifier
        self.encoder.fc = nn.Identity()
        
        if encoder_frozen:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
                
        if frame_classifier_frozen:
            for p in self.fc_pda.parameters():
                p.requires_grad = False
        
    def forward(self, x, mask):
        h = self.encoder(x)
        
        # the frame-level probability of PDA
        p_frame = torch.sigmoid(self.fc_pda(h))
        
        # attention logit for all frames
        alpha = self.fc_attention(h)
        
        # setting logit attn to -40 for masked frames allows us to avoid python for loops
        # alpha = torch.ones_like(alpha)  # use this for sanity check: this should be identical to just averaging frames
        # the resulting masked attention will be very close to zero for frames not in the clip under consideration
        alpha = torch.where(mask, alpha, -40)
        attn = torch.softmax(alpha, axis=0)
        
        # compute video probability as the sum of the attention-weighted probabilities 
        
        p_vid = torch.sum(p_frame.expand(-1,mask.shape[-1]) * attn, axis=0)
        
        return p_vid, attn