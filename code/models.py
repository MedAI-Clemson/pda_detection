import torch.nn as nn
import torch
import copy


class VideoClassifier(nn.Module):
    """
    Video classifier using attention over frames. Two things are computed for each frame:
    1) the probability that the frame belongs to a PDA clip
    2) the log attention for the frame which can be interpreted as the relevance of the frame
    for pda classification
    The latter is normalized 
    
    """
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
    
class VideoClassifierZ(nn.Module):
    """
    Average the instance embeddings instead of probabilities as in 
    https://arxiv.org/abs/1802.04712
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifierZ, self).__init__()
        
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
        
        # attention logit for all frames
        alpha = self.fc_attention(h)
        
        # setting logit attn to -40 for masked frames allows us to avoid python for loops
        # alpha = torch.ones_like(alpha)  # use this for sanity check: this should be identical to just averaging frames
        # the resulting masked attention will be very close to zero for frames not in the clip under consideration
        alpha = torch.where(mask, alpha, -40)
        attn = torch.softmax(alpha, axis=0)
        
        # compute video probability as the sum of the attention-weighted probabilities 
        
        z = torch.sum(h[...,None]*attn[:,None], axis=0).T
        
        p_vid = torch.sigmoid(self.fc_pda(z))
        
        return p_vid, attn
    
class MultiTaskFrameClassifier(nn.Module):
    def __init__(self, encoder, encoder_frozen=True):
        super(MultiTaskFrameClassifier, self).__init__()
        
        # hidden dim size
        h_dim = encoder.fc.in_features
        
        # encoder
        self.encoder = encoder
        self.encoder.fc = nn.Identity()
        
        # classification heads
        self.clf_type = nn.Linear(h_dim, 1)
        self.clf_mode = nn.Linear(h_dim, 3)
        self.clf_view = nn.Linear(h_dim, 3)
        
        # freeze the encoder
        if encoder_frozen:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        h = self.encoder(x)
        
        pred_dict = dict(
            type = self.clf_type(h),
            mode = self.clf_mode(h),
            view = self.clf_view(h)
        )
        
        return pred_dict
    
    @staticmethod
    def multi_task_loss(outputs, targets, weights):
        # type loss:
        # zero out the "type" loss if view is "nonPDAView" or mode is "2d"
        # because these do not show relevant structure
        # consult dataset.py for relevant codes
        ltype = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['type'], 
            targets['trg_type'][...,None].type(torch.float32),
            reduction = 'none'
        )
        type_filter = (targets['trg_view']==0) | (targets['trg_mode']==0)
        ltype = torch.where(type_filter, 0, ltype.squeeze())

        # mode loss
        lmode = torch.nn.functional.cross_entropy(
            outputs['mode'], 
            targets['trg_mode'].type(torch.long), 
            reduction = 'none'
        )

        # view loss
        lview = torch.nn.functional.cross_entropy(
            outputs['view'], 
            targets['trg_view'].type(torch.long), 
            reduction = 'none'
        )

        total_loss = \
            weights['type'] * ltype +\
            weights['mode'] * lmode +\
            weights['view'] * lview

        loss_dict = {
            'total': total_loss.mean(axis=0),
            'type': ltype.mean(axis=0),
            'type_filtered': ltype[~type_filter].mean(axis=0),
            'mode': lmode.mean(axis=0),
            'view': lview.mean(axis=0)
        }

        return loss_dict