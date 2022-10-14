import torch.nn as nn
from models import AttentionNetwork, VideoClassifier
import torch
import copy
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchvision.transforms import Resize

class FrameClassifier(nn.Module):
    """
    Output logit scores for pda classification, mode, and view. 
    This network produces a common embedding and outputs logit scores for 
    pda (called 'type' in code), mode, and view. 
    """
    def __init__(self, encoder, encoder_frozen=True):
        super(FrameClassifier, self).__init__()
        
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
        """
        Takes target classes and network outputs and computes the loss. 
        Loss is computed as the weighted sum of pda, mode, and view classification cross entropy losses. 
        Additionaly, samples with view '2d' or mode 'nonPDAView' have pda loss set to zero. 
        That is because these views/modes do not adequately visualize the PDA. 
        """
        # type loss:
        ltype = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['type'], 
            targets['trg_type'][...,None].type(torch.float32),
            reduction = 'none'
        )
        # zero out views/modes that do not visualize pda
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
        
        # compute the total 
        total_loss = \
            weights['type'] * ltype +\
            weights['mode'] * lmode +\
            weights['view'] * lview

        loss_dict = {
            'total': total_loss.mean(axis=0),
            'type': ltype.mean(axis=0),
            'mode': lmode.mean(axis=0),
            'view': lview.mean(axis=0)
        }

        return loss_dict
    
    
class VideoClassifier_PI_PI(nn.Module):
    """
    This version uses permutation invariant attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifier_PI_PI, self).__init__()
        
        self.clf_type = copy.deepcopy(frame_classifier.clf_type)
        self.clf_view = copy.deepcopy(frame_classifier.clf_view)
        self.clf_mode = copy.deepcopy(frame_classifier.clf_mode)
        self.encoder = frame_classifier.encoder

        # copy linear layer to match dimensions
        self.attn_net_type = AttentionNetwork()
        self.attn_net_view = AttentionNetwork()
        self.attn_net_mode = AttentionNetwork()
        
        # free weights
        if encoder_frozen:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
                
        if frame_classifier_frozen:
            for p in self.fc_pda.parameters():
                p.requires_grad = False
        
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad(self.encoder(x), num_frames)
        #p_frame = torch.sigmoid(self.fc_pda(h))
        y_type = self.clf_type(h)
        y_view = self.clf_view(h)
        y_mode = self.clf_mode(h)
        
        # attention
        alpha_type = self.attn_net_type(x)
        alpha_view = self.attn_net_view(x)
        alpha_mode = self.attn_net_mode(x)
        alpha_type, alpha_view, alpha_mode = [self.pad(alpha, num_frames) for alpha in (alpha_type, alpha_view, alpha_mode)]
        
        for alpha in (alpha_type, alpha_view, alpha_mode):
            for ix, n in enumerate(num_frames):
                alpha[n:, ix]=-40
        
        attn_type, attn_view, attn_mode = [torch.softmax(alpha, axis=0) for alpha in (alpha_type, alpha_view, alpha_mode)]

        Y_type = torch.sum(y_type * attn_type, axis=0) 
        Y_view = torch.sum(y_view * attn_view, axis=0) 
        Y_mode = torch.sum(y_mode * attn_mode, axis=0) 
        return {'type': Y_type, 'mode': Y_mode, 'view': Y_view}, {'type': attn_type, 'mode': attn_mode, 'view': attn_view}
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.clf_type
        )
    
    @staticmethod
    def pad(h, num_frames):
        # reshape as batch of vids and pad
        start_ix = 0
        h_ls = []
        for n in num_frames:
            h_ls.append(h[start_ix:(start_ix+n)])
            start_ix += n
        h = pad_sequence(h_ls)
        
        return h