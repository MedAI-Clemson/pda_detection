import torch.nn as nn
import torch
import copy
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class VideoClassifier(nn.Module):
    """
    This model gives each frame equal weight. This is used as base class for the other methods. 
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifier, self).__init__()
      
        self.fc_pda = copy.deepcopy(frame_classifier.fc)
        self.encoder = frame_classifier
        self.encoder.fc = nn.Identity()
        
        if encoder_frozen:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
                
        if frame_classifier_frozen:
            for p in self.fc_pda.parameters():
                p.requires_grad = False
        
    def forward(self, x, num_frames):
        h = self.pad_encodings(self.encoder(x), num_frames)
        p_frame = torch.sigmoid(self.fc_pda(h))
        p_vid = torch.mean(p_frame, axis=0)
        return p_vid, torch.zeros_like(p_vid)
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.fc_pda
        )
    
    @staticmethod
    def pad_encodings(h, num_frames):
        # reshape as batch of vids and pad
        start_ix = 0
        h_ls = []
        for n in num_frames:
            h_ls.append(h[start_ix:(start_ix+n)])
            start_ix += n
        h = pad_sequence(h_ls)
        
        return h
    
    
class VideoClassifier_PI_PI(VideoClassifier):
    """
    This version uses permutation invariant attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifier_PIattn, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)

        # copy linear layer to match dimensions
        self.fc_attention = copy.deepcopy(self.fc_pda)

        # "at least as good" initialization
        self.fc_attention.weight = \
            nn.Parameter(torch.zeros_like(self.fc_attention.weight))
        self.fc_attention.bias = \
            nn.Parameter(torch.ones_like(self.fc_attention.bias))
        
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad_encodings(self.encoder(x), num_frames)
        p_frame = torch.sigmoid(self.fc_pda(h))

        # attention
        alpha = self.fc_attention(h)
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-40
        attn = torch.softmax(alpha, axis=0)

        p_vid = torch.sum(p_frame * attn, axis=0)        
        return p_vid, attn
   

class VideoClassifier_LSTM_PI(VideoClassifier):
    """
    This version uses an LSTM to compute attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True, lstm_hidden_dim = 256):
        super(VideoClassifier_LSTM_PI, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)

        self.lstm_alpha = nn.LSTM(self.fc_pda.weight.shape[-1], lstm_hidden_dim, bidirectional=True)
        self.fc_alpha = nn.Linear(2*lstm_hidden_dim,1)
        
        # "at least as good" initialization"
        self.fc_alpha.weight = \
            nn.Parameter(torch.zeros_like(self.fc_alpha.weight))
        self.fc_alpha.bias = \
            nn.Parameter(torch.ones_like(self.fc_alpha.bias))
        
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad_encodings(self.encoder(x), num_frames)
        ## Swap this and alpha for LSTM / PI integraation
        p_frame = torch.sigmoid(self.fc_pda(h))
        
        # pack frames for lstm
        h = pack_padded_sequence(h, num_frames, enforce_sorted=False)
        eta, _= self.lstm_alpha(h)
        eta, _ = pad_packed_sequence(eta)
        
        # attention
        alpha = self.fc_alpha(eta)
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-40
        attn = torch.softmax(alpha, axis=0)
        
        p_vid = torch.sum(p_frame * attn, axis=0)
        return p_vid, attn
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.fc_pda
        )
    
class VideoClassifier_PI_LSTM(VideoClassifier):
    """
    This version uses an LSTM to compute attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True, lstm_hidden_dim = 256):
        super(VideoClassifier_PI_LSTM, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)

        # copy linear layer to match dimensions
        self.fc_attention = copy.deepcopy(self.fc_pda)
        
        # "at least as good" initialization
        self.fc_attention.weight = \
            nn.Parameter(torch.zeros_like(self.fc_attention.weight))
        self.fc_attention.bias = \
            nn.Parameter(torch.ones_like(self.fc_attention.bias))
        
        # LSTM for prob computation
        self.lstm = nn.LSTM(self.fc_pda.weight.shape[-1], lstm_hidden_dim, bidirectional=True)
        self.fc_pda = nn.Linear(2*lstm_hidden_dim,1)
        


       
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad_encodings(self.encoder(x), num_frames)
        
        # attention
        alpha = self.fc_attention(h)
        # alpha = self.fc_alpha(h)
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-40
        attn = torch.softmax(alpha, axis=0)
        
        # pack frames for lstm
        h = pack_padded_sequence(h, num_frames, enforce_sorted=False)
        eta, _ = self.lstm(h)
        eta, _ = pad_packed_sequence(eta)
        
        # import pdb; pdb.set_trace()
        p_frame = torch.sigmoid(self.fc_pda(eta))
        # p_frame = torch.sigmoid(self.fc_attention(eta))
        
        p_vid = torch.sum(p_frame * attn, axis=0)
        # p_vid = torch.sum(p_frame * attn)/2048
        # p_vid = torch.mean(p_vid)
        # import pdb; pdb.set_trace()
        return p_vid, attn
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.fc_pda
        )
    
class MultiTaskFrameClassifier(nn.Module):
    """
    Output logit scores for pda classification, mode, and view. 
    This network produces a common embedding and outputs logit scores for 
    pda (called 'type' in code), mode, and view. 
    """
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
        
        # compute thte total 
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