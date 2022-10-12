import torch.nn as nn
import timm
import torch
import copy
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchvision.transforms import Resize

class AttentionNetwork(nn.Module):
    def __init__(self):
        super(AttentionNetwork, self).__init__()

        # feature encoder
        self.encoder = nn.Sequential(
            Resize((32,32)),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # linear head for computing un-normalized temporal attention
        self.fc_alpha = nn.Linear(96, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc_alpha(x)
        return x

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
        h = self.pad(self.encoder(x), num_frames)
        logit_frame = self.fc_pda(h)
        for ix, n in enumerate(num_frames):
            logit_frame[n:, ix] = 0
        
        p_vid = torch.sum(logit_frame, axis=0) / torch.tensor(num_frames, dtype = logit_frame.dtype, device = logit_frame.device)[:,None]
        return p_vid, torch.zeros_like(p_vid)
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.fc_pda
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
    
    
class VideoClassifier_PI_PI(VideoClassifier):
    """
    This version uses permutation invariant attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifier_PI_PI, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)

        # copy linear layer to match dimensions
        self.fc_attention = copy.deepcopy(self.fc_pda)

        # "at least as good" initialization
        self.fc_attention.weight = \
            nn.Parameter(torch.zeros_like(self.fc_attention.weight))
        self.fc_attention.bias = \
            nn.Parameter(torch.ones_like(self.fc_attention.bias))
        
        # # random init
        # self.fc_attention.weight = \
        #     nn.Parameter(torch.randn_like(self.fc_attention.weight))
        # self.fc_attention.bias = \
        #     nn.Parameter(torch.randn_like(self.fc_attention.bias))
        
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad(self.encoder(x), num_frames)
        p_frame = torch.sigmoid(self.fc_pda(h))

        # attention
        alpha = self.fc_attention(h)
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-40
        attn = torch.softmax(alpha, axis=0)

        p_vid = torch.sum(p_frame * attn, axis=0)        
        return p_vid, attn
    
class VideoClassifier_PIlw_PI(VideoClassifier):
    """
    This version uses permutation invariant attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True):
        super(VideoClassifier_PIlw_PI, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)

        # copy linear layer to match dimensions
        self.attn_net = AttentionNetwork()
        
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad(self.encoder(x), num_frames)
        #p_frame = torch.sigmoid(self.fc_pda(h))
        p_frame = self.fc_pda(h)

        # attention
        alpha = self.attn_net(x)
        alpha = self.pad(alpha, num_frames)

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
        h = self.pad(self.encoder(x), num_frames)
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
        h = self.pad(self.encoder(x), num_frames)
        
        # attention
        alpha = self.fc_attention(h)
        for ix, n in enumerate(num_frames):
            alpha[n:, ix]=-40
        attn = torch.softmax(alpha, axis=0)
        
        # pack frames for lstm
        h = pack_padded_sequence(h, num_frames, enforce_sorted=False)
        eta, _ = self.lstm(h)
        eta, _ = pad_packed_sequence(eta)
        
        p_frame = torch.sigmoid(self.fc_pda(eta))
        
        p_vid = torch.sum(p_frame * attn, axis=0)

        return p_vid, attn
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.fc_pda
        )

class VideoClassifier_LSTM_LSTM(VideoClassifier):
    """
    This version uses an LSTM to compute attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True, lstm_hidden_dim = 256):
        super(VideoClassifier_LSTM_LSTM, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)
        
        # LSTM 1 for prob computation
        self.lstm_alpha = nn.LSTM(self.fc_pda.weight.shape[-1], lstm_hidden_dim, bidirectional=True)
        """
        Don't know if fc_alpha is necessary, decided not to us it in forward method
        """
        self.fc_alpha = nn.Linear(2*lstm_hidden_dim,1)
        
        # "at least as good" initialization"
        self.fc_alpha.weight = \
            nn.Parameter(torch.zeros_like(self.fc_alpha.weight))
        self.fc_alpha.bias = \
            nn.Parameter(torch.ones_like(self.fc_alpha.bias))
        
        # LSTM 2
        self.lstm_bravo = nn.LSTM(lstm_hidden_dim*2, lstm_hidden_dim//2, bidirectional=True)
        self.fc_bravo = nn.Linear(lstm_hidden_dim,1)
        
        # "at least as good" initialization"
        """
        Maybe we should randomize these?
        """
        self.fc_bravo.weight = \
            nn.Parameter(torch.zeros_like(self.fc_bravo.weight))
        self.fc_bravo.bias = \
            nn.Parameter(torch.ones_like(self.fc_bravo.bias))

       
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad_encodings(self.encoder(x), num_frames)
        
        # pack frames for lstm 1
        h = pack_padded_sequence(h, num_frames, enforce_sorted=False)
        eta, _ = self.lstm_alpha(h)
        eta, _ = pad_packed_sequence(eta)
        # Optional linear layer?
        # alpha = self.fc_alpha(eta)
        
        # pack frames for lstm 2
        h = pack_padded_sequence(eta, num_frames, enforce_sorted=False)
        iota, _ = self.lstm_bravo(h)
        iota, _ = pad_packed_sequence(iota)
        bravo = self.fc_bravo(iota)
                
        p_vid = torch.mean(bravo, axis=0)
  
        return p_vid, _
    
    def get_frame_classifier(self):
        return nn.Sequential(
            self.encoder,
            self.fc_pda
        )

class VideoClassifier_LSTM_CNN(VideoClassifier):
    """
    This version uses an LSTM to compute attention
    """
    def __init__(self, frame_classifier, encoder_frozen=True, frame_classifier_frozen=True, lstm_hidden_dim = 256):
        super(VideoClassifier_LSTM_CNN, self).__init__(frame_classifier, encoder_frozen=True, frame_classifier_frozen=True)
        
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # LSTM for prob computation
        self.lstm = nn.LSTM(self.fc_pda.weight.shape[-1], lstm_hidden_dim, bidirectional=True)
        """
        Don't know if fc_alpha is necessary, decided not to us it in forward method
        """
        self.fc_alpha = nn.Linear(2*lstm_hidden_dim,1)
        
        # "at least as good" initialization"
        self.fc_alpha.weight = \
            nn.Parameter(torch.zeros_like(self.fc_alpha.weight))
        self.fc_alpha.bias = \
            nn.Parameter(torch.ones_like(self.fc_alpha.bias))
        
        # CNN
        self.conv_layer1 = nn.Conv1d(1,4,kernel_size = 3, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm1d(4)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(255, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Load Predefined CNN using TIMM
        # self.cnn_model = timm.create_model('resnet18', pretrained=False)
        
    def forward(self, x, num_frames):
        # get frame embeddings
        h = self.pad_encodings(self.encoder(x), num_frames)
        # import pdb; pdb.set_trace()
        
        # pack frames for lstm
        h = pack_padded_sequence(h, num_frames, enforce_sorted=False)
        _, (eta,_) = self.lstm(h)
        # Combine h_before and h_after
        eta_combined = eta.transpose(0,1).reshape(len(num_frames), 2*self.lstm_hidden_dim)
                
        # CNN stuff
        # Refactor for CNN
        zulu = eta_combined[:,None,:]
        # Send it down the pipe
        zulu = self.conv_layer1(zulu)
        zulu = self.batchnorm1(zulu)
        zulu = self.relu1(zulu)
        zulu = self.maxpool1(zulu)
        p_vid = self.linear1(zulu)
        # p_vid = self.logsoftmax(zulu).float()
        # import pdb; pdb.set_trace()
  
        return p_vid, _
    
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