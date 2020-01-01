import torch.nn as nn
import torch
import pcpnet
import torch.nn.functional as F
class ngran(nn.Module):
    def __init__(self, points_per_patch=256,  dim_pts=3, num_gpts=128, dim_gpts=1,                           
                use_mask=False,  sym_op='max', ith = 0, 
                use_point_stn=True, use_feat_stn=True,               
                device=0):
       

                super(ngran, self).__init__()
                self.wpcp = pcpnet.PCPNet(num_pts=points_per_patch, dim_pts=dim_pts, num_gpts=num_gpts, dim_gpts=1, 
                            use_point_stn=use_point_stn, 
                            use_feat_stn=use_feat_stn, device=device,
                            b_pred = True, use_mask=False, sym_op=sym_op, ith = 0)

    def forward(self, pts, dist=None, gfeat=None, patch_rot=None):
        
        batch_size = pts.size(0)
        data_size = pts.size(2)
        pt_weight, _, _, _, _ = self.wpcp(pts, dist, gfeat=None, patch_rot=None)
        log_probs = F.logsigmoid(pt_weight)
        log_probs = log_probs.view(batch_size, -1)
        normalizer = torch.logsumexp(log_probs, dim=1)
        normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
        log_probs = log_probs - normalizer
        log_probs = log_probs.view(batch_size, 1, data_size, 1)
      
        return log_probs