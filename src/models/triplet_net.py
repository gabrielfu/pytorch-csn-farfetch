import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, csn):
        super(TripletNet, self).__init__()
        self.csn = csn

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.csn(x, c)
        embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.csn(y, c)
        embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.csn(z, c)
        mask_norm = (masknorm_norm_x + masknorm_norm_y + masknorm_norm_z) / 3
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm