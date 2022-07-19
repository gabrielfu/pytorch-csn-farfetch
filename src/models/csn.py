import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CSN(nn.Module):
    def __init__(self, backbone, n_conditions, embedding_size, learned_mask=True, prein=False):
        """
        backbone: The network that projects the inputs into an embedding of embedding_size
        n_conditions: Integer defining number of different similarity notions
        embedding_size: Number of dimensions of the embedding output from the embeddingnet
        learned_mask: Boolean indicating whether masks are learned or fixed
        prein: Boolean indicating whether masks are initialized in equally sized disjoint
            sections or random otherwise
        """
        super(CSN, self).__init__()
        self.learned_mask = learned_mask
        self.backbone = backbone

        # create the mask
        if learned_mask:
            if prein:
                # define masks
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)


    def forward(self, x, c):
        embedded_x = self.backbone(x)
        mask = self.masks(c)
        if self.learned_mask:
            mask = F.relu(mask)
        masked_embedding = embedded_x * mask
        return masked_embedding, mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)