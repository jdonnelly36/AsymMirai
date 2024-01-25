import torch

import sys
sys.path.append('../')

def extract_mirai_backbone(path):

    # first pull mirai onto the CPU to avoid putting the whole transformer on the gpu
    mirai = torch.load(path, map_location='cpu')
    embedding = []
    for l in mirai.children():
        for m in l.children():
            for name, n in m.named_children():
                embedding.append(n)
                if name == 'layer4_1':
                    break
            break
        break
    
    return torch.nn.Sequential(*embedding)

# Torch by default adds data elements to tensors, but we need numpy arrays because our functions are
# written to uint16, which is not supported by torch
# We could just change to int32 - but we need to make sure everything works as expected.
# There's a placeholder in the MiraiMetadataset for changing the datatypes of the images.
def exam_collate_function(batch):
    eids = [b.eid for b in batch]

    return batch

class LocalizedDifModel(torch.nn.Module):

    def __init__(self, alignment_space=None,
                asymmetry_metric=None,
                latent_h=5,
                latent_w=5,
                embedding_channel=512,
                embedding_model=None,
                use_stretch=False,
                train_backbone=False,
                initial_asym_mean=8000000,
                initial_asym_std=1520381,
                flexible_asymmetry=False,
                use_stretch_matrix=False,
                device_ids=[0],
                use_addon_layers=False,
                topk_for_heatmap=None,
                use_bias=False,
                use_bn=False):
        super().__init__()
        self.initial_asym_mean = initial_asym_mean
        self.initial_asym_std = initial_asym_std
        self.use_bn = use_bn
        
        if use_bn:
            print("Using batch norm instead of learned stats")
            self.use_bias = False
            self.bn = torch.nn.BatchNorm2d(embedding_channel).cuda()
            
            self.learned_asym_mean = torch.tensor(initial_asym_mean).type(torch.FloatTensor).cuda()
            self.learned_asym_mean.requires_grad = True
            
            self.learned_asym_std = torch.tensor(initial_asym_std).type(torch.FloatTensor).cuda()
            self.learned_asym_std.requires_grad = True
        elif use_bias:
            self.learned_asym_mean = torch.tensor(initial_asym_mean).type(torch.FloatTensor).cuda()
            self.learned_asym_mean.requires_grad = True
            
            self.learned_asym_std = torch.tensor(initial_asym_std).type(torch.FloatTensor).cuda()
            self.learned_asym_std.requires_grad = True
        else:
            self.bias_term = None
        
        # Getting the embedding layers from Mirai
        if embedding_model is None:
            mirai = extract_mirai_backbone('../snapshots/mgh_mammo_MIRAI_Base_May20_2019.p')
            mirai.requires_grad = train_backbone
            if use_addon_layers:
                print("USING ADDON LAYERS")
                self.conv1 = torch.nn.Conv2d(512, 512, 3, padding=1)
                self.conv1.requires_grad = True
                self.backbone = torch.nn.Sequential(mirai, self.conv1, torch.nn.ReLU())
            else:
                self.backbone = mirai
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=device_ids).cuda()
        else:
            self.backbone = embedding_model.cuda()
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=device_ids)
            self.backbone.requires_grad = train_backbone
        self.train_backbone = train_backbone

        # The asymmetry function to use
        if asymmetry_metric is None:
            self.asymmetry_metric = self.fallback_asym
        else:
            self.asymmetry_metric = asymmetry_metric
        self.flexible_asymmetry = flexible_asymmetry
        
        
        # What size we want to pool our latent space to
        self.latent_h = latent_h
        self.latent_w = latent_w
        
        # The top k activations we want to consider
        self.topk_for_heatmap = topk_for_heatmap
        if not self.topk_for_heatmap is None:
            self.topk_weights = (torch.arange(self.topk_for_heatmap) 
                                 / (torch.sum(torch.arange(self.topk_for_heatmap))))\
                                .type(torch.FloatTensor)\
                                .cuda()
            self.topk_weights.requires_grad = True
            
        # Indicator for whether we want to flip in pixel or latent space
        self.alignment_space = alignment_space

        # Whether or not to use a stretching procedure
        self.use_stretch = use_stretch
        
        # Whether to use the matrix version of stretching or the
        # vector version
        self.use_stretch_matrix = use_stretch_matrix

        if use_stretch_matrix:
            # Matrix weights for doing stretched L2
            self.cc_stretch_params = (torch.eye(embedding_channel) 
                                       + torch.normal(mean=0, std=0.01, size=(embedding_channel * embedding_channel,))\
                                            .view(embedding_channel, embedding_channel))\
                                        .type(torch.FloatTensor).cuda()
            self.cc_stretch_params.requires_grad = True
            self.cc_stretch_bias = torch.normal(mean=0, std=0.5, size=(1, embedding_channel,1,1))\
                                        .type(torch.FloatTensor).cuda()
            self.cc_stretch_bias.requires_grad = True
            
            self.mlo_stretch_params = (torch.eye(embedding_channel) 
                                       + torch.normal(mean=0, std=0.01, size=(embedding_channel * embedding_channel,))\
                                           .view(embedding_channel, embedding_channel))\
                                        .type(torch.FloatTensor).cuda()
            self.mlo_stretch_params.requires_grad = True
            self.mlo_stretch_bias = torch.normal(mean=0, std=0.5, size=(1, embedding_channel,1,1))\
                                        .type(torch.FloatTensor).cuda()
            self.mlo_stretch_bias.requires_grad = True
        else:
            # Vector weights for doing stretched L2
            self.cc_stretch_params = torch.ones(embedding_channel).type(torch.FloatTensor).cuda()
            self.cc_stretch_params.requires_grad = True
            self.mlo_stretch_params = torch.ones(embedding_channel).type(torch.FloatTensor).cuda()
            self.mlo_stretch_params.requires_grad = True
    
    def fallback_asym(left, right, **kwargs):
        return torch.norm(left - right)

    def stretched_asymmetries(self, left, right, mode="MLO"):
        if mode == "MLO":
            left_stretched = self.mlo_stretch_params.view(1, -1, 1, 1) * left
            right_stretched = self.mlo_stretch_params.view(1, -1, 1, 1) * right
        else:
            left_stretched = self.cc_stretch_params.view(1, -1, 1, 1) * left
            right_stretched = self.cc_stretch_params.view(1, -1, 1, 1) * right

        norms = torch.norm(left_stretched - right_stretched, dim=-1)
        return norms
    
    def matrix_stretch(self, embedding, mode="MLO"):
        old_embedding_shape = embedding.shape

        # We need to move the channel dimension to be the first dim so stretching
        # occurs along that axis, not spatial or batch
        if mode == "MLO":
            final_embedding = (self.mlo_stretch_params @ embedding.transpose(0, 1)\
                                            .reshape(old_embedding_shape[1], -1))\
                                            .view(old_embedding_shape[1], old_embedding_shape[0], 
                                                    old_embedding_shape[2], old_embedding_shape[3])\
                                            .transpose(0, 1)
        else:
            final_embedding = (self.cc_stretch_params @ embedding.transpose(0, 1)\
                                            .reshape(old_embedding_shape[1], -1))\
                                            .view(old_embedding_shape[1], old_embedding_shape[0], 
                                                    old_embedding_shape[2], old_embedding_shape[3])\
                                            .transpose(0, 1)
        
        return final_embedding

    def forward(self, left_cc, right_cc, left_mlo, right_mlo, exam_list=None, **kwargs):
        if exam_list is None:
            exam_list = [(left_cc, right_cc, 'CC'), (left_mlo, right_mlo, 'MLO')]
        mean_asymmetry = 0
        
        count_list = torch.ones(exam_list[0][0].shape[0]) * len(exam_list)
        count_list.requires_grad = False
        
        other_returns = []
        for left, right, view in exam_list:
            sub_list = torch.zeros_like(count_list)
            sub_list.requires_grad = False
            sub_list[torch.sum(left, dim=(1,2,3)).cpu() == torch.zeros_like(count_list)] = 1
            count_list = count_list - sub_list
            
            if self.alignment_space == 'pixel':
                right = torch.flip(right, dims=[-1])
            
            with torch.set_grad_enabled(self.train_backbone):
                left_embedding = self.backbone(left)
                right_embedding = self.backbone(right)
            if self.alignment_space == 'latent':
                right_embedding = torch.flip(right_embedding, dims=[-1])

            if self.use_stretch:
                if self.use_stretch_matrix:
                    left_embedding = self.matrix_stretch(left_embedding, mode=view)
                    right_embedding = self.matrix_stretch(right_embedding, mode=view)
                else:
                    if 'CC' in view:
                        left_embedding = self.cc_stretch_params.view(1, -1, 1, 1) * left_embedding
                        right_embedding = self.cc_stretch_params.view(1, -1, 1, 1) * right_embedding
                    else:
                        left_embedding = self.mlo_stretch_params.view(1, -1, 1, 1) * left_embedding
                        right_embedding = self.mlo_stretch_params.view(1, -1, 1, 1) * right_embedding
            
            if self.use_bn:
                left_embedding = self.bn(left_embedding)
                right_embedding = self.bn(right_embedding)

            asymmetry, other = self.asymmetry_metric(left_embedding, right_embedding, 
                                                     latent_h=self.latent_h,
                                                     latent_w=self.latent_w,
                                                     flexible=self.flexible_asymmetry,
                                                     topk=self.topk_for_heatmap)#,
                                                     #bias_params=self.cc_stretch_bias if 'CC' in view else self.mlo_stretch_bias)
            
            if not self.topk_for_heatmap is None:
                asymmetry = asymmetry @ self.topk_weights.view(-1, 1)
                asymmetry = asymmetry.view(-1)
                
            other_returns.append(other)
            mean_asymmetry += (asymmetry * (1 - sub_list).cuda())
            
        count_list = count_list.cuda()
        if not self.use_bn:
            scaled_asym = ((mean_asymmetry / count_list) - self.learned_asym_mean) / self.learned_asym_std
        else:
            scaled_asym = mean_asymmetry / count_list
            
        pred = torch.sigmoid(scaled_asym)
        return torch.stack([1 - pred, pred], dim=1), other_returns

        