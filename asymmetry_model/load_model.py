import torch
from mirai_localized_dif_head import LocalizedDifModel
from asymmetry_metrics import hybrid_asymmetry

# This function loads pre-trained weights and returns the asymmetry model
# constructed from them
def load_model(use_stretch=True, stretch_vec_path='./training_preds/stretch_vector_epoch_0_3_2.pt',
               backbone_path=None, device=2):
    model = LocalizedDifModel(asymmetry_metric=hybrid_asymmetry,
                embedding_channel=512,
                embedding_model=backbone_path,
                use_stretch=use_stretch,
                train_backbone=False,
                flexible_asymmetry=True,
                use_stretch_matrix=True)
    stretch_params = torch.load(stretch_vec_path, map_location=lambda storage, loc: storage.cuda(device))
    model.stretch_params = stretch_params
    return model