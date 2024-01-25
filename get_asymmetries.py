import asymmetry_model

def get_asymmetries(backbone_path='./snapshots/mgh_mammo_MIRAI_Base_May20_2019.p', **kwargs):
    """
    This is a hook to preserve the existing interface of get_asymmetries to avoid making too many
    drastic changes throughout the code at once.
    """
    return asymmetry_model.get_asymmetries.get_asymmetries(backbone_path=backbone_path,
        **kwargs)

if __name__ == '__main__':
    get_asymmetries(
        use_latent = True,
        use_topk = False,
        use_emd = False,
        use_mean = False,
        use_std = False,
        use_localized_max = True,
        use_pairwise_max = False,
        use_localized_difs = False,
        pixel_align=True,
        latent_align=False,
        topk_percent = 0.05
    )