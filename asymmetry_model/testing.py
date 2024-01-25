from asymmetry_metrics import hybrid_asymmetry
from mirai_localized_dif_head import LocalizedDifModel
import pandas as pd
import cv2
import numpy as np
import torch

def get_imgs_for_eid(cleaned_df, eid,
                    img_mean=7699.5,
                    img_std=11765.06):
    cur_exam = cleaned_df[cleaned_df['exam_id'].values == eid]
        
    view = 'CC'
    def indices_for_side_view(side):
        indices = np.logical_and(cur_exam['view'].values == view, cur_exam['laterality'].values == side)
        return indices

    if len(cur_exam[indices_for_side_view('L')]['file_path'].values) > 0:
        l_path = cur_exam[indices_for_side_view('L')]['file_path'].values[-1]
    else:
        print("Missing {} left view for exam {}".format(view, eid))
        return None, None, None, None

    if len(cur_exam[indices_for_side_view('R')]['file_path'].values) > 0:
        r_path = cur_exam[indices_for_side_view('R')]['file_path'].values[-1]
    else:
        print("Missing {} right view for exam {}".format(view, eid))
        return None, None, None, None

    l_img_cc = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
    l_img_cc = torch.tensor((l_img_cc - img_mean)/img_std)\
                                .expand(1, 3, *l_img_cc.shape)\
                                .type(torch.FloatTensor)\
                                .cuda()
    r_img_cc = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)
    r_img_cc = torch.tensor((r_img_cc - img_mean)/img_std)\
                                .expand(1, 3, *r_img_cc.shape)\
                                .type(torch.FloatTensor)\
                                .cuda()

    view = 'MLO'
    def indices_for_side_view(side):
        indices = np.logical_and(cur_exam['view'].values == view, cur_exam['laterality'].values == side)
        return indices

    if len(cur_exam[indices_for_side_view('L')]['file_path'].values) > 0:
        l_path = cur_exam[indices_for_side_view('L')]['file_path'].values[-1]
    else:
        print("Missing {} left view for exam {}".format(view, eid))
        return None, None, None, None

    if len(cur_exam[indices_for_side_view('R')]['file_path'].values) > 0:
        r_path = cur_exam[indices_for_side_view('R')]['file_path'].values[-1]
    else:
        print("Missing {} right view for exam {}".format(view, eid))
        return None, None, None, None

    l_img_mlo = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
    l_img_mlo = torch.tensor((l_img_mlo - img_mean)/img_std)\
                                .expand(1, 3, *l_img_mlo.shape)\
                                .type(torch.FloatTensor)\
                                .cuda()
    r_img_mlo = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)
    r_img_mlo = torch.tensor((r_img_mlo - img_mean)/img_std)\
                                .expand(1, 3, *r_img_mlo.shape)\
                                .type(torch.FloatTensor)\
                                .cuda()
    
    return l_img_cc, r_img_cc, l_img_mlo, r_img_mlo

def main():
    model = LocalizedDifModel(asymmetry_metric=hybrid_asymmetry,
                embedding_channel=512,
                embedding_model=None,
                use_stretch=False,
                train_backbone=False)
    
    target_file_name = '../1_19_mirai_form_cohorts_1-2.csv'
    cleaned_df = pd.read_csv(target_file_name)
    
    for i, eid in enumerate(cleaned_df['exam_id'].unique()):
        
        l_img_cc, r_img_cc, l_img_mlo, r_img_mlo = get_imgs_for_eid(cleaned_df, eid)
        if (l_img_cc is None) or (r_img_cc is None) or \
            (l_img_mlo is None) or (r_img_mlo is None):
            continue
        
        print(model(l_img_cc, r_img_cc, l_img_mlo, r_img_mlo))
        del r_img_mlo, l_img_mlo, l_img_cc, r_img_cc