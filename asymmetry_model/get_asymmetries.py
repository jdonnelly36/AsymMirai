import cv2
import pandas as pd
import numpy as np
import onconet
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import time
from datetime import datetime

from asymmetry_model.mirai_metadataset import MiraiMetadataset


def extract_mirai_backbone(mirai_backbone, layer):

    if type(mirai_backbone) == str:
        mirai = torch.load(mirai_backbone, map_location='cpu')

    # Allows passing in a prebuilt instance of MIRAI
    elif type(mirai_backbone) == onconet.models.mirai_full.MiraiFull:
        mirai = mirai_backbone
    else:
        raise Exception('mirai should either be an object or a tensor path')

    embedding = []
    for l in mirai.children():
        for m in l.children():
            for name, n in m.named_children():
                embedding.append(n)
                if name == layer:
                    break
            break
        break
    
    return torch.nn.Sequential(*embedding)

def get_asymmetries(use_latent = False, use_segment = False,
                    use_topk = True, use_emd = True, 
                    use_mean = False, use_std = False, 
                    use_localized_max=False, use_pairwise_max=False,
                    use_localized_difs=False, pixel_align=True, latent_align=False,
                    mahalanobis=True,
                    use_hybrid_difs=False,
                    num_segments=(4,3),
                    topk_percent = 1,
                    match_sizes=True,
                    save_path='./{}_feature_asymmetries_max_local_dif.csv',
                    mirai_backbone='./snapshots/mgh_mammo_MIRAI_Base_May20_2019.p',
                    mirai_cutoff_layer='layer4_1',
                    use_latent_mask=True,
                    target_file_name='../combined_mini_dataset_2_11.csv',
                    device=torch.device('cuda'),
                    image_resizer=lambda img: img):
    assert use_hybrid_difs or use_mean or use_emd or use_std or use_localized_max or use_pairwise_max or use_localized_difs, "Error: No comparison metric specified"
    assert not (use_mean and use_emd and use_std and use_localized_max and use_pairwise_max and use_localized_difs), "Error: Multiple comparison metric specified"
    assert not (pixel_align and latent_align)

    # Fixed batch size today
    batch_size, i = 1, 0

    if use_latent:
        emb_model = extract_mirai_backbone(mirai_backbone, mirai_cutoff_layer).to(device)

    img_mean = 7699.5
    img_std = 11765.06

    #NOTE: Want to compute symmetry metric in latent space. First, we need to 
    # remove everything from this after the last encoder layer

    cleaned_df = pd.read_csv(target_file_name)
    target = []
    eids = []
    indices = []
    mlo_x_argmins = []
    mlo_y_argmins = []
    cc_x_argmins = []
    cc_y_argmins = []
    mlo_asyms = []
    cc_asyms = []
            
    start = time.time()
    print(len(target))

    mirai_dataset = MiraiMetadataset(cleaned_df, resizer=image_resizer)

    dataloader = DataLoader(mirai_dataset, batch_size=batch_size)
                
    target_df = pd.DataFrame()
    with torch.no_grad():
        for b, batch in enumerate(tqdm(dataloader)):
        
            target_df = pd.DataFrame()

            total_difs = []
            
            mlo_x_argmin = -1
            mlo_y_argmin = -1
            cc_x_argmin = -1
            cc_y_argmin = -1
            mlo_asym = -1
            cc_asym = -1

            def image_lookup(view):
                # NB: we could avoid the extra image reads by just coverting the existing images to grayscale
                # but we need to handle multiple input colors palettes
                read_img = lambda img: image_resizer(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
                
                if view == 'MLO':
                    l_img, r_img = batch.l_mlo_img[i].numpy(), batch.r_mlo_img[i].numpy()
                    l_mask = read_img(batch.l_mlo_path[i])
                    r_mask = read_img(batch.r_mlo_path[i])
                elif view == 'CC':
                    l_img, r_img = batch.l_cc_img[i].numpy(), batch.r_cc_img[i].numpy()
                    l_mask = read_img(batch.l_cc_path[i])
                    r_mask = read_img(batch.r_cc_path[i])
                else:
                    raise Exception('should be one of MLO, CC')
                
                return l_img, r_img, l_mask, r_mask

            eid = batch.eid[i].item()
            cur_exam = cleaned_df[cleaned_df['exam_id'] == eid]

            for view in ['MLO', 'CC']:

                l_img, r_img, l_img_for_mask, r_img_for_mask = image_lookup(view)

                if use_latent:
                    if pixel_align:
                        # casting to uint16 for affine transform
                        r_img = r_img.astype('uint16')
                        # convert the grayscale image to binary image
                        ret,l_img_mask = cv2.threshold(l_img_for_mask,1,255,0)

                        # calculate moments of binary image
                        M = cv2.moments(l_img_mask)

                        # calculate x,y coordinate of center
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        ret,r_img_mask = cv2.threshold(r_img_for_mask,1,255,0)

                        # calculate moments of binary image
                        new_M = cv2.moments(r_img_mask)

                        # calculate x,y coordinate of center
                        new_cX = int(new_M["m10"] / new_M["m00"])
                        new_cY = int(new_M["m01"] / new_M["m00"])
                        
                        translation_matrix = np.float32([ [1,0,cX-new_cX], [0,1,cY-new_cY] ])  
                        
                        num_rows, num_cols = r_img.shape[:2]   
                        r_img = cv2.warpAffine(r_img, translation_matrix, (num_cols, num_rows))
                        r_img = r_img.astype('int32')
                    
                    l_img_normed = torch.tensor((l_img - img_mean)/img_std)\
                                    .expand(1, 3, *l_img.shape)\
                                    .type(torch.FloatTensor).to(device)
                    
                    r_img_normed = torch.tensor((r_img - img_mean)/img_std)\
                                    .expand(1, 3, *r_img.shape)\
                                    .type(torch.FloatTensor).to(device)
                    if match_sizes:
                        r_img_normed = F.upsample(r_img_normed, size=(l_img_normed.shape[-2], l_img_normed.shape[-1]), mode='bilinear')

                    l_img = emb_model(l_img_normed.to(device))
                    r_img = emb_model(r_img_normed.to(device))
                elif use_segment:
                    l_img = l_img[l_img > 0]
                    l_img = torch.tensor(l_img.astype(np.int32)).view(1, -1).to(device)
                    r_img = r_img[r_img > 0]
                    r_img = torch.tensor(r_img.astype(np.int32)).view(1, -1).to(device)

                else:
                    l_img = torch.tensor(l_img.astype(np.int32)).view(1, -1).to(device)
                    r_img = torch.tensor(r_img.astype(np.int32)).view(1, -1).to(device)

                # Taking standard deviation along each channel,
                # and L2 distance between the resulting vectors
                # as asymmetry metric
                #l_max, _ = torch.max(l_embedding, dim=-1)
                #r_max, _ = torch.max(r_embedding, dim=-1)
                #l_max, _ = torch.max(l_max, dim=-1)
                #r_max, _ = torch.max(r_max, dim=-1)
                if use_topk:
                    percent = topk_percent
                    k = int(percent * l_img.shape[-1] )
                    l_values = torch.topk(l_img, k, dim=-1)[0]
                    k = int(percent * r_img.shape[-1])
                    r_values = torch.topk(r_img, k, dim=-1)[0]
                else:
                    l_values = l_img.view(1, -1)
                    r_values = r_img.view(1, -1)

                if use_emd:
                    emd_res = torch.zeros(l_img.shape[-2])
                    l_values = l_values.cpu()
                    r_values = r_values.cpu()
                    for c in range(r_img.shape[-2]):
                        emd_res[c] = wasserstein_distance(l_values[c,:], r_values[c,:])
                    total_difs = total_difs + [torch.norm(emd_res)]
                elif use_mean:
                    total_difs = total_difs + [torch.norm(torch.mean(l_values, dim=-1) - torch.mean(r_values, dim=-1), p=2)]
                elif use_std:
                    total_difs = total_difs + [torch.norm(torch.std(l_values, dim=-1) - torch.std(r_values, dim=-1), p=2)]
                elif use_hybrid_difs:
                    # This section implements the lower-granularity local difference
                    # method discussed in our 1/27 meeting
                    try:
                        aligned_right = torch.flip(r_img, dims=[-1])
                        if aligned_right.shape[-2] % num_segments[0] != 0:
                            print("WARNING: Height dimension {} not divisible by {}".format(aligned_right.shape[-2], num_segments))

                        if aligned_right.shape[-1] % num_segments[1] != 0:
                            print("WARNING: Width dimension {} not divisible by {}".format(aligned_right.shape[-1], num_segments))

                        '''if use_latent_mask:
                            r_mask_kernel_h = r_img_normed.shape[-2] // aligned_right.shape[-2]
                            r_mask_kernel_w = r_img_normed.shape[-1] // aligned_right.shape[-1]
                            r_mask_kernel_shape = (r_mask_kernel_h, r_mask_kernel_w)
                            r_img_mask = r_img_mask.astype(np.int16)#.reshape(1, r_img_mask.shape[-2], r_img_mask.shape[-1])
                            r_img_mask[r_img_mask > 0] = 1
                            r_mask_latent = F.max_pool2d(torch.tensor(r_img_mask).unsqueeze(0).float(), r_mask_kernel_shape).cuda()
                            print(r_img[r_img == 0].shape, r_img[r_img > 0].shape)
                            r_img = r_img * r_mask_latent[:, :r_img.shape[-2], :r_img.shape[-1]]
                            print(r_img[r_img == 0].shape, r_img[r_img > 0].shape)

                            l_mask_kernel_h = l_img_normed.shape[-2] // l_img.shape[-2]
                            l_mask_kernel_w = l_img_normed.shape[-1] // l_img.shape[-1]
                            l_mask_kernel_shape = (l_mask_kernel_h, l_mask_kernel_w)
                            l_img_mask = l_img_mask.astype(np.int16)#.reshape(1, l_img_mask.shape[-2], l_img_mask.shape[-1])
                            l_img_mask[l_img_mask > 0] = 1
                            l_mask_latent = F.max_pool2d(torch.tensor(l_img_mask).unsqueeze(0).float(), l_mask_kernel_shape).cuda()

                            l_img = l_img * l_mask_latent[:, :l_img.shape[-2], :l_img.shape[-1]]'''

                        kernel_h = aligned_right.shape[-2] // num_segments[0]
                        kernel_w = aligned_right.shape[-1] // num_segments[1]
                        pooling_kernel_shape = (kernel_h, kernel_w)

                        l_img = F.max_pool2d(l_img, pooling_kernel_shape, 
                                    stride=(kernel_h, kernel_w))
                        aligned_right = F.max_pool2d(aligned_right, pooling_kernel_shape, 
                                    stride=(kernel_h, kernel_w))

                        dif = torch.norm(torch.abs(l_img - aligned_right), dim=-3)[0]
                        
                        if use_latent_mask:
                            r_mask_kernel_h = r_img_normed.shape[-2] // dif.shape[-2]
                            r_mask_kernel_w = r_img_normed.shape[-1] // dif.shape[-1]
                            r_mask_kernel_shape = (r_mask_kernel_h, r_mask_kernel_w)
                            r_img_mask = r_img_mask.astype(np.int16)#.reshape(1, r_img_mask.shape[-2], r_img_mask.shape[-1])
                            r_img_mask[r_img_mask > 0] = 1
                            r_mask_latent = F.max_pool2d(torch.tensor(r_img_mask).unsqueeze(0).float(), r_mask_kernel_shape).to(device)

                            l_mask_kernel_h = l_img_normed.shape[-2] // dif.shape[-2]
                            l_mask_kernel_w = l_img_normed.shape[-1] // dif.shape[-1]
                            l_mask_kernel_shape = (l_mask_kernel_h, l_mask_kernel_w)
                            l_img_mask = l_img_mask.astype(np.int16)#.reshape(1, l_img_mask.shape[-2], l_img_mask.shape[-1])
                            l_img_mask[l_img_mask > 0] = 1
                            l_mask_latent = F.max_pool2d(torch.tensor(l_img_mask).unsqueeze(0).float(), l_mask_kernel_shape).to(device)

                            if l_mask_latent.shape[-2] > r_mask_latent.shape[-2]:
                                combined_mask = l_mask_latent[:, :dif.shape[-2], :dif.shape[-1]]
                            else:
                                combined_mask = r_mask_latent[:, :dif.shape[-2], :dif.shape[-1]]
                                
                            print(dif[dif == 0].shape, dif[dif > 0].shape)
                            print(dif.shape, combined_mask.shape)
                            dif = dif * combined_mask[0]
                            print(dif[dif == 0].shape, dif[dif > 0].shape)

                        max_by_ftr, y_argmin = torch.max(dif, dim=-1)
                        max_by_ftr, x_argmin = torch.max(max_by_ftr, dim=-1)

                        if view == 'MLO':
                            mlo_x_argmin = x_argmin.item()
                            mlo_y_argmin = y_argmin[x_argmin].item()
                            mlo_asym = max_by_ftr.item()
                        else:
                            cc_x_argmin = x_argmin.item()
                            cc_y_argmin = y_argmin[x_argmin].item()
                            cc_asym = max_by_ftr.item()
                        total_difs = total_difs + [max_by_ftr]
                    except Exception as e:
                        # This can happen if the two sides have different shapes...
                        # will need to address this at some point
                        #resampled_r = F.upsample(aligned_right, size=(l_img.shape[0], l_img.shape[-2], l_img.shape[-1]), mode='bilinear')
                        if l_mask_latent.shape[-2] > r_mask_latent.shape[-2]:
                                combined_mask = l_mask_latent[:, :dif.shape[-2], :dif.shape[-1]]
                        else:
                            combined_mask = r_mask_latent[:, :dif.shape[-2], :dif.shape[-1]]

                        print(dif[dif == 0].shape, dif[dif > 0].shape)
                        dif = dif * combined_mask[0]
                        print(dif[dif == 0].shape, dif[dif > 0].shape)
                        total_difs = total_difs
                        break
                elif use_localized_difs:
                    try:
                        aligned_right = torch.flip(r_img, dims=[-1])
                        dif = torch.abs(l_img - aligned_right)
                        max_by_ftr, _ = torch.max(dif, dim=-1)
                        max_by_ftr, _ = torch.max(max_by_ftr, dim=-1)
                        total_difs = total_difs + [torch.norm(max_by_ftr)]
                    except:
                        # This can happen if the two sides have different shapes...
                        # will need to address this at some point
                        print(l_img.shape)
                        print(r_img.shape)
                        resampled_r = F.upsample(aligned_right, size=(l_img.shape[0], l_img.shape[-2], l_img.shape[-1]), mode='bilinear')
                        total_difs = -1
                        break
                elif use_localized_max:
                    # NOTE: with use_localized_max, we aim to find the greatest difference
                    # between activation in the left and activation in the right breast for 
                    # each individual feature across fixed spatial locations; that is, something
                    # like for each filter f, max_ij (|R_ij - L_ij|)
                    try:
                        if mahalanobis:
                            # Should be a (d, d) matrix
                            print(l_img.view(l_img.shape[1], -1).shape)
                            cov = torch.cov(l_img.view(l_img.shape[1], -1))
                            for j in range(cov.shape[0]):
                                if cov[j, j] == 0:
                                    print("Found 0 variance with index {}".format(i))
                                    cov[j, j] += 1e-4
                            # (1, d, h, w)
                            dif = l_img - r_img
                            # (1, h, d, w)
                            dif = torch.transpose(dif, 1, 2)
                            # (1, h, w, d)
                            dif = torch.transpose(dif, 2, 3)
                            # (hw, 1, d)
                            dif_T = dif.view(-1, 1, dif.shape[-1])
                            # (hw, d, 1)
                            dif = dif.view(-1, dif.shape[-1], 1)
                            # Should be (hw, d)
                            r_term = torch.matmul(torch.inverse(cov), dif)
                            dif = torch.sqrt(dif_T @ r_term)
                            max_by_ftr = torch.max(dif)
                            total_difs = total_difs + [max_by_ftr]

                        elif latent_align:
                            aligned_right = torch.flip(r_img, dims=[-1])
                            dif = l_img - aligned_right

                            dif_norms = torch.norm(dif, p=2, dim=-3)

                            max_dif, _ = torch.max(dif_norms, dim=-1)
                            max_dif, _ = torch.max(max_dif, dim=-1)
                            total_difs = total_difs + [max_dif]
                        else:
                            dif = l_img - r_img

                            dif_norms = torch.norm(dif, p=2, dim=-3)

                            max_dif, _ = torch.max(dif_norms, dim=-1)
                            max_dif, _ = torch.max(max_dif, dim=-1)
                            total_difs = total_difs + [max_dif]
                    except Exception as e:
                        # This can happen if the two sides have different shapes...
                        # will need to address this at some point
                        print(l_img.shape)
                        resampled_r = F.upsample(aligned_right, size=(l_img.shape[-2], l_img.shape[-1]), mode='bilinear')
                        dif = l_img - resampled_r

                        dif_norms = torch.norm(dif, p=2, dim=-3)

                        max_dif, _ = torch.max(dif_norms, dim=-1)
                        max_dif, _ = torch.max(max_dif, dim=-1)
                        print(max_dif.shape)
                        total_difs = total_difs + [max_dif]
                elif use_pairwise_max:
                    with torch.no_grad():
                        # Reshaping to (512, (hxw))
                        l_img = l_img.view(l_img.shape[1], -1)
                        r_img = r_img.view(r_img.shape[1], -1)

                        # This should be of shape (512, (hxw), (hxw))
                        try:
                            pairwise_difs = l_img - r_img.view(r_img.shape[0], 1, r_img.shape[-1])
                        except:
                            total_dif = -1
                            break

                        # For each filter, take the max difference along spatial dimensions
                        max_by_ftr, _ = torch.max(pairwise_difs, dim=-1)
                        max_by_ftr, _ = torch.max(max_by_ftr, dim=-1)
                        total_difs = total_difs + [torch.norm(max_by_ftr, p=2)]

                    '''l_percent = (l_rough_seg[l_rough_seg > l_mean].size) \
                                / (l_rough_seg.size)
                    r_percent = (r_rough_seg[r_rough_seg > r_mean].size) \
                                / (r_rough_seg.size)'''

            #if isinstance(total_difs, int):
            #    total_dif = total_difs[0]
            #    target = target + [total_difs]
            if len(total_difs) == 0:
                total_dif = 0
                target = target + [0]
            else:
                total_dif = torch.mean(torch.stack(total_difs)).cpu().item()
                target = target + [total_dif]
            eids = eids + [eid]
            indices = indices + [b + b*i]

            mlo_x_argmins = mlo_x_argmins + [mlo_x_argmin]
            mlo_y_argmins = mlo_y_argmins + [mlo_y_argmin]
            mlo_asyms = mlo_asyms + [mlo_asym]
            cc_x_argmins = cc_x_argmins + [cc_x_argmin]
            cc_y_argmins = cc_y_argmins + [cc_y_argmin]
            cc_asyms = cc_asyms + [cc_asym]

            if b % 1 == 0:
                print("Completed {} iters in {} seconds".format(i, time.time() - start))
                print("Iter {}: Found exam {} with ytc {} has difs {}".format(i, eid, cur_exam['years_to_cancer'].values[0], total_dif))

                target_df['indices'] = indices
                target_df['exam_id'] = eids
                target_df['asymmetries'] = target
                
                target_df['mlo_x_argmin'] = mlo_x_argmins
                target_df['mlo_y_argmin'] = mlo_y_argmins
                target_df['cc_x_argmin'] = cc_x_argmins
                target_df['cc_y_argmin'] = cc_y_argmins
                target_df['mlo_asym'] = mlo_asyms
                target_df['cc_asym'] = cc_asyms

    target_df.to_csv(save_path.format(datetime.now().date()), index=False)
    return target_df
                
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