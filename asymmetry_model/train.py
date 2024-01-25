from asymmetry_metrics import hybrid_asymmetry
from mirai_localized_dif_head import LocalizedDifModel
from mirai_metadataset import MiraiMetadataset
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time


def main(device_ids=[2],
        num_epochs=50,
        use_stretch=True,
        train_backbone=False,
        flexible_asymmetry=True,
        use_stretch_matrix=False,
        batch_size=40,
        lr=1,
        max_workers=20,
        initial_asym_mean=4000,
        initial_asym_std=200,
        latent_h=5,
        latent_w=5,
        use_addon_layers=False,
        save_file_suffix="full",
        use_all_training_data=False,
        oversample_cancer_rate=13,
        topk_for_heatmap=None,
        align_images=False,
        multiple_pairs_per_exam=False,
        verbose=False,
        model=None,
        batch_acc=5,
        use_bias=False,
        lr_step_size=None,
        weight_decay=1e-3,
        use_bn=False,
        linear_only_epochs=[]):
    
    # Setting the primary cuda device to be the first listed device
    torch.cuda.set_device(device_ids[0])
    val_accs = []
    avg_loss_vals = []
    
    augmentations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=(-20, 20)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    
    def crop(img):
        nonzero_inds = torch.nonzero(img - torch.min(img))
        top = torch.min(nonzero_inds[:, 0])
        left = torch.min(nonzero_inds[:, 1])
        bottom = torch.max(nonzero_inds[:, 0])
        right = torch.max(nonzero_inds[:, 1])

        return img[top:bottom, left:right]
    
    def resize_and_normalize(img, use_crop=False, augment=True):
        img_mean = 7699.5
        img_std = 11765.06
        target_size = (1664, 2048)
        dummy_batch_dim = False
        
        if np.sum(img) == 0:
            img = torch.tensor(img).expand(1, 3, *img.shape)\
                            .type(torch.FloatTensor)
            return F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')[0]
        
        # Adding a dummy batch dimension if necessary
        if len(img.shape) == 3:
            img = torch.unsqueeze(img, 0)
            dummy_batch_dim = True
        
        with torch.no_grad():
            if use_crop:
                img = crop(torch.tensor((img - img_mean)/img_std))
            else:
                img = torch.tensor((img - img_mean)/img_std)
            img = img.expand(1, 3, *img.shape)\
                            .type(torch.FloatTensor)
            img_resized = F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')
            if augment:
                img_resized = augmentations(img_resized[0])
                return img_resized
        #img_resized = img
        
        if dummy_batch_dim:
            return img_resized[0]
        else:
            return img_resized[0]

    val_file_name = '../../4_26_1000_patient_mirai_form_cohorts_1-2.csv'
    val_df = pd.read_csv(val_file_name)
    val_dataset = MiraiMetadataset(val_df, resizer=resize_and_normalize, mode="val", align_images=align_images,
                                   multiple_pairs_per_exam=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(max_workers, batch_size))
    
    if not use_all_training_data:
        train_file_name = '../../cohorts_3-8_combined_1605_cancer_8000_healthy_dataset_2_24.csv'
        train_df = pd.read_csv(train_file_name)
        pos_count = train_df[train_df['years_to_cancer'] < 10].shape[0]
        neg_count = train_df[train_df['years_to_cancer'] > 10].shape[0]
        train_dataset = MiraiMetadataset(train_df, resizer=resize_and_normalize, mode="training", align_images=align_images, 
                                         oversample_cancer_rate=oversample_cancer_rate, multiple_pairs_per_exam=multiple_pairs_per_exam)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=multiple_pairs_per_exam,
                                      shuffle=True, num_workers=min(max_workers, batch_size))
    else:
        train_file_name = '../../2_24_mirai_form_extended_no_diag_cohorts_3-8.csv'
        train_df = pd.read_csv(train_file_name)
        pos_count = train_df[train_df['years_to_cancer'] < 10].shape[0] * oversample_cancer_rate
        neg_count = train_df[train_df['years_to_cancer'] > 10].shape[0]
        train_dataset = MiraiMetadataset(train_df, resizer=resize_and_normalize, mode="training", 
                                         oversample_cancer_rate=oversample_cancer_rate, align_images=align_images, 
                                         multiple_pairs_per_exam=multiple_pairs_per_exam)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=multiple_pairs_per_exam,
                                      shuffle=True, num_workers=min(max_workers, batch_size))

    print("Have not yet entered model")
    if model == None:
        print("Loading model from scratch")
        model = LocalizedDifModel(asymmetry_metric=hybrid_asymmetry,
                    embedding_channel=512,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    embedding_model=None,
                    initial_asym_mean=initial_asym_mean,
                    initial_asym_std=initial_asym_std,
                    use_stretch=use_stretch,
                    train_backbone=train_backbone,
                    flexible_asymmetry=flexible_asymmetry,
                    use_stretch_matrix=use_stretch_matrix,
                    device_ids=device_ids,
                    use_addon_layers=use_addon_layers,
                    topk_for_heatmap=topk_for_heatmap,
                    use_bias=use_bias,
                    use_bn=use_bn)
    
    if oversample_cancer_rate is not None:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([1-neg_count/(pos_count+neg_count), 
                                                               1-pos_count/(pos_count+neg_count)]).cuda())
        
    param_list = [{'params': model.mlo_stretch_params, 'lr': lr, 'weight_decay': weight_decay},
                  {'params': model.cc_stretch_params, 'lr': lr, 'weight_decay': weight_decay}]
    if train_backbone:
        param_list = param_list + [{'params': model.backbone.parameters(), 'lr': lr / 10, 'weight_decay': weight_decay}]
    if use_addon_layers:
        param_list = param_list + [{'params': model.conv1.parameters(), 'lr': lr, 'weight_decay': weight_decay}]
    if not (topk_for_heatmap is None):
        param_list = param_list + [{'params': model.topk_weights, 'lr': lr}]
    if use_bias:
        param_list = param_list + [{'params': model.learned_asym_mean, 'lr': 1e-1}, 
                                    {'params': model.learned_asym_std, 'lr': 1e-2}]
    if model.use_bn:
        param_list = param_list + [{'params': model.bn.parameters(), 'lr': lr}]
        
    optimizer = torch.optim.Adam(param_list)#, lr=lr)
    if lr_step_size is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.2)
    
    if verbose:
        print("About to enter train loop")
    for epoch in range(num_epochs):
        if epoch in linear_only_epochs:
            print("Setting no grad")
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
        total_loss = 0
        total_loss_for_subbatch = 0
        num_samples = 0
        subbatch_samples = 0
        start = time.time()
        optimizer.zero_grad()
        
        predictions_for_epoch_pos = []
        predictions_for_epoch_neg = []
        eids_for_epoch = []
        
        for sample_ind, sample in enumerate(train_dataloader):
            if verbose:
                print(f"Loaded next sample in {time.time() - start} seconds")
            if multiple_pairs_per_exam:
                eid, label, exam_list = sample
                label = label.cuda()
                output, _ = model(None, None, None, None, exam_list=exam_list)
            else:
                eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
                label = label.cuda()
            
                output, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)
            
            with torch.no_grad():
                predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
                predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
                eids_for_epoch = eids_for_epoch + list(eid.numpy())
            
            loss = loss_func(output, label) / batch_acc
            avg_loss_vals.append(loss.item())
            loss.backward()
            if sample_ind % batch_acc == 0 and sample_ind > 0:
                
                optimizer.step()
                optimizer.zero_grad()
                if verbose:
                    print(f"loss: {total_loss_for_subbatch}", f"asym mean: {model.learned_asym_mean}", f"asym std: {model.learned_asym_std}")
                
            total_loss += loss.item()
            total_loss_for_subbatch += loss.item()
            num_samples += 1
            subbatch_samples += 1
            start = time.time()
            if sample_ind % 100 == 0:
                print(f"Saving for sample index {sample_ind}")
                print(f"Last 10 samples had average loss value {total_loss_for_subbatch/subbatch_samples}")
                print("cc_stretch_params", model.cc_stretch_params)
                print("mlo_stretch_params", model.mlo_stretch_params)
                subbatch_samples = 0
                total_loss_for_subbatch = 0
                torch.save(model, f'./training_preds/full_model_partial_epoch_{epoch}_{save_file_suffix}.pt')
            
        if lr_step_size is not None:
            print("Stepping")
            scheduler.step()
        print(f"Epoch {epoch} had average loss value {total_loss/num_samples}")
        print("cc_stretch_params", model.cc_stretch_params)
        print("mlo_stretch_params", model.mlo_stretch_params)
        
        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds.to_csv(f'./training_preds/trainn_preds_epoch_{epoch}_{save_file_suffix}.csv', index=False)
        
        correct_count = 0
        num_samples = 0
        
        predictions_for_epoch_pos = []
        predictions_for_epoch_neg = []
        eids_for_epoch = []
        
        with torch.no_grad():
            start = time.time()
            if not topk_for_heatmap is None:
                print("topk weights: ", model.topk_weights)
            for sample in val_dataloader:
                if verbose:
                    print(f"Loaded next sample in {time.time() - start} seconds")
                eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
                l_cc_img, r_cc_img, l_mlo_img, r_mlo_img = l_cc_img.cuda(), r_cc_img.cuda(), l_mlo_img.cuda(), r_mlo_img.cuda()
                label = label.cuda()

                output, _ = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)
                preds = torch.argmax(output, dim=1)
                
                predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
                predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
                eids_for_epoch = eids_for_epoch + list(eid.numpy())
                if verbose:
                    print(preds)
                
                correct_count += label[preds == label].shape[0]
                num_samples += label.shape[0]
                start = time.time()
                
        print(f"Epoch {epoch} had average val accuracy {correct_count/num_samples}")
        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds.to_csv(f'./training_preds/validation_preds_epoch_{epoch}_{save_file_suffix}.csv', index=False)
        torch.save(model, f'./training_preds/full_model_epoch_{epoch}_{save_file_suffix}.pt')
        val_accs.append(correct_count/num_samples)
    return val_accs, avg_loss_vals
        
if __name__ == '__main__':
    main(device_ids=[2],
        num_epochs=50,
        use_stretch=True,
        train_backbone=False,
        flexible_asymmetry=True,
        batch_size=40,
        lr=1,
        max_workers=20)