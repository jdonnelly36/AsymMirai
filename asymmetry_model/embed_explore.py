import pandas as pd
import numpy as np
import re
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import os
import imageio
import aiofiles
import asyncio
import re
import cv2
import logging
import urllib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from IPython.core import display
from ast import literal_eval
from pathlib import Path
import time
import pickle

from random import sample

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

tqdm.pandas()
basepath = '/data/mammo/png/'

logger = logging.getLogger(__name__)

device = torch.device(f'cuda:{0}') if torch.cuda.is_available() else torch.device('cpu')


KEY_COLUMNS_MAG = ['index', 'empi_anon', 'acc_anon', 'side', 'numfind', 'desc']
KEY_COLUMNS_MET = ['Unnamed: 0', 'empi_anon', 'acc_anon', 'ImageLateralityFinal', 'ViewPosition', 'FinalImageType', 'AcquisitionTime']
CONTEXT_COLUMNS = ['cohort_num', 'study_date_anon']

def stats(df, patient_column=None):

    if patient_column is not None:
        patient_column = df[patient_column]
    elif 'empi_anon' in df.index.names:
        patient_column = df.index.to_frame()["empi_anon"]
    else:
        patient_column = df["empi_anon"]

    stats_dict = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Unique Patients": patient_column.nunique(),
    }
    
    if 'acc_anon_scr' in df.columns:
        stats_dict["Unique Screening Exams"] = df["acc_anon_scr"].nunique()
    if 'acc_anon_diag' in df.columns:
        stats_dict["Unique Diagnostic Exams"] = df["acc_anon_diag"].nunique()
    if 'acc_anon' in df.columns:
       stats_dict["Unique Exams"] = df["acc_anon"].nunique()

    return pd.DataFrame(stats_dict, index=['counts'])

__birads_map = {
    'A': 0,
    'N': 1,
    'B': 2,   
    'P': 3,
    'S': 4,
    'M': 5,
    'K': 6,
    'X': 'X'
}
    # BIRADS 0: A – Additional evaluation
    # BIRADS 1: N – Negative
    # BIRADS 2: B - Benign
    # BIRADS 3: P – Probably benign
    # BIRADS 4: S – Suspicious
    # BIRADS 5: M- Highly suggestive of malignancy
    # BIRADS 6: K - Known biopsy proven

def translate_birads(embed_asses):
    return __birads_map[embed_asses]

def load_legend(file_path):
    # this is just a hook for better future connections
    return pd.read_csv(file_path)


    # tissueden
    # BIRADS breast density
    # 1: The breasts are almost entirely fat (BIRADS A)
    # 2: Scattered fibroglandular densities (BIRADS B)
    # 3: Heterogeneously dense (BIRADS C)
    # 4: Extremely dense (BIRADS D)
    # 5: Normal male**

    # path_severity
    # 0: invasive cancer
    # 1: non-invasive cancer
    # 2: high-risk lesion
    # 3: borderline lesion
    # 4: benign findings
    # 5: negative (normal breast tissue)
    # 6: non-breast cancer

    # from csv - massshape, massmargin, massdens, calcfind, calcdistri
    # type, path1- path10

    # spot_mag
    # 0: image is a full field digital mammogram (FFDM). All screening studies are FFDM.
    # 1: image is a special. Often used in diagnostic exams but should not occur for screening exams.

    # match_level
    # 1: indicates this ROI from the screensave was directly mapped to this image as a primary match. These ROIs are most reliable and have little to no errors.
    # 2: indicates this ROI was generated as a secondary match. For example, if a screensave had a primary match to a 2D L CC view, the secondary match will be the C-view L CC view. These ROIs are slightly less robust and can be eliminated if you are experiencing noisy data.

def lead_with_columns(df, lead_columns):
    """
    Rearranges dataframe column order by pulling a few columns to the front but keeping the order for other columns
    """
    post_columns = [col for col in df.columns if col not in lead_columns]
    return df[lead_columns + post_columns]
    
    
def load_met(file_path=os.path.join(basepath, 'metadata_all_cohort_with_ROI.csv'), 
             scope='min', extra_cols=[], cohorts=None):
    
    force_string_cols = ['CollimatorShape', '0_ProcedureCodeSequence_CodeValue', 'DerivationDescription',
                         'CommentsOnRadiationDose', 'DetectorDescription', 'WindowCenter', 'WindowWidth']
    
    categorical_cols = ['AcquisitionDeviceProcessingCode', 'AcquisitionDeviceProcessingDescription',
                        'DetectorConfiguration', 'FieldOfViewShape', 'CollimatorShape',
                        'DetectorActiveShape', 'ExposureStatus', 'VOILUTFunction'
                        '0_IconImageSequence_PhotometricInterpretation']
    
    tuple_cols = ['FieldOfViewDimensions', 'DetectorActiveDimensions', 'DetectorElementPhysicalSize',
                  'DetectorElementSpacing', 'WindowCenterWidthExplanation']
    
    date_cols = ['study_date_anon']
    
    forced_types = {col: 'string' for col in force_string_cols}
    forced_types.update({col: 'category' for col in categorical_cols})
    forced_types.update({col: 'string' for col in tuple_cols})
    
    if scope == 'min':
        usecols = (KEY_COLUMNS_MET + CONTEXT_COLUMNS + 
                    ['png_path', 'png_filename', 'num_roi', 'ROI_coords', 'match_level', 'spot_mag'] + extra_cols)
    
    elif scope == 'full':
        usecols = None
    
    else:
        return NotImplemented(f'{scope} is not a valid column scope')
    
    df_metadata = pd.read_csv(file_path, dtype=forced_types, parse_dates=date_cols, usecols=usecols)
    
    if cohorts:
        df_metadata = df_metadata.loc[df_metadata['cohort_num'].isin(cohorts)]

    return df_metadata.pipe(lead_with_columns, KEY_COLUMNS_MET + CONTEXT_COLUMNS)
    
def load_mag(file_path=os.path.join(basepath, 'magview_all_cohorts_anon.csv'), 
             scope = 'min', extra_cols=[], cohorts=None):
    # columns that have inconsistent value types - default casting doesn't work properly
    inconsistent_columns = ['case', 'biopsite', 'bcomp', 'path7', 'path8', 'path9', 'path10', 'hgrade',
                            'tnmpt', 'tnmpn', 'tnmm', 'tnmdesc', 'stage', 'bdepth','focality',
                            'specinteg', 'specembed', 'her2', 'fish', 'extracap', 'methodevl',
                            'eic', 'first_3_zip']
    type_overrides = {col: 'string' for col in inconsistent_columns}

    categories = ['massshape', 'massmargin', 'massdens', 'calcfind', 'calcdistri', 'otherfind', 
                  'implanfind', 'side', 'location', 'depth', 'distance', 'asses', 'recc', 'proccode',
                  'vtype','tissueden', 'MARITAL_STATUS_DESC']
    type_overrides.update({col: 'category' for col in categories})

    date_cols = {'study_date_anon','sdate_anon','procdate_anon','pdate_anon'}
    
    if scope == 'min':
        usecols = (KEY_COLUMNS_MAG + CONTEXT_COLUMNS + 
            ['massshape', 'massmargin', 'otherfind', 'path_group', 'path_severity'] + extra_cols)
        date_cols = date_cols.intersection(usecols)
        
    elif scope == 'full':
        usecols = None
        
    else:
        return NotImplemented(f'{scope} is not a valid column scope')
    
    df_mag =  pd.read_csv(file_path, dtype=type_overrides, header=0,
                          parse_dates=list(date_cols), usecols=usecols)
    
    df_mag['scr_or_diag'] = df_mag['desc'].apply(lambda desc: 'S' if 'screen' in desc.lower() else 'D').astype("category")
    df_mag['birads'] = df_mag['asses'].apply(translate_birads)

    if cohorts:
        df_mag = df_mag.loc[df_mag['cohort_num'].isin(cohorts)]

    return df_mag.pipe(lead_with_columns, KEY_COLUMNS_MAG + CONTEXT_COLUMNS)

def add_negative_findings():
    pass

def metadata_for(df_legend, cols):
    """
    Generate a dataframe with the metadata for a particular column in the embed dataset.
    """
    return df_legend[df_legend['Header in export']
          .isin(cols)] \
          .sort_values('Header in export')


def col_search(df, pattern, ignore_case=True):
    """
    Find columns that match a pattern
    """
    flag = re.NOFLAG if ignore_case == False else re.I
    return [c for c in df.columns if re.match(pattern, c, flags = flag)]

def img_from_row(met_row):
    """
    function to pull an image from a metadata row that is aware that images may be cached.
    """
    if 'img' in met_row and met_row['img'] is not None:
        logger.debug('retrieving cached image')
        img_array = np.frombuffer(met_row['img'], dtype=np.uint8)
        logger.debug('Got it!')
        logger.debug('using local cached image for %s %s', met_row['ViewPosition'], met_row['ImageLateralityFinal'])
        return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    else:
        logger.debug('loading image from filesystem %s', met_row['png_filename'])
        return imageio.imread(met_row['png_path'])
        

def run_asym_with_heatmap(exam_id,
                        original_df,
                        model='./asymmetry_model/training_preds/full_model_epoch_21_3_11_corrected_flex.pt',
                        device=6,
                        target_size=(1664, 2048),
                        use_crop=True,
                        pooling_size=None):
    # Loading our trained model onto the given device
    if type(model) is str:
        torch.cuda.set_device(device)
        model = torch.load(model, map_location = device)
    
    if pooling_size is not None:
        model.latent_h = pooling_size[0]
        model.latent_w = pooling_size[1]
        
    cur_exam = original_df[original_df['exam_id'] == exam_id]
    
    # Loading in each image for this exam
    imgs = []
    for view in ['CC', 'MLO']:
        for side in ['L', 'R']:
            cur_path = cur_exam[(cur_exam['view'] == view) & 
                                (cur_exam['laterality'] == side)]['file_path'].values[-1]
            imgs.append(resize_and_normalize(cv2.imread(cur_path, cv2.IMREAD_UNCHANGED), use_crop=use_crop).unsqueeze(0))

    start = time.perf_counter()
    logger.debug("Running predictions")                  
    prediction, other = model(*tuple(imgs))

    logger.info("Prediction: %s, calculated in %s", prediction, time.perf_counter() - start)
    
    res = {}
    cc_heatmap = (other[0]['heatmap'] - model.initial_asym_mean) / (2 * model.initial_asym_std)
    cc_heatmap = torch.sigmoid(cc_heatmap)
    res['cc_heatmap'] = cc_heatmap.detach().cpu()
    
    mlo_heatmap = (other[1]['heatmap'] - model.initial_asym_mean) / (2 * model.initial_asym_std)
    mlo_heatmap = torch.sigmoid(mlo_heatmap)
    res['mlo_heatmap'] = mlo_heatmap.detach().cpu()

    res['risk_score']  = prediction[0, 1]
    return res

def boxwise_upsample(array, target_size, pool_size=(5,5), mode="average"):
    """
    Upsample a tensor to target_size using overlapping
    boxes; take the max 
    array is the lower resolution activation map
    target_size is the ultimate target shape
    """
    res = torch.zeros(*target_size)
    div_matrix = torch.zeros(*target_size)
    og_h = array.shape[0]
    og_w = array.shape[1]
    
    b_h = target_size[0] / (og_h)
    b_w = target_size[1] / (og_w)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            top = int(round(i * b_h))
            bot = int(round((i + (og_h / pool_size[0])) * (b_h)))
            left = int(round(j * b_w))
            right = int(round((j + (og_w / pool_size[1])) * (b_w)))
            
            cur_chunk = res[top:bot, left:right]
            
            tmp = np.zeros_like(cur_chunk)
            tmp[:, :] = array[i, j]
            
            if mode == "average":
                # If we are averaging, we're gonna keep track of how many
                # overlapping boxes we average over with div_matrix
                div_matrix[top:bot, left:right] = div_matrix[top:bot, left:right] + 1
                res[top:bot, left:right] = res[top:bot, left:right] + torch.tensor(tmp)
            else:
                maximum = np.maximum(cur_chunk, tmp)

                res[top:bot, left:right] = maximum
    if mode == "average":
        res = res / div_matrix
    return res

def crop(img):
    nonzero_inds = torch.nonzero(img - torch.min(img))
    top = torch.min(nonzero_inds[:, 0])
    left = torch.min(nonzero_inds[:, 1])
    bottom = torch.max(nonzero_inds[:, 0])
    right = torch.max(nonzero_inds[:, 1])

    return img[top:bottom, left:right]

def resize_and_normalize(img, use_crop=False):
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)
    dummy_batch_dim = False

    # Adding a dummy batch dimension if necessary
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
        dummy_batch_dim = True

    with torch.no_grad():
        img = torch.tensor((img - img_mean)/img_std)
        if use_crop:
            img = crop(img)
        img = img.expand(1, 3, *img.shape)\
                        .type(torch.FloatTensor)
        img_resized = F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')
    #img_resized = img

    if dummy_batch_dim:
        return img_resized[0]
    else:
        return img_resized[0]

def align_images_given_img(left_data, right_data):
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(left_data,np.amin(left_data)+1e-5,np.amax(left_data),0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    ret,thresh = cv2.threshold(right_data,np.amin(right_data)+1e-5,np.amax(right_data),0)

    # calculate moments of binary image
    new_M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    new_cX = int(new_M["m10"] / new_M["m00"])
    new_cY = int(new_M["m01"] / new_M["m00"])

    num_rows, num_cols = right_data.shape[:2]   

    translation_matrix = np.float32([ [1,0,cX-new_cX], [0,1,cY-new_cY] ])   
    right_data = cv2.warpAffine(right_data, translation_matrix, (num_cols, num_rows))
    return left_data, right_data

def highlight_asym(asym_risk_df, original_data, target_exam, axs=None,
                   given_full_eid=False, align=False, flex=True, overlay_heatmap=False, 
                   model=None, use_crop=False, draw_box=True, pooling_size = (5, 5),
                  roi_info=None, model_input_size=(1664, 2048), latent_size=(52, 64),
                  asymetry_distribution=None):
    latent_h, latent_w = latent_size[0], latent_size[1]
    pool_h = latent_h // pooling_size[0]
    pool_w = latent_w // pooling_size[1]
    
    if given_full_eid:
        eid = target_exam
        #print(asym_risk_df[asym_risk_df['exam_id'] == eid]['asymmetries'])
    else:
        eid = asym_risk_df[(asym_risk_df['years_to_cancer'] < 100) & 
                           (asym_risk_df['years_to_cancer'] >= 0)]['exam_id'].values[target_exam]
        
    # Creating the layout for our plot
    if axs is not None:
        logger.debug('Appending to provided axis')
        ax = axs
        fig = None
    else:
        logger.debug('Generating heatmap in new figure')
        fig, ax = plt.subplots(2,2, figsize=(8,10))
        [axi.set_axis_off() for axi in ax.ravel()]

    if hasattr(model, "topk_for_heatmap") and model.topk_for_heatmap is not None:
        topk_count = model.topk_for_heatmap
    else:
        topk_count = 1
    # If we are adding heatmaps, compute them
    if model is not None:
        asym_results = run_asym_with_heatmap(target_exam, original_data, target_size=model_input_size, 
                                        model=model, use_crop=use_crop, pooling_size=pooling_size)
    else:
        asym_results = run_asym_with_heatmap(target_exam, original_data, 
                                        target_size=model_input_size, use_crop=use_crop, pooling_size=pooling_size)
    
    # Iterate over the two standard views
    logger.debug('Rendering Heatmap for %s', eid)
    start = time.perf_counter()
    for i, view in enumerate(['MLO', 'CC']):
        for j, side in enumerate(['L', 'R']):
            # Grab the indices we'll use to draw the max box
            logger.debug('Heatmap view for %s-%s',view,side)

            ax[i,j].set_title(side + ' ' + view)
            
            if view == 'MLO':
                #max_by_ftr, x_argmin = torch.max(asym_results[f'mlo_heatmap'], dim=-1)
                max_by_ftr, argmax_inds = torch.topk(asym_results[f'mlo_heatmap'].view(-1), topk_count, dim=-1)
                x_argmax = argmax_inds % asym_results[f'mlo_heatmap'].shape[-1]
                y_argmax = argmax_inds // asym_results[f'mlo_heatmap'].shape[-1]
                logger.debug('mlo max: %d, %d', x_argmax, y_argmax)
            else:
                max_by_ftr, argmax_inds = torch.topk(asym_results[f'cc_heatmap'].view(-1), topk_count, dim=-1)
                x_argmax = argmax_inds % asym_results[f'cc_heatmap'].shape[-1]
                y_argmax = argmax_inds // asym_results[f'cc_heatmap'].shape[-1]
                logger.debug('cc max: %d, %d', x_argmax, y_argmax)

            asym_score = asym_results['risk_score']
            
            if asymetry_distribution is None:
                ax[i,j].text(x=0, y=-.06, s=f"Asym Score [0-1]: {round(asym_score.item(),3)}", transform=ax[i,j].transAxes)
            else:
                quantile = asymetry_distribution.quantile(asym_score.item())
                ax[i,j].text(x=0, y=-.06, s=f"Asym Quantile: {round(quantile,3)} (Score of {round(asym_score.item(),3)}", transform=ax[i,j].transAxes)
            #max_by_ftr, y_argmin = torch.max(max_by_ftr, dim=-1)
            #for topk_ind in range(topk_count):
            #    y_max_by_ftr, y_argmin = torch.topk(max_by_ftr.view(-1), topk_count, dim=-1)
            
            # Convert these values to fractional locations in [0, 1]
            if view == 'MLO':
                tuple_list = []
                for topk_ind in range(topk_count):
                    y_loc_frac = y_argmax[topk_ind].item() / asym_results[f'mlo_heatmap'].shape[-2]
                    x_loc_frac = x_argmax[topk_ind].item() / asym_results[f'mlo_heatmap'].shape[-1]
                    logger.debug(f"True max (MLO) ---- {torch.max(asym_results[f'mlo_heatmap'])}", )
                    logger.debug(f"Max found using indices ---- {asym_results[f'mlo_heatmap'][0, y_argmax[topk_ind].item(), x_argmax[topk_ind].item()]}", )
                    tuple_list.append((x_loc_frac, y_loc_frac))
                    #y_loc_frac = y_argmin / asym_results[f'mlo_heatmap'].shape[-2]
                    #x_loc_frac = x_argmin[0, y_argmin].item() / asym_results[f'mlo_heatmap'].shape[-1]
            else:
                tuple_list = []
                for topk_ind in range(topk_count):
                    y_loc_frac = y_argmax[topk_ind].item() / asym_results[f'cc_heatmap'].shape[-2]
                    x_loc_frac = x_argmax[topk_ind].item() / asym_results[f'cc_heatmap'].shape[-1]
                    logger.debug(f"True max (CC) ---- {torch.max(asym_results[f'cc_heatmap'])}")
                    logger.debug(f"Max found using indices ---- {asym_results[f'cc_heatmap'][0, y_argmax[topk_ind].item(), x_argmax[topk_ind].item()]}", )
                    tuple_list.append((x_loc_frac, y_loc_frac))
            
            # Grab the file path for each side for the current view
            img = original_data[(original_data['exam_id'] == eid) 
                                & (original_data['view'] == view)
                                & (original_data['laterality'] == side)]['file_path'].values[-1]

            # Read in our image, crop and align if needed
            img_data = imageio.imread(img)
            if use_crop:
                img_data = crop(torch.tensor(img_data / 1)).numpy()
            if align and side == 'R':
                l_img = original_data[(original_data['exam_id'] == eid) 
                                & (original_data['view'] == view)
                                & (original_data['laterality'] == 'L')]['file_path'].values[-1]
                l_img_data = imageio.imread(l_img)
                if use_crop:
                    l_img_data = crop(torch.tensor(l_img_data / 1)).numpy()
                _, img_data = align_images_given_img(l_img_data, img_data)
        
            # Overlay our heatmap onto the image and display it if desired;
            # otherwise just display the image
            if overlay_heatmap:
                if view == 'MLO':
                    heatmap = asym_results[f'mlo_heatmap'].numpy()[0]
                else:
                    heatmap = asym_results[f'cc_heatmap'].numpy()[0]
                overlayed_img = overlay_heatmap_on_image(img_data, heatmap, pooling_size=pooling_size)
                img_with_heatmap = ax[i, j].imshow(overlayed_img, interpolation='nearest', )
                plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.get_cmap("jet")), ax=ax[i,j], pad=.01)
            else:
                img_data_rescaled = img_data - np.amin(img_data)
                img_data_rescaled = img_data_rescaled / np.amax(img_data_rescaled)
                img_data_rgb = np.repeat(img_data_rescaled[:, :, np.newaxis], 3, axis=2)
                
                ax[i, j].imshow(img_data_rgb, cmap='gray', interpolation='nearest', )
            
            height = img_data.shape[0]
            width = img_data.shape[1]
            
            for topk_ind in range(topk_count):
                x_loc_frac = tuple_list[topk_ind][0]
                y_loc_frac = tuple_list[topk_ind][1]
                weight = model.topk_weights[-topk_ind].item()
                if view == 'MLO':
                    heatmap = asym_results[f'mlo_heatmap'].numpy()[0]
                else:
                    heatmap = asym_results[f'cc_heatmap'].numpy()[0]
                y_loc = int(round(y_loc_frac * height))
                x_loc = int(round(x_loc_frac * width))

                rect_width = int(width / pooling_size[1])
                rect_height = int(height / pooling_size[0])

                # Add the region of asymmetry actually used by our model
                if draw_box and flex:
                    rect = Rectangle((x_loc, y_loc), 
                                            rect_width, rect_height,
                                            linewidth=1, edgecolor=(1,0,0,weight), facecolor='none')
                    # Add the patch to the Axes
                    ax[i, j].add_patch(rect)
                elif draw_box:
                    rect = Rectangle((x_loc * (width / latent_w), y_loc * (height / latent_h)), 
                                            pool_w * (width // latent_w), pool_h * (height // latent_h), 
                                            linewidth=1, edgecolor=(1,0,0,weight), facecolor='none')
                    # Add the patch to the Axes
                    ax[i, j].add_patch(rect)

                if roi_info is not None and roi_info[4] == side and roi_info[5] == view:
                    tl_y, tl_x, br_y, br_x = roi_info[0], roi_info[1], roi_info[2], roi_info[3]
                    rect = Rectangle((tl_x, tl_y), 
                                    br_x - tl_x, br_y - tl_y, 
                                    linewidth=1, edgecolor='g', facecolor='none')
                    # Add the patch to the Axes
                    ax[i, j].add_patch(rect)

    logger.debug('Heatmap rendered for %s in %s', eid, time.perf_counter()-start)

    if fig is not None:
        fig.show()
        fig.savefig(f'./visualization_{target_exam}.png', dpi=300)
        return f'./visualization_{target_exam}.png'        

def mammo_img_to_ax(ax, met_row, normalize=False):
    """
    Given a screening, add the image to an axis with the ROI's highlighted in bounding boxes.
    
    row represents a metadata row with screening exams
    """
    
    img_data = img_from_row(met_row)
    
    height = img_data.shape[0]
    width = img_data.shape[1]

    logger.debug('rendering image %s', met_row['png_filename'])
    # This step is handing frequently
    if normalize:
        img_data_rescaled = img_data - np.amin(img_data)
        img_data_rescaled = img_data_rescaled / np.amax(img_data_rescaled)
        img_data_rgb = np.repeat(img_data_rescaled[:, :, np.newaxis], 3, axis=2)
    else:
        img_data_rgb = img_data
    
    img = ax.imshow(img_data_rgb, cmap='gray', interpolation='bilinear')

    title = met_row['ImageLateralityFinal'] + ' ' + met_row['ViewPosition']

    ax.set_title(title)

    if met_row['num_roi'] > 0:
        
        match_levels = literal_eval(met_row['match_level'])
        ax.text(x=0, y=-.1, s=f"ROI Quality (1 > 2): {max(match_levels)}", transform=ax.transAxes)
     

        for t in literal_eval(met_row['ROI_coords']):
            y_dist = t[2] - t[0]
            x_dist = t[3] - t[1]

            rect = patches.Rectangle((t[1], t[0]), y_dist, x_dist, 
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    logger.debug('image complete %s', met_row['png_filename'])

def text_axis(ax, lines, title=None, horizontalalignment='left', verticalalignment='center'):
    
    x = 0 if horizontalalignment == 'left' else .5
    y = 1 if verticalalignment == 'top' else .5
    
    ax.set_title(title)
    ax.text(x=x, y=y, s="\n".join(lines), color='black', fontsize=12, wrap=True,
        horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, transform=ax.transAxes)

    # Remove the axis labels and tick marks
    ax.set_axis_off()


def add_patient_details_to_ax(ax, df_magXmet_exam):

    def val_of(col):
        return df_magXmet_exam[col].iloc[0]
    patient_notes = [
        f"Patient Id: {val_of('empi_anon')}",
       ]
    text_axis(ax, patient_notes, title='Patient History', verticalalignment='top')


def add_exam_details_to_ax(ax, typ, df_magXmetXmiraiinput):
    if df_magXmetXmiraiinput.index.name == 'acc_anon':
        acc_anon = df_magXmetXmiraiinput.index[0]
    else:
        acc_anon = df_magXmetXmiraiinput['acc_anon'].iloc[0]

    def val_of(col):
        return df_magXmetXmiraiinput[col].iloc[0]
    
    exam_notes = [
         f"Exam Id: {acc_anon}",
         f"Date: {val_of('study_date_anon_mag')}",
         f"Patient Age: {val_of('age_at_study')}",
         f"Years to Cancer: {val_of('years_to_cancer')}",
         f"BIRADS: {val_of('birads')}",
         f"SpotMag: {val_of('spot_mag')}",
         f"Calc: {val_of('calcfind')}",
         f"Mass Shape: {val_of('massshape')}, Margin: {val_of('massmargin')}",
        #  f"Other Find: {val_of('other')}",
         f"Pathology Severity: {val_of('path_severity')}, Location: {val_of('path_loc')}",
    ]

    if '1year_risk' in df_magXmetXmiraiinput.columns:
        if not df_magXmetXmiraiinput['1year_risk'].isna().iloc[0]:
            cols = [f"{i}year_risk" for i in range(1,6)]
            mirai_risks = [round(df_magXmetXmiraiinput[col].iloc[0], 3) for col in cols]

            print_risks = [f"{i}Y:{risk}" for i, risk in enumerate(mirai_risks)]
            exam_notes.extend([
                f"MIRAI Risks:",
                "  " + " ".join(print_risks[:3]),
                "    " + " ".join(print_risks[3:6])
            ])
    
    text_axis(ax, exam_notes, title=typ + ' Summary', verticalalignment='top')

def plot_heatmap():
    pass

def plot_screening(df_magXmetXmiraiinput_oneexam, ax, normalize=False):
    """
    Generate the 4 image view for a screening exam.
    axs is a 2x3 grid of axes. This grid may need to be composed before calling
    """
    
    df_this_exam = df_magXmetXmiraiinput_oneexam

    if len(df_this_exam.groupby(['acc_anon']).count().index) == 0:
        raise ValueError('No accessions in the screening df')
    if len(df_this_exam.groupby(['acc_anon']).count().index) > 1:
        raise ValueError('multiple accessions in the screen df; can only plot 1')

    add_patient_details_to_ax(ax[0,0], df_this_exam)
    add_exam_details_to_ax(ax[1,0], "Screening", df_this_exam)

    laterality_views = []
    for i, view in enumerate(['MLO', 'CC']):
        for j, side in enumerate(['L', 'R']):
            view_side_selection = df_this_exam[(df_this_exam['ViewPosition'] == view) 
                              & (df_this_exam['ImageLateralityFinal'] == side)
                              & (df_this_exam['FinalImageType'] == '2D')]
            
            axij = ax[i,j+1]
            if len(view_side_selection) < 1:
                logger.debug('No image for %s %s. Skipping', side, view)
                text_axis(axij, ['No screening image for',f'{side} {view}'], horizontalalignment='center')
                continue

            laterality_views.append((side, view))
            mammo_img_to_ax(axij, view_side_selection.iloc[0], normalize=normalize)

    return laterality_views


def plot_diagnostic(df_magXmet_diag, axs):

    ax_sum = axs[0]

    logger.debug('Creating Diagnostic Exam ')
    add_exam_details_to_ax(ax_sum, "Diagnostic", df_magXmet_diag)

    for i, (idx, row) in enumerate(df_magXmet_diag.iterrows()):
        mammo_img_to_ax(axs[1+i], row)


def select__goldscr(def_met, image_type='2D'):

    def_met_goldscreen = def_met[(def_met['ViewPosition'].isin(['MLO', 'CC'])) 
                              & (def_met['ImageLateralityFinal'].isin(['L', 'R']))
                              & (def_met['FinalImageType'] == image_type)
                              & (def_met['spot_mag'].isna())
                              & (def_met['scr_or_diag'] == 'S')]
    
    if def_met.index.name == 'acc_anon':
        def_met['acc_anon'] = def_met.index

    def_met_goldscreen = def_met_goldscreen.sort_values(['AcquisitionTime'], ascending=False)
    def_met_goldscreen = def_met_goldscreen.drop_duplicates(['ViewPosition', 'ImageLateralityFinal'])

    return def_met_goldscreen


def plot_exam(df_magXmetXmiraiinput_exam, heatmap_model=None, include_all_images=False, asymetry_distribution=None):

    do_heatmap = True if heatmap_model else False
    logger.debug('Plotting exam from %s. heatmapping is turned %s.', df_magXmetXmiraiinput_exam['study_year'].iloc[0], 'ON' if do_heatmap else 'OFF')

    # Select the data to render
    df_goldscr = df_magXmetXmiraiinput_exam.pipe(select__goldscr)
    df_scr = df_magXmetXmiraiinput_exam.loc[df_magXmetXmiraiinput_exam['scr_or_diag'] == 'S']
    df_diag = df_magXmetXmiraiinput_exam.loc[df_magXmetXmiraiinput_exam['scr_or_diag'] == 'D']

    if include_all_images:
        df_scr_for_comp = df_scr.set_index('png_path', append=True)
        df_goldscr_for_comp = df_goldscr.set_index('png_path', append=True)

        nongoldscr_idx = df_scr_for_comp.index.difference(df_goldscr_for_comp.index)
        if len(nongoldscr_idx) > 0:
            df_nongoldscr = df_scr_for_comp.loc[nongoldscr_idx]
            logger.debug('non-gold scope: %s-%s=%s', len(df_scr), len(df_goldscr), len(df_nongoldscr))
            df_extra_exams = pd.concat((df_nongoldscr.reset_index(level='png_path'), df_diag))
        else:
            df_extra_exams = df_diag
    else:
        df_extra_exams = df_diag

    # Construct the layout based on the available data
    num_screens = int(len(df_goldscr) > 0)
    num_extra = len(df_extra_exams)

    num_extra = 0 if num_extra == 0 else num_extra + 1 # info column
    logger.debug('%s extra views', num_extra)

    ncols = 5 if do_heatmap else 3
    height = 8 * num_screens + 4 * math.ceil(num_extra / ncols)
    extra_rows =  math.ceil(num_extra/ncols)

    one_screen_rows = 2
    screen_rows = one_screen_rows * num_screens

    total_rows = screen_rows + extra_rows

    if total_rows == 0:
        logger.error('Exam has no images was still present in the datasets. Skipping')
        return

    logger.debug('final layout is %sx%s', total_rows, ncols)

    # Render the figure
    fig, axs = plt.subplots(nrows=total_rows, ncols=ncols, figsize=(4*ncols, height))
    [axi.set_axis_off() for axi in axs.ravel()]

    if len(np.shape(axs))==1:
         axs = np.expand_dims(axs, 0)

    year = df_magXmetXmiraiinput_exam['study_year'].iloc[0]
    patient_id =  df_magXmetXmiraiinput_exam['empi_anon'].iloc[0]
    exam = df_magXmetXmiraiinput_exam.index[-1]
    fig.suptitle(f'{year} - Patient #{patient_id} - Exam #{exam}')

    logger.debug('there are %s screening images for patient:%s exam:%s', len(df_goldscr), patient_id, exam)

    if len(df_goldscr) > 0:
        
        scr_row_lower = 0
        scr_row_upper = one_screen_rows
        screen_axs = axs[scr_row_lower:scr_row_upper, :3]

        logger.debug('plotting %s screening images', len(df_goldscr))
        laterality_views = plot_screening(df_goldscr, screen_axs)
        logger.debug('plotted %s images: %s', len(laterality_views), laterality_views)
        
        if do_heatmap:
            
            heatmap_axs = axs[scr_row_lower:scr_row_upper, ncols-2:ncols]

            if len(laterality_views) == 4:
                logger.debug('building heatmap for %s - %s', patient_id, year)
                
                highlight_asym(df_goldscr, df_goldscr, df_goldscr['exam_id'].values[-1],
                               model=heatmap_model, axs=heatmap_axs, given_full_eid=True, 
                               overlay_heatmap=True, use_crop=False, 
                               pooling_size=(heatmap_model.latent_h, heatmap_model.latent_w),
                               asymetry_distribution=asymetry_distribution)
            else:
                logger.debug('skipping heatmap for %s - %s - because there are only %s screening images', patient_id, year, len(laterality_views))
                for ax in heatmap_axs.flat:
                    text_axis(ax, ['Heatmap Not Available', '(missing screening views)'])

    logger.debug('there are %s extra images for patient:%s exam:%s', len(df_extra_exams), patient_id, exam)    
    if extra_rows > 0:
        extra_axs = axs[screen_rows:, :]
        plot_diagnostic(df_extra_exams, extra_axs.flat)

    fig.show()


def get_diagnostic_for_patient(df_met, full_df):
    """
    Given a dataframe, gets the dataframe
    containing the diagnostic image (if any)
    for the given patient
    """
    print(df_met.columns)
    patient_id = df_met['empi_anon_x'].values[0]
    diag_df = full_df[(full_df['empi_anon_x'] == patient_id)
               & (full_df['desc'].str.contains('diag', case=False))]
    return diag_df

def get_roi_for_exam(exam_id, df_with_rois):
    rows_of_interest = df_with_rois[(df_with_rois['acc_anon'] == exam_id)
                        & (df_with_rois['ROI_coords'] != '()')]
    print(rows_of_interest.shape)
    m_tuple = (*rows_of_interest['roi_tuples'].values[-1][0], 
               rows_of_interest['ImageLateralityFinal'].values[-1],
               rows_of_interest['ViewPosition'].values[-1])
    print(m_tuple)
    return m_tuple
    
async def load_img_async(path):
    """
    Load images asynchronously. The file load is moved to another thread.
    @returns cv2 image from the path
    """
    logger.debug('loading file at %s', path)
    async with aiofiles.open(path, mode='rb') as file:
        file_contents = await file.read()
    
    logger.debug('file at %s loaded', path)
    return file_contents


async def load_imgs_async(paths):
    """
    Load a batch of images using asyncio in parallel.
    """
    coroutines = [load_img_async(path) for path in paths]
    return await asyncio.gather(*coroutines)


def img_display_raw(df_met__row, filetype="png"):
    """
    Displays an image from a metadata row as a full scale image.
    
    If the image was previously cached in the row in the 'img' column, that is used.
    Otherwise, it's loaded from png_path.
    """
    
    if filetype != 'png':
        raise Exception('only png supported today')
    
    if 'img' in df_met__row and not df_met__row['img'] is not None:
        return display.Image(data=df_met__row['img'], format='png')
    else:
        return display.Image(df_met__row.png_path)
   
    
def link_to(url, title=None):
    """
    Returns an HTML anchor tag that links to the given URL and opens it in a new tab.
    Args:
        url (str): The URL to link to.
        title (str, optional): The title to display for the link. If None, the URL is used as the title.
                               Default is None.
    Returns:
        display.HTML: An HTML object containing the anchor tag.
    """
    
    return display.HTML(f'<a target="_blank" href="{url}">{url if title is None else title}</a>')


def img_urls_from_met(df_met):
    """
    Given a metadata dataframe, get all the URLs to the PNG images.
    """
    return [file_url(png_path) for png_path in df_met['png_path']]

def overlay_heatmap_on_image(img, heatmap, pooling_size=(5,5)):
    height, width = img.shape[0], img.shape[1]
    upsampled_heatmap = boxwise_upsample(heatmap, (height, width), pool_size=pooling_size)
    
    final_heatmap = cv2.applyColorMap(np.uint8(255*upsampled_heatmap), cv2.COLORMAP_JET)
    final_heatmap = np.float32(final_heatmap) / 255
    final_heatmap = final_heatmap[...,::-1]
    
    img_rescaled = img - np.amin(img)
    img_rescaled = img_rescaled / np.amax(img_rescaled)
    img_data_rgb = np.repeat(img_rescaled[:, :, np.newaxis], 3, axis=2)
    
    overlayed_img = 0.5 * img_data_rgb + 0.3 * final_heatmap
    return overlayed_img

def pathological_diags(df_mag_scr2diagXmet, empi_anon):
    """
    For a patient, find the diagnostics that represent their eventual cancer outcomes
    """
    return NotImplemented


def patient_profile(df_mag_scr2diagXrisk, empi_anon):
    """
    Summarize the patient's medical history, including MIRAI's risk score. Creates a new dataframe.
    
    Dataframe indexed by year.
    """
    return NotImplemented


def diagnostics_for_screen(df_mag_scr2diag, acc_anon):
    """
    Given a screening exam, get the follow-up diagnostic exams for that screening
    """
    return NotImplemented


def viz_diagnostics_with_screen(df_mag_scr2diagXmet, acc_anon):
    """
    Given a screening exam that had follow-up diagnostics, visualize the two screenings with diagnostic exams laid out the right. Only the side with a finding from the screening is included.
    
    Top Row: MLO, CC from the side with the finding, One box of Summary, ROIs
    Rows of Three with Labeled Views
    """
    return NotImplemented


def summarize_screen(df_magXmet, acc_anon):
    """
    Given a screening exam, return the facts about the outcome of the screening.
    """
    return NotImplemented


def summarize_diag():
    """
    Given a screening exam, return the facts about the outcome of the screening.
    """
    return NotImplemented

def select__df_magXmet_final_imgs(df_magXmet):
    df_magXmet = df_magXmet.sort_values(['study_date_anon_mag', 'AcquisitionTime'], ascending=False)
    df_magXmet = df_magXmet.drop_duplicates(['acc_anon', 'side', 'ViewPosition'])
    df_magXmet = df_magXmet.set_index(['empi_anon', 'acc_anon', 'side', 'ViewPosition']).pipe(lead_with_columns, ['study_date_anon_mag', 'num_roi', 'ROI_coords'])
    return df_magXmet.sort_index()

def as__df_mag_scr2diag(df_mag, months_forward=6):
    """
    Merge screenings with their follow-up diagnostics
    
    standard return name is 
    """
    
    df_mag_scr = df_mag.loc[df_mag['scr_or_diag'] == 'S']
    df_mag_diag = df_mag.loc[df_mag['scr_or_diag'] == 'D']

    df_full_scr_diag_match = df_mag_scr.merge(df_mag_diag, 
                                      left_on=['empi_anon', 'side', 'numfind'], 
                                      right_on=['empi_anon', 'side', 'numfind'], 
                                      suffixes=('_scr', '_diag')).drop_duplicates()
    
    def filter_by_month_diff(row):
        """
        finds all diagnostics that are within x months of a screening for a row
        """
        start_date = row['study_date_anon_scr'] 
        end_date = row['study_date_anon_scr'] + pd.DateOffset(months=months_forward)
        return (row['study_date_anon_diag'] >= start_date) & (row['study_date_anon_diag'] <= end_date)

    df_full_scr_diag_match[f'screen_{months_forward}_before_diag'] = df_full_scr_diag_match.apply(filter_by_month_diff, axis=1)
    
    df_full_scr_diag_match = df_full_scr_diag_match.loc[df_full_scr_diag_match[f'screen_{months_forward}_before_diag']] \
        .drop_duplicates()

    return df_full_scr_diag_match

def to__df_mag_patient_hist(df_mag):

    this_df_mag = df_mag.copy()
    this_df_mag['study_year'] = this_df_mag['study_date_anon'].dt.year

    this_df_mag = this_df_mag.set_index(['empi_anon', 'study_year', 'scr_or_diag', 'acc_anon'])
    return this_df_mag.sort_index(ascending=[True, False, True, False]).pipe(lead_with_columns, ['study_date_anon', 'side', 'numfind'])

def shorthand_to_text(df, columns=None):

    if columns is None:
        return df
    
def sample_on_conditions():
    
    return NotImplemented
    
    def sample_up_to_5(group):
        sample_size = np.minimum(len(group), 5)
        return group.sample(n=sample_size)
    
    df_mismatched_path_sev_sample = df_mismatched_path_sev_groups[['acc_anon_scr', 'acc_anon_diag']].apply(sample_up_to_5)

    index_of_mismatch_samples = df_mismatched_path_sev_sample.index.get_level_values(2)

    df_mismatched_path_sev_sample = df_full_scr_diag_match.iloc[index_of_mismatch_samples].pipe(lead_with_columns, ['path_severity_diag', 'path_severity_scr'])

    df_mismatched_path_sev_sample.groupby(['path_severity_scr', 'path_severity_diag']).count()[['empi_anon']]

def pdf_report_of_figures(pdf_doc_name, figure_hooks, title="Report"):
    """
    Generate a PDF report of figures from a list of hooks.
    Args:
        pdf_doc_name (str): Name of the PDF file to create.
        figure_hooks (list): List of functions or callable objects that create
            figures to include in the report. Each hook should take a single argument
            - a Matplotlib pyplot object - and use it to create a figure.
            If a tuple of size 2 is provided, the first element is used to set the
            size of the figure.
        title (str, optional): Title to include in the PDF metadata. Defaults to "Report".
    Example usage:
        >>> def my_figure_hook(plt):
        ...     plt.plot([1, 2, 3], [4, 5, 6])
        ...
        >>> pdf_report_of_figures("my_report.pdf", [my_figure_hook], "My Report")
    """
    original_rc_params = mpl.rcParams
    
    try:
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['pgf.preamble'] = [r'\usepackage{hyperref} \hypersetup{hidelinks,' 
                                'colorlinks=true, urlcolor=cyan}', ]

        
        with PdfPages(pdf_doc_name) as pdf:
            
            for hook in figure_hooks:
                
                if hasattr(hook, '__iter__') and len(hook) == 2:
                    size = hook[0]
                    hook = hook[1]
                else:
                    size = (8, 11)
            
                plt.figure(figsize=size)
                hook(plt)
                pdf.savefig()
                plt.close()

            d = pdf.infodict()
            d['Title'] = title
    finally:
        mpl.rcParams.update(original_rc_params)

def merge__df_magXmet(df_mag, df_met, image_type=None):

    if image_type is not None:
        df_met = df_met.loc[df_met['FinalImageType']==image_type]
    
    left_on = ['empi_anon', 'acc_anon']
    right_on = ['empi_anon', 'acc_anon']
    df_magXmet_full = df_mag.merge(df_met,
                left_on=left_on,
                right_on=right_on,
                suffixes=('_mag', '_met'))
    
    matching_side_filter = ((df_magXmet_full['side'].isna()) | 
                            (df_magXmet_full['side'] == 'B') | 
                            (df_magXmet_full['side'] == df_magXmet_full['ImageLateralityFinal']))
    return df_magXmet_full.loc[matching_side_filter]

def df_magXmet_for_patients(empi_anons, df_mag=None, df_met=None, df_magXmet=None):  
    if df_magXmet is not None:
        return df_magXmet.loc[df_magXmet['empi_anon'].isin(empi_anons)]
    elif df_mag is not None and df_met is not None:

        df_mag_patients = df_mag.loc[df_mag['empi_anon'].isin(empi_anons)]
        df_met_patients = df_met.loc[df_met['empi_anon'].isin(empi_anons)]
        return merge__df_magXmet(df_mag_patients, df_met_patients)
    else:
        raise ValueError('must supply either df_mag and df_met or df_magXmet')
    

def patient_report(empi_anon, pdf_path=None, with_heatmap=False, df_mag=None, df_met=None, df_magXmet=None, asym_model=None, include_all_images=False, asymetry_distribution=None):
    df_magXmet_patient = df_magXmet_for_patients([empi_anon], df_mag, df_met, df_magXmet)

    if asym_model is not None:
        logger.debug('using provided asym model')
        with_heatmap = True
    elif with_heatmap:
        path = 'REDACTED/PATH/TO/MODEL'
        logger.debug('loading asym model from %s', path)
        asym_model = torch.load(path, map_location = device)
        logger.debug('asym model loaded')
    else:
        asym_model = None

    if include_all_images:
        logger.debug('All images will be included in the report')

    pdf_path = pdf_path if pdf_path is not None else f'{empi_anon}.pdf'

    def screen_hook(df_magXmet_exam_year):
        def screen_with_asym(plt):
            if with_heatmap:
                plot_exam(df_magXmet_exam_year, heatmap_model=asym_model, include_all_images=include_all_images,
                         asymetry_distribution=asymetry_distribution)
            else:
                plot_exam(df_magXmet_exam_year, include_all_images=include_all_images,
                          asymetry_distribution=asymetry_distribution)

        return screen_with_asym

    df_magXmet_patient['study_year'] = df_magXmet_patient['study_date_anon_mag'].dt.year
    df_magXmet_patient = df_magXmet_patient.set_index('acc_anon')
    
    hooks = []            
    df_magXmet_patient = df_magXmet_patient.sort_values('study_year', ascending=True)   
    print("sorting again")
    for _, df in df_magXmet_patient.groupby(['study_year', 'acc_anon'], sort=False):
        # df['acc_anon'] = df.index
        hooks.append(screen_hook(df))

    pdf_report_of_figures(pdf_path, hooks)

    return pdf_path

class EmbedDebugContext:
    def __init__(self):
        self.original_level = logger.getEffectiveLevel()

    def __enter__(self):
        logger.setLevel(logging.DEBUG)

    def __exit__(self, exc_type, exc_value, traceback):
        logger.setLevel(self.original_level)

debug_logging = EmbedDebugContext()

def patients_with_cancer_and_prev_screenings(df_magXmetXmiraiinput, return_groups=False):
    df_magXmetXmiraiinput_sel_y2c_lt_100 = df_magXmetXmiraiinput.loc[df_magXmetXmiraiinput['years_to_cancer'] < 100]
    cancer_patients = df_magXmetXmiraiinput_sel_y2c_lt_100['empi_anon'].unique()

    df_magXmetXmiraiinput_cp = df_magXmetXmiraiinput.loc[df_magXmetXmiraiinput['empi_anon'].isin(cancer_patients)]

    df_magXmetXmiraiinput_cp_scr = df_magXmetXmiraiinput_cp.loc[(df_magXmetXmiraiinput_cp['years_to_cancer'] > 1) 
                                                    & (df_magXmetXmiraiinput_cp['scr_or_diag'] == 'S') 
                                                    & (df_magXmetXmiraiinput_cp['ViewPosition'].isin(['MLO', 'CC']))
                                                    & (df_magXmetXmiraiinput_cp['ImageLateralityFinal'].isin(['L', 'R'])
                                                    & (~df_magXmetXmiraiinput_cp['1year_risk'].isna()))]

    df_4view_groups = df_magXmetXmiraiinput_cp_scr.drop_duplicates(['acc_anon', 'ImageLateralityFinal', 'ViewPosition']).groupby(['empi_anon', 'acc_anon']).count()[['ImageLateralityFinal']]
    df_4view_groups = df_4view_groups.loc[df_4view_groups['ImageLateralityFinal'] == 4]
    
    unique_empi = df_4view_groups.index.get_level_values('empi_anon').to_series().unique()
    if return_groups:
        return unique_empi, df_4view_groups
    else:
        return unique_empi