import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from itertools import chain
import asyncio
import aiofiles
from embed_explore import align_images_given_img

exam_record = namedtuple('exam_foo', 
    ['eid', 'label', 'l_cc_img', 'l_cc_path', 'r_cc_img', 'r_cc_path', 'l_mlo_img', 'l_mlo_path', 'r_mlo_img', 'r_mlo_path'])

class MiraiMetadataset(Dataset):
    """
    Creates a torch dataset out of a MIRAI metadata csv.
    Rows are exam_record named tuples.
    """
    def __init__(self, metadata_frame, allow_incomplete=False, verbose=False, resizer=None, mode="training", load_images_async=True, align_images=True, oversample_cancer_rate=None, multiple_pairs_per_exam=False):
        """
        
        only_complete - both views, both lateralities
        """
        super().__init__()
        assert not (align_images and multiple_pairs_per_exam), "Error: Alignment not currently supported with multiple pairs per exam"
        self.exams = []
        self.resizer = resizer
        self.load_images_async = load_images_async
        self.align_images = align_images
        self.mode = mode
        # Whether we want to average over multiple image pairings for each view
        self.multiple_pairs_per_exam = multiple_pairs_per_exam 
        print("Using oversample_cancer_rate of", oversample_cancer_rate)
        
        # This is necessary to make batching play nice. If we allow different exams to have different
        # numbers of image, we will get jagged input tensors. This is a workaround where we set a max
        # number of images, and pad exams with less than the max number of images with null images.
        # The null images are explicitly handled in the forward pass of the model when present.
        self.max_per_view = 4

        for i, eid in enumerate(metadata_frame['exam_id'].unique()):
            
            cur_exam = metadata_frame[metadata_frame['exam_id'].values == eid]

            if not self.multiple_pairs_per_exam:
                patient_exam = {'MLO': {'L': None, 'R': None},
                                'CC': {'L': None, 'R': None}}
            else:
                patient_exam = {'MLO': [],
                                'CC': []}
            complete_exam = True
            for view in patient_exam.keys():
                def indices_for_side_view(side):
                    indices = np.logical_and(cur_exam['view'].values == view, cur_exam['laterality'].values == side)
                    return indices
                
                if len(cur_exam[indices_for_side_view('L')]['file_path'].values) == 0:
                    if verbose:
                        print("Missing {} {} view for exam {}".format(view, laterality, eid))
                    complete_exam = False
                    continue
                    
                elif not self.multiple_pairs_per_exam:
                    for laterality in ['L', 'R']: 
                        if len(cur_exam[indices_for_side_view(laterality)]['file_path'].values) == 0:
                            if verbose:
                                print("Missing {} {} view for exam {}".format(view, laterality, eid))
                            complete_exam = False
                            continue
                        old_path = cur_exam[indices_for_side_view(laterality)]['file_path'].values[-1]
                        patient_exam[view][laterality] = old_path
                        
                else:
                    for l_path in cur_exam[indices_for_side_view('L')]['file_path'].values:
                        cur_row = cur_exam[cur_exam['file_path'] == l_path]
                        # This line is checking for nan
                        if type(cur_row['matched_image']) is str:
                            patient_exam[view].append({'L': l_path, 'R': cur_row['matched_image'].values[0]})
                        else:
                            r_imgs = cur_exam[indices_for_side_view('R')]['file_path'].values
                            if len(r_imgs) > 0:
                                patient_exam[view].append({'L': l_path, 'R': cur_exam[indices_for_side_view('R')]['file_path'].values[0]})
                        

            if allow_incomplete or complete_exam:
                # exam is cached with just the paths - but the images are loaded from __getitem__
                label = 1 if (cur_exam['years_to_cancer'].values < 15).any() else 0
                label = torch.tensor(label)
                
                def add_record():
                    if self.multiple_pairs_per_exam:
                        self.exams.append((eid, label, patient_exam))
                    else:
                        self.exams.append(exam_record(eid, label,
                                           patient_exam['CC']['L'],
                                           None,
                                           patient_exam['CC']['R'],
                                           None,
                                           patient_exam['MLO']['L'],
                                           None,
                                           patient_exam['MLO']['R'],
                                           None))
                        
                if oversample_cancer_rate is None:
                    add_record()
                elif label == 1:
                    for i in range(oversample_cancer_rate):
                        add_record()
                else:
                    add_record()
                    
        print(len(self.exams), metadata_frame[metadata_frame['years_to_cancer'] < 100]['exam_id'].unique().shape, metadata_frame[metadata_frame['years_to_cancer'] == 100]['exam_id'].unique().shape)
    
    def load_img(self, path):
        if path is None:
            img = np.zeros((2,2))
            if self.resizer is not None:
                img = self.resizer(img, augment=self.mode=="training")
            return img
        
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)#.astype('int32')
        # Most of our framework expects a channel dimension,
        # so it may be easiest to start complying with that structure
        # img = torch.tensor(img).expand(3, *img.shape)
        if self.resizer is not None:
            img = self.resizer(img, augment=self.mode=="training")
        return img
    
    async def load_img_async(self, path):
        """
        Load images asynchronously. The file load is moved to another thread.
        @returns cv2 image from the path
        """
        if path is None:
            img = np.zeros((2,2))
            if self.resizer is not None:
                img = self.resizer(img, augment=self.mode=="training")
            return (img, None)
        async with aiofiles.open(path, mode='rb') as file:
            file_contents = await file.read()
        img_array = np.frombuffer(file_contents, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        # img = torch.tensor(img).expand(3, *img.shape)

        if self.resizer is not None:
            img = self.resizer(img, augment=self.mode=="training")
        
        return (img, path)
    
    async def load_imgs_async(self, paths):
        """
        Load a batch of images using asyncio in parallel.
        """
        coroutines = [self.load_img_async(path) for path in paths]
        return await asyncio.gather(*coroutines)

    def __len__(self):
        return len(self.exams)
        
    def __getitem__(self, index):
        exam = self.exams[index]
        #if exam[3] is not None:
        #    return exam
        #else:
        if not self.multiple_pairs_per_exam:
            paths = exam[2::2]
        else:
            patient_exam = exam[2]
            def paths_from_exam(view):
                res = []
                for i in range(len(patient_exam[view])):
                    res = res + [patient_exam[view][i]['L'], patient_exam[view][i]['R']]
                    
                if len(res) < self.max_per_view * 2:
                    return res + [None for j in range(self.max_per_view * 2 - len(res))]
                else:
                    return res[:self.max_per_view * 2]
            
            paths = paths_from_exam('CC') + paths_from_exam('MLO')
            
        
        if self.multiple_pairs_per_exam:
            if self.load_images_async:
                path_image_pairs = asyncio.run(self.load_imgs_async(paths))
            else:
                path_image_pairs = [(self.load_img(path), path) for path in paths]
              
            res_tuples = []
            cur_view = 'CC'
            for i in range(0, len(path_image_pairs), 2):
                if i >= len(paths_from_exam('CC')):
                    cur_view = 'MLO'
                a = path_image_pairs[i][0]
                b = path_image_pairs[i+1][0]
                res_tuples = res_tuples + [(a, b, cur_view)]#[(path_image_pairs[i][0], path_image_pairs[i+1][0], cur_view)]

            return ([exam[0], exam[1], res_tuples])
        
        else:
            if self.load_images_async:
                path_image_pairs = asyncio.run(self.load_imgs_async(paths))
            else:
                path_image_pairs = list(map(lambda path: (self.load_img(path), path), paths))
                
            if self.align_images:
                l_cc, r_cc = align_images_given_img(path_image_pairs[0][0].numpy()[0], path_image_pairs[1][0].numpy()[0])
                l_mlo, r_mlo = align_images_given_img(path_image_pairs[2][0].numpy()[0], path_image_pairs[3][0].numpy()[0])
                path_image_pairs = [(torch.tensor(l_cc).expand(3, *l_cc.shape).type(torch.FloatTensor), path_image_pairs[0][1]),
                                    (torch.tensor(r_cc).expand(3, *r_cc.shape).type(torch.FloatTensor), path_image_pairs[1][1]),
                                    (torch.tensor(l_mlo).expand(3, *l_mlo.shape).type(torch.FloatTensor), path_image_pairs[2][1]),
                                    (torch.tensor(r_mlo).expand(3, *r_mlo.shape).type(torch.FloatTensor), path_image_pairs[3][1])]

            #self.exams[index] = ([exam[0], exam[1]] + list(chain(*path_image_pairs)))
            return ([exam[0], exam[1]] + list(chain(*path_image_pairs)))
