from unittest import TestCase
import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytest
import warnings

from argparse import Namespace

from asymmetry_model import get_asymmetries
from asymmetry_model.mirai_metadataset import MiraiMetadataset

def assert_almost_equal(expecteds, actuals, epsilon=.0001):
    """
    Given two iterables of floating point numbers, compares that the absolute difference between
    the elements in each does not exceed epsilon.
    """
    assert len(expecteds) == len(actuals), f'Size Mismatch: {len(expecteds)} does not match {len(actuals)}'
    for expected, actual in zip(expecteds, actuals):
        abs_diff = abs(expected - actual)
        assert abs_diff <= epsilon, f'Difference between {expected} and {actual} is {abs_diff} > {epsilon}'

class TestAsymmetryMetrics(TestCase):
    """
    These tests do not guarantee the correctness of the asymetry calculations.
    Rather, they represent the point-in-time behavior of get_asymmetry.

    Noteably, not all parameter combinations work, even if the combination is valid.
    These test cases cover every parameter, but not valid combination.
    """
    
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        from onconet.models.mirai_full import MiraiFull
        cls.mirai_original = MiraiFull(Namespace(
            img_encoder_snapshot=None,
            block_layout=[[('BasicBlock', 2)]],
            block_widening_factor=1,
            use_precomputed_hiddens=False,
            num_chan=3,
            num_groups=1,
            pool_name='GlobalMaxPool',
            use_risk_factors=False,
            dropout=0.0,
            num_classes=2,
            use_region_annotation=False,
            predict_birads=True,
            pred_risk_factors=False,
            survival_analysis_setup=False,
            model_parallel=False,
            pretrained_imagenet_model_name=None,
            pretrained_on_imagenet=False,
            model_name='mirai_full',
            multi_image=False,
            state_dict_path=None,
            num_gpus=0,
            cuda=False,
            img_only_dim=4,
            transformer_snapshot=None,
            transfomer_hidden_dim=4,
            num_images=1,
            num_layers=1,
            num_heads=1))
        
        cls.defaultArgs = {
            "use_latent": False,
            "use_topk": False,
            "use_emd": False,
            "use_mean": False,
            "use_std": False,
            "use_localized_max": False,
            "use_pairwise_max": False,
            "use_localized_difs": False,
            "pixel_align": False,
            "latent_align": False,
            "mahalanobis": False,
            "use_hybrid_difs": False,
            "num_segments": (1, 1),
            "topk_percent": 0.05,
            "match_sizes": True,	
            "mirai_backbone": cls.mirai_original,
            "mirai_cutoff_layer": 'layer1_1',
            "save_path": './{}_feature_asymmetries_hybrid_4_3_fixed.csv',
            "use_latent_mask": False,
            "target_file_name": 'tests/two-patient-inbreast.csv',
            "device": torch.device('cpu')
        }
        
    def test_mean(self):
        args = dict(self.defaultArgs)
        args['use_latent'] = True
        args['use_mean'] = True
        args['match_sizes'] = True

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([0.024225592613220215, 0.049686700105667114], result_df['asymmetries'])

    def test_std(self):
        args = dict(self.defaultArgs)
        args['use_latent'] = True
        args['use_std'] = True
        args['match_sizes'] = True

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([0.028319954872131348, 0.013112127780914307], result_df['asymmetries'])

    def test_localized_diff(self):
        args = dict(self.defaultArgs)
        args['use_latent'] = True
        args['use_localized_difs'] = True
        args['match_sizes'] = True

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([291.1678161621094, 256.3028869628906], result_df['asymmetries'])

    def test_hybrid(self):
        args = dict(self.defaultArgs)
        args['use_latent'] = True
        args['use_hybrid_difs'] = True
        args['image_resizer'] = lambda img: np.resize(img, (4084, 3328))
        
        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([7.283517837524414, 123.49803161621094], result_df['asymmetries'])

    def test_hybrid_extensions(self):
        args = dict(self.defaultArgs)
        args['use_latent'] = True
        args['pixel_align'] = True
        args['use_hybrid_difs'] = True
        args['match_sizes'] = True
        args['use_latent_mask'] = True
        args['num_segments'] = (3,3)

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([214.07235717773438, 194.13047790527344], result_df['asymmetries'])


    def test_localized_max_mahalanobis(self):
        args = dict(self.defaultArgs)

        # latent helps because it takes too much mem in pixel space
        args['use_latent'] = True
        args['use_localized_max'] = True
        args['mahalanobis'] = True
        
        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([174.36080932617188, 195.1273651123047], result_df['asymmetries'])

    def test_localized_max_latent_align(self):
        args = dict(self.defaultArgs)

        # segment helps because it takes too much mem in pixel space
        args['use_latent'] = True
        args['use_localized_max'] = True
        args['latent_align'] = True
        
        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([145.18161010742188, 157.759033203125], result_df['asymmetries'])

    def test_pairwise_max(self):
        args = dict(self.defaultArgs)
        args['use_pairwise_max'] = True

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([0, 0], result_df['asymmetries'])

    def test_segment_space(self):
        args = dict(self.defaultArgs)
        args['use_segment'] = True
        args['use_topk'] = True
        args['topk_percent'] = .1
        args['use_emd'] = True

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([2263.35498046875, 7089.05126953125], result_df['asymmetries'])

    def test_pixel_space(self):
        args = dict(self.defaultArgs)
        args['use_emd'] = True

        result_df = get_asymmetries.get_asymmetries(**args)
        
        assert_almost_equal([857.4647827148438, 3870.38232421875], result_df['asymmetries'])


class TestMiraiDataset(TestCase):

    def setUp(self) -> None:
        self.df = pd.read_csv('tests/two-patient-inbreast.csv')
        self.mmd = MiraiMetadataset(self.df)

    def test_len(self):
        assert len(self.mmd) == 2

        with pytest.raises(IndexError):
            self.mmd.__getitem__(2)

    def check_test_img_files(self, exams):
        sizes = [(3328, 2560), (4084, 3328)]
        for i in range(2):
            exam = exams[i]
            # this works because the exam ids are coded to be 0 and 1 for testing
            assert exam[0] == i
            
            for j in range(2,10):
                if j % 2 == 0:
                    # these are the images
                    assert exam[j].shape == sizes[i], exam[i].shape
                    if exam[j].dtype != np.dtype('uint16'):
                        warnings.warn(f"Warning: dataset is using data type {exam[j].dtype}, which differs from uint16", UserWarning)
                else:
                    # these are the paths
                    assert type(exam[j]) == str, type(exam[j])
    
    def test_image_retrieval(self):
        self.check_test_img_files([self.mmd.__getitem__(0), self.mmd.__getitem__(1)])
    
    def test_image_retrieval_async(self):
        mmd = MiraiMetadataset(self.df, load_images_async=True)
        img_dataset_items = [mmd.__getitem__(0), mmd.__getitem__(1)]
        
        self.check_test_img_files(img_dataset_items)