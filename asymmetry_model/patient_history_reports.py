from IPython.core.display import display
import pathlib
import numpy as np
import torch
import pandas as pd
import asymmetry_model.embed_explore as mbd
import os
import logging
import argparse
import sys
sys.path.append('./asymmetry_model')

parser = argparse.ArgumentParser(
                prog='patient_history_report',
                description='Generate reports for a set of patients')

parser.add_argument('output_path', type=str)
parser.add_argument('--sample-strategy', default=None, type=str)
parser.add_argument('--sample-size', default=1, type=int)
parser.add_argument('--max', default=None, type=int)
parser.add_argument('--patient-ids', '--empi-anon', default=None, nargs='*')
parser.add_argument('--sample-ascending', default=True, type=bool)
parser.add_argument('--include-all-images', default=False, type=bool)
parser.add_argument('--model-path', default='./full_model_epoch_21_3_11_corrected_flex.pt', type=str)

args = parser.parse_args()

op = pathlib.Path(args.output_path)
if not op.is_dir():
    os.mkdir(op)

logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(message)s', filename=f'{args.output_path}/generate.log')
logging.getLogger().addHandler(logging.StreamHandler())
mbd.logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load Datasets
logger.info('loading df_mag')
df_mag = mbd.load_mag(scope='full', cohorts=[1,2])

logger.info('loading df_met')
df_met = mbd.load_met(cohorts=[1,2])

logger.info('merging df_magXmet')
df_magXmet2d = mbd.merge__df_magXmet(df_mag, df_met, image_type='2D')


# Grab MIRAI met and mag
logger.info('loading MIRAI input')
df_miraiinput = pd.read_csv('./2_10_mirai_form_extended_cohorts_1-2.csv')
df_miraiinput_unique = df_miraiinput.loc[df_miraiinput['spot_mag'].isna()][['exam_id', 'years_to_cancer', 'years_to_last_followup']].drop_duplicates()
df_magXmet2dXmiraiinput = df_magXmet2d.merge(df_miraiinput_unique, how='left', left_on=['acc_anon'], right_on=['exam_id'], validate='many_to_one')


## and add some columns
df_magXmet2dXmiraiinput['exam_id'] = df_magXmet2dXmiraiinput['acc_anon']
df_magXmet2dXmiraiinput['patient_id'] = df_magXmet2dXmiraiinput['empi_anon']
df_magXmet2dXmiraiinput['laterality'] = df_magXmet2dXmiraiinput['ImageLateralityFinal']
df_magXmet2dXmiraiinput['view'] = df_magXmet2dXmiraiinput['ViewPosition']
df_magXmet2dXmiraiinput['file_path'] = df_magXmet2dXmiraiinput['png_path']

df_magXmet2dXmiraiinput['has_all_views'] = 1


# Add Risk Scores
logger.info('loading MIRAI risk')
df_mirairisk = pd.read_csv('./2023-01-25_risk_by_patient_full.csv', header=None, names=['LMLO_png', 'RMLO_png', 'LCC_png', 'RCC_png', '1year_risk', '2year_risk', '3year_risk', '4year_risk', '5year_risk', 'unk'])

df_mirairisk_with_acc = df_mirairisk.merge(df_met[['png_path', 'acc_anon', 'empi_anon']], left_on=['LMLO_png'], right_on='png_path')[['acc_anon', 'empi_anon'] + [f'{i}year_risk' for i in range(1,6)]].drop_duplicates()

df_magXmet2dXmiraiall = df_magXmet2dXmiraiinput.merge(df_mirairisk_with_acc, how='left', on=['acc_anon', 'empi_anon'])


# Load Asym Model
logger.info('loading asym model')
mbd.logger.debug('loading asym model from %s', args.model_path)
asym_model = torch.load(args.model_path, map_location = torch.device(mbd.device))

if args.patient_ids is not None and len(args.patient_ids) > 0:
    logger.info('loading provided patient ids %s', args.patient_ids)
    df_for_loop = pd.DataFrame.from_dict({
        "empi_anon": args.patient_ids,
        "file_suffix": [''] * len(args.patient_ids)
    })

elif args.sample_strategy == 'with-path':
    # Find Patients who do develop cancer
    logger.info('finding cases for patients who develop cancer')
    empi_cp, df_4view_groups = mbd.patients_with_cancer_and_prev_screenings(df_magXmet2dXmiraiall, return_groups=True)

    df_4view_groups = df_4view_groups.groupby(level=[0]).size().sort_values(ascending=False)
    df_4view_sample = df_4view_groups.to_frame('file_suffix').groupby(['file_suffix']).apply(lambda group: group.sample(args.sample_size))

    df_for_loop = df_4view_sample[[]].reset_index().sort_values('num_full_screen', ascending=args.sample_ascending)

# Generate Reports
files = []

if args.max is not None:
    df_for_loop = df_for_loop.head(args.max)

logger.info('Starting Report Loop for %s patients', df_for_loop.shape[0])
df_magXmet_for_patients = mbd.df_magXmet_for_patients(df_for_loop['empi_anon'].unique(), df_magXmet=df_magXmet2dXmiraiall)
logger.info('total rows to process %s with %s patients', df_magXmet_for_patients.shape, df_magXmet_for_patients['empi_anon'].unique())

for row in df_for_loop.itertuples():
    logger.info('*************  %s', row.empi_anon)
    df_magXmet_for_patient = df_magXmet_for_patients[df_magXmet_for_patients['empi_anon'] == int(row.empi_anon)]
    logger.info('total rows to process for patient %s: %s', df_magXmet_for_patient['empi_anon'].unique(), df_magXmet_for_patient.shape)
    
    files.append(mbd.patient_report(
        row.empi_anon,
        asym_model=asym_model,
        df_magXmet=df_magXmet_for_patient,
        include_all_images=args.include_all_images,
        pdf_path=f"{args.output_path}/{row.empi_anon}{row.file_suffix}.pdf"
    ))
    
    logger.info(f'*************  {row.empi_anon} Complete with files %s', files)

logger.info(f'Completed entire generation job, %s', files)