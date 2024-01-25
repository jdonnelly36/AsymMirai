import csv
import glob

target_file = "./INBreast_simplified_metadata_4a_floor_risk_by_patient_swapped.csv"
data_dir = "datasets/INBreast/SIMPLIFIED-PNG-IMGS-SWAPPED/"
metadata_file = "datasets/INBreast/INBreast_four_views.csv"
use_risk_by_patient = True
exclude_no_lesion = False
high_risk_birads = ['4a', '4b', '4c', '5', '6']


with open(target_file, 'r+') as target_csv:
    new_metadata_writer = csv.writer(target_csv)
    new_metadata_writer.writerow(['patient_id','exam_id','laterality','view','file_path','years_to_cancer','years_to_last_followup','split_group'])

    with open(metadata_file) as original_metadata:
        old_metadata_reader = csv.reader(original_metadata)

        prev_hash = ''
        patient_id = -1
        rows_by_patient = []

        for i, row in enumerate(old_metadata_reader):
            if i == 0:
                continue

            # Note -- hash remains constant across an exam, so we could
            # use it as our patient ID
            file_path = glob.glob(data_dir + row[5] + '_*')[0]
            exam_hash = file_path.split('/')[-1][9:25]
            if exam_hash != prev_hash:

                if use_risk_by_patient and prev_hash != '' \
                    and not (exclude_no_lesion and rows_by_patient[0][-3] == 100):
                    # Note: the assertion "not (exclude_no_lesion and rows_by_patient[0][-3] == 100)"
                    # says "either we aren't excluding lesionless people, or this person has a lesion"
                    for writing_row in rows_by_patient:
                        new_metadata_writer.writerow(writing_row)

                patient_id += 1 
                prev_hash = exam_hash
                rows_by_patient = []


            exam_id = 0
            laterality = row[2]
            view = row[3]

            # Taking the integer part of BIRADS score
            birads = row[7]
            # In the sample metadata, it seems like 0 indicates cancer, 100 indicates no cancer
            if birads in high_risk_birads:
                years_to_cancer = 0
            else:
                years_to_cancer = 100

            years_to_last_followup = 10
            split_group = 'test'

            new_row = [patient_id, exam_id, laterality, view, file_path, years_to_cancer, years_to_last_followup, split_group]
            
            if not use_risk_by_patient and not (exclude_no_lesion and new_row[-3] == 100):
                new_metadata_writer.writerow(new_row)
                continue

            for patient_row in rows_by_patient:
                # If any previous row for this patient had a sufficiently
                # high BiRads, mark both views of both breasts as high risk
                if len(patient_row) > 0 and patient_row[-3] == 0:
                    new_row[-3] = 0
            
            # If the current view is high risk, update
            # all other views accordingly
            if new_row[-3] == 0:
                for j in range(len(rows_by_patient)):
                    rows_by_patient[j][-3] = 0

            rows_by_patient.append(new_row)
