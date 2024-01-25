import csv
import glob
import copy

def prep_csv(
    target_file = "./INBreast_metadata_5_floor_risk_by_patient_swapped.csv",
    data_dir = "datasets/INBreast/PNG-IMGS-SWAPPED/",
    metadata_file = "datasets/INBreast/INBreast_four_views.csv",
    use_risk_by_patient = True,
    flip_positives = False,
    high_risk_birads = ['5', '6'],
):

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

                laterality = row[2]

                exam_id = 0
                view = row[3]

                birads = row[7]

                # In the sample metadata, it seems like 0 indicates cancer, 100 indicates no cancer
                if birads in high_risk_birads:
                    years_to_cancer = 0
                else:
                    years_to_cancer = 100

                years_to_last_followup = 10
                split_group = 'test'

                # Note -- hash remains constant across an exam, so we could
                # use it as our patient ID
                file_path = glob.glob(data_dir + row[5] + '_*')
                if len(file_path) != 1:
                    print("Skipping file path {}".format(file_path))
                    continue
                else:
                    file_path = file_path[0]

                # The exam ID
                exam_hash = file_path.split('/')[-1][9:25]
                if exam_hash != prev_hash:

                    if use_risk_by_patient and prev_hash != '':
                        for writing_row in rows_by_patient:
                            new_metadata_writer.writerow(writing_row)

                    patient_id += 1 
                    prev_hash = exam_hash
                    rows_by_patient = []

                new_row = [patient_id, exam_id, laterality, view, file_path, years_to_cancer, years_to_last_followup, split_group]
                
                if not use_risk_by_patient:
                    new_metadata_writer.writerow(new_row)
                    continue

                # If we reach this point, we want diagnoses consistent across each exam
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


            if use_risk_by_patient and prev_hash != '':
                for writing_row in rows_by_patient:
                    new_metadata_writer.writerow(writing_row)
