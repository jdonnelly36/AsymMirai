import csv
import cv2
import glob
import os
import shutil

data_dir = "datasets/INBreast/SIMPLIFIED-PNG-IMGS/"
target_data_dir = "SIMPLIFIED-PNG-IMGS-DUP-LESION-RISK-5"
metadata_file = "datasets/INBreast/INBreast_four_views.csv"
high_risk_birads = ['5', '6']
flip_and_swap = False
flip_all_left = False
flip_positives = True
files_to_flip = []

trg = 'datasets/INBreast/' + target_data_dir + '/'

#os.mkdir(trg)
 
'''# iterating over all the files in
# the source directory
for fname in files:
    # copying the files to the
    # destination directory
    shutil.copy2(os.path.join(data_dir,fname), trg)'''


with open(metadata_file) as original_metadata:
    old_metadata_reader = csv.reader(original_metadata)

    prev_hash = ''
    patient_id = -1

    for i, row in enumerate(old_metadata_reader):
        if i == 0:
            continue

        laterality = row[2]
        view = row[3][:2]
        file_path = glob.glob(data_dir + row[5] + '_*')[0]

        # Note -- hash remains constant across an exam, so we could
        # use it as our patient ID
        exam_hash = file_path.split('/')[-1][9:25]
        first_hash = file_path.split('/')[-1][:8]
        # Taking the integer part of BIRADS score
        birads = row[7]

        # In the sample metadata, it seems like 0 indicates cancer, 100 indicates no cancer
        if flip_and_swap:
            image = cv2.imread(file_path)
            print("View {}, side {}".format(view, laterality))

            if laterality == 'L':
                opposite_file_path = glob.glob(data_dir + '*_' + exam_hash + '_MG_R_' + view + '*')
            else:
                opposite_file_path = glob.glob(data_dir + '*_' + exam_hash + '_MG_L_' + view + '*')

            if len(opposite_file_path) >= 1:
                target_path = opposite_file_path[0].replace('PNG-IMGS', target_data_dir)
                flippedimage = cv2.flip(image, 1)
                cv2.imwrite(target_path, flippedimage)
                continue
            elif len(opposite_file_path) == 0:
                print("Dropping images with exam_id {}".format(exam_hash))
                continue
            else:
                print("Error: found length > 1 with {}".format(other_side))
                continue

        elif flip_all_left:
            if laterality == 'L':
                image = cv2.imread(file_path)
                target_path = file_path.replace('PNG-IMGS', target_data_dir)
                cv2.imwrite(target_path, image)

                opposite_file_path = glob.glob(data_dir + '*_' + exam_hash + '_MG_R_' + view + '*')

                if len(opposite_file_path) >= 1:
                    target_path = opposite_file_path[0].replace('PNG-IMGS', target_data_dir)
                    flippedimage = cv2.flip(image, 1)
                    cv2.imwrite(target_path, flippedimage)
                    continue

                elif len(opposite_file_path) == 0:
                    print("Dropping images with exam_id {}".format(exam_hash))
                    continue
                else:
                    print("Error: found length > 1 with {}".format(other_side))
                    continue
        ''' 
        Want to do something along the lines of 
            "If we are considering the left breast, the right breast
            has a lession, and we are overwriting lessioned breasts,
            flip left breast and write it over the right breast.
            Otherwise, if we are considering the left breast, the left breast
            has a lession, and we are overwriting healthy breasts,
            flip the left breast and write it over the right breast."
        '''
        # Future problem: If we are flipping lesioned breasts to 
        # overwrite healthy, we will often overwrite both breasts
        if (not flip_positives and birads in high_risk_birads):
            if laterality == 'L':
                other_side = glob.glob(data_dir + '*_' + exam_hash + '_MG_R_' + view + '*')
            else:
                other_side = glob.glob(data_dir + '*_' + exam_hash + '_MG_L_' + view + '*')

            if len(other_side) >= 1:
                # If the length of other_side is 1, the other side exists, so we're good
                # Read the file for the other side, flip it, and write it to our new dir
                image = cv2.imread(other_side[0])
                target_path = file_path.replace('PNG-IMGS', target_data_dir)
                image = cv2.flip(image, 1)
                cv2.imwrite(target_path, image)
            elif len(other_side) == 0:
                print("Dropping images with exam_id {}".format(exam_hash))
                continue
            else:
                print("Error: found length > 1 with {}".format(other_side))
                continue
        else:
            # If we don't need to flip, just copy this image over normally
            if (flip_positives and birads in high_risk_birads):
                files_to_flip.append(file_path)
            image = cv2.imread(file_path)
            target_path = file_path.replace('SIMPLIFIED-PNG-IMGS', target_data_dir)
            cv2.imwrite(target_path, image)

for file_path in files_to_flip:
    exam_hash = file_path.split('/')[-1][9:25]
    laterality = file_path.split('/')[-1][29]
    view = file_path.split('/')[-1][31:33]

    if laterality == 'L':
        other_side = glob.glob(data_dir + '*_' + exam_hash + '_MG_R_' + view + '*')
    else:
        other_side = glob.glob(data_dir + '*_' + exam_hash + '_MG_L_' + view + '*')

    if len(other_side) >= 1:
        # If the length of other_side is 1, the other side exists, so we're good
        # Read the file for the other side, flip it, and write it to our new dir
        image = cv2.imread(file_path)
        target_path = other_side[0].replace('SIMPLIFIED-PNG-IMGS', target_data_dir)
        print(file_path)
        image = cv2.flip(image, 1)
        cv2.imwrite(target_path, image)
        print("Flipped and wrote image {}".format(file_path))
    elif len(other_side) == 0:
        print("Not copying images with exam_id {}".format(exam_hash))
        continue
    else:
        print("Error: found length > 1 with {}".format(other_side))
        continue

