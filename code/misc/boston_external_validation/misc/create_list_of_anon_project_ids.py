import os

filepath = "/project/dane2/wficai/pda/external_validation/Boston/exports/"

dirs = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]

# save to file
with open("/home/dane2/Code/pda_detection/code/misc/boston_external_validation/data_resources/anonymous_project_ids.csv", "w") as f:
    for item in dirs:
        f.write("%s\n" % item)