# Export this project.
import json
import os

import labelbox

export_path = "/home/dane2/Code/pda_detection/code/misc/boston_external_validation/data_resources/"

client = labelbox.Client(api_key=os.environ['LBX_API_KEY'])
params = {
	"data_row_details": True,
	"metadata_fields": True,
	"attachments": True,
	"project_details": True,
	"performance_details": True,
	"label_details": True,
	"interpolated_frames": True
}

project = client.get_project('clvmkr23r0cb407xt8zz95ou2')
export_task = project.export_v2(params=params)

export_task.wait_till_done()
if export_task.errors:
	print(export_task.errors)

export_json = export_task.result

# Serialize the json array and save to a file
with open(export_path + 'export_after_review.json', 'w') as file:
    json.dump(export_json, file)

