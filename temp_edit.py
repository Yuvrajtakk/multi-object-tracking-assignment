import json
from pathlib import Path

# Load the notebook
path = Path('mot_pipeline_final.ipynb')
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the config cell (cell 3, index 3 since 0-based)
config_cell = nb['cells'][3]
source = config_cell['source']

# Modify the source
new_source = []
for line in source:
    if 'VIDEO_PATH  = os.path.join(os.getcwd(), "clips", "08fd33_0.mp4")' in line:
        new_source.append('# VIDEO_PATH  = os.path.join(os.getcwd(), "clips", "08fd33_0.mp4")\n')
        new_source.append('INPUT_VIDEO = "clips/08fd33_0.mp4"\n')
    elif 'OUTPUT_PATH = os.path.join(os.getcwd(), "clips", "football.mp4")' in line:
        new_source.append('# OUTPUT_PATH = os.path.join(os.getcwd(), "clips", "football.mp4")\n')
        new_source.append('OUTPUT_DIR = "output"\n')
        new_source.append('os.makedirs(OUTPUT_DIR, exist_ok=True)\n')
        new_source.append('OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated.mp4")\n')
    elif 'assert os.path.isfile(VIDEO_PATH)' in line:
        new_source.append('assert os.path.isfile(INPUT_VIDEO), f"Input video file not found: {INPUT_VIDEO}"\n')
    elif 'cap    = cv2.VideoCapture(VIDEO_PATH)' in line:
        new_source.append('cap    = cv2.VideoCapture(INPUT_VIDEO)\n')
    else:
        new_source.append(line)

config_cell['source'] = new_source

# Now, modify other cells
# Model test cell (index 4)
model_cell = nb['cells'][4]
model_source = model_cell['source']
new_model_source = [line.replace('VIDEO_PATH', 'INPUT_VIDEO') for line in model_source]
model_cell['source'] = new_model_source

# Tracker test cell (index 6)
tracker_test_cell = nb['cells'][6]
tracker_test_source = tracker_test_cell['source']
new_tracker_test_source = [line.replace('VIDEO_PATH', 'INPUT_VIDEO') for line in tracker_test_source]
tracker_test_cell['source'] = new_tracker_test_source

# Main tracker cell (index 7)
main_tracker_cell = nb['cells'][7]
main_tracker_source = main_tracker_cell['source']
new_main_tracker_source = []
for line in main_tracker_source:
    line = line.replace('VIDEO_PATH', 'INPUT_VIDEO')
    line = line.replace('OUTPUT_PATH', 'OUTPUT_VIDEO')
    new_main_tracker_source.append(line)
main_tracker_cell['source'] = new_main_tracker_source

# Output check cell (index 8)
output_cell = nb['cells'][8]
output_source = output_cell['source']
new_output_source = [line.replace('OUTPUT_PATH', 'OUTPUT_VIDEO') for line in output_source]
output_cell['source'] = new_output_source

# Save the notebook
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully.')