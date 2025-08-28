import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ROIs extract using: "classic" for classical method
#                     "det"     for detection model 
#                     "seg"     for segmenation model
option = "seg"

# detection_model: "detr" for RT-DETR
#                  "yolo" for YOLO
detection_model = "yolo"

# rectify
rectify_input_dir   = "0_IDs_input/"
rectify_output_dir  = "1_IDs_rectified/"

# derotation
derotation_ref_dir  = "0_IDs_derotation_ref/"
derotated_dir       = "2_IDs_derotated/"

# classical method
classic_out_dir     = "3_ROIs_classic"
rois_json           = "ROIs.json"
crops_output_fromat = "png"

# detect method
det_out_dir = "3_ROIs_det"

# segment method
seg_out_dir = "3_ROIs_seg"
