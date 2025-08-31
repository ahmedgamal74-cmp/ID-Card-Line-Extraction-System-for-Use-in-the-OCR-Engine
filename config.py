# device: "cuda" if available | "cpu" is default
device = "cpu"

# ROIs extract using: "classic" for classical method | "det" for detection model | "seg" for segmenation model
extract_option = "seg"

# detection_model: "detr" for RT-DETR | "yolo" for YOLO
detection_model = "yolo"

# models paths
segmenation_model_path = "models/yolov8s_seg.pt"
if detection_model=='detr':
    detection_model_path = "models/rt_detr_1.pt"  
else:
    detection_model_path = "models/yolov8s_detect.pt"

# input image types
image_input_types = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# rectify
rectify_input_dir   = "0_IDs_input/"
rectify_output_dir  = "1_IDs_rectified/"

# derotation
derotation_ref_dir  = "0_IDs_derotation_ref/"
derotated_dir       = "2_IDs_derotated/"

# classical method
classic_out_dir     = "3_ROIs_classic"
rois_json           = "ROIs.json"

# detect method
det_out_dir = "3_ROIs_det"

# segment method
seg_out_dir = "3_ROIs_seg"
