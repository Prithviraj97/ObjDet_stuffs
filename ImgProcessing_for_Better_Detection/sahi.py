import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from PIL import Image

# Load YOLOv5 model
model = attempt_load('yolov5s.pt', map_location=torch.device('cpu'))

# Define SAHI layers
sahi_layers = torch.nn.Sequential(
    torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
    torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
    torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
    torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
)

# Define function for applying SAHI to feature maps
def apply_sahi(features):
    scales = [0.125, 0.25, 0.5, 1.0]
    sahi_maps = []
    for scale in scales:
        scaled_features = torch.nn.functional.interpolate(features, scale_factor=scale, mode='bilinear')
        sahi_map = sahi_layers(scaled_features)
        sahi_map = torch.nn.functional.interpolate(sahi_map, size=features.shape[-2:], mode='bilinear')
        sahi_maps.append(sahi_map)
    return torch.cat(sahi_maps, dim=1)

# Define function for detecting small objects using SAHI
def detect_small_objects(img_path):
    img = Image.open(img_path)
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_size = img_tensor.shape[-2:]
    features = model.backbone(img_tensor)
    features = model.neck(features)
    features = apply_sahi(features)
    det = model.head(features)
    det = non_max_suppression(det, conf_thres=0.5, iou_thres=0.5)
    if det[0] is not None:
        # Get bounding box coordinates for detected small objects
        bboxes = det[0][:, :4]
        bboxes[:, 0] *= img_size[0]
        bboxes[:, 1] *= img_size[1]
        bboxes[:, 2] *= img_size[0]
        bboxes[:, 3] *= img_size[1]
        bboxes = bboxes.round().long()
        return bboxes.tolist()
    else:
        return None

detect_small_objects("left005200.png")