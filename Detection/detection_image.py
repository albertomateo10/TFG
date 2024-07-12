import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import detection
import os
import numpy as np
from typing import Tuple, Union, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import Compose, Normalize, v2


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

####### Pytorch model #######

### Utils functions ###
def set_seed(seed: int) -> None:
    """
    Define a seed for reproducibility. It allows experiment repetition obtaining the exact same results.
    :param seed: integer number indicating which seed you want to use.
    :return: None.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_inner_model(model: detection) -> Any:
    """
    PyTorch provides a model wrapper to enable multiple GPUs. This function returns the inner model (without wrapper).
    :param model: Torch model, with or without nn.DataParallel wrapper.
    :return: if model is wrapped, it returns the inner model (model.module). Otherwise, it returns the input model.
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def torch_load_cpu(load_path: str) -> Any:
    """
    Load the data saved from a trained model (model weights, optimizer state, last epoch number to resume training...)
    :param load_path: string indicating the path to the data saved from a trained model.
    :return: dictionary containing data saved from a trained model.
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def load_model_path(path: str, model: detection, device: torch.device, optimizer: torch.optim = None) -> Tuple[Any, Any, int]:
    """
    Load the trained weights of a model into the given model.
    :param path: string indicating the path to the trained weights of a model.
    :param model: the model where you want to load the weights.
    :param device: whether gpu or cpu is being used.
    :param optimizer: the optimizer initialized before loading the weights.
    :return:
        model: Torchvision model.
        optimizer: Torch optimizer.
        initial_epoch: first epoch number.
    """

    # Load model state
    load_data = torch_load_cpu(path)
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Load rng state
    torch.set_rng_state(load_data['rng_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

    # Load optimizer state
    if 'optimizer' in load_data and optimizer is not None:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Get initial epoch
    initial_epoch = load_data['initial_epoch']

    return model, optimizer, initial_epoch

def torchvision_model(model_name: str, pretrained: bool = False, num_classes: int = 2) -> Any:
    """
    Return a model from a list of Torchvision models.
    :param model_name: name of the Torchvision model that you want to load.
    :param pretrained: whether pretrained weights are going to be loaded or not.
    :param num_classes: number of classes. Minimum is 2: 0 = background, 1 = object.
    :return:
        model: Torchvision model.
    """

    # Torchvision models
    model_dict = {
        'faster_rcnn_v1': detection.fasterrcnn_resnet50_fpn,
        'faster_rcnn_v2': detection.fasterrcnn_resnet50_fpn_v2,
        'faster_rcnn_v3': detection.fasterrcnn_mobilenet_v3_large_fpn,
        # 'faster_rcnn_v4': detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        # 'fcos_v1': detection.fcos_resnet50_fpn,
        'retinanet_v1': detection.retinanet_resnet50_fpn,
        'retinanet_v2': detection.retinanet_resnet50_fpn_v2,
        'ssd_v1': detection.ssd300_vgg16,
        'ssd_v2': detection.ssdlite320_mobilenet_v3_large,
    }

    # Create model and load pretrained weights (if pretrained=True)
    if model_name in model_dict:
        model = model_dict[model_name](weights='COCO_V1' if pretrained else None)

        # Modify the model's output layer for the number of classes in your dataset
        if 'faster_rcnn' in model_name:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        elif 'retinanet' in model_name:
            in_features = model.head.classification_head.cls_logits.in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = detection.retinanet.RetinaNetClassificationHead(
                in_features, num_anchors, num_classes
            )
        elif 'fcos' in model_name:
            in_features = model.head.classification_head.cls_logits.in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = detection.fcos.FCOSClassificationHead(
                in_features, num_anchors, num_classes
            )
        elif 'ssd_v1' in model_name:
            in_features = [module.in_channels for module in model.head.classification_head.module_list]
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = detection.ssd.SSDClassificationHead(
                in_features, num_anchors, num_classes
            )
        elif 'ssd_v2' in model_name:
            in_features = [module[0][0].in_channels for module in model.head.classification_head.module_list]
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = detection.ssd.SSDClassificationHead(
                in_features, num_anchors, num_classes
            )

    # Error: Model not in list
        else:
            assert False, 'Model {} not in list. Indicate a Torchvision model from the list.'.format(model_name)
    else:
        assert False, 'Model {} not in list. Indicate a Torchvision model from the list.'.format(model_name)

    return model

def get_model(model_name: str, model_path: str = '', num_classes: int = 2,
              lr_data: list = None, pretrained: bool = False,
              use_gpu: bool = False) -> Tuple[Any, Any, int, torch.device]:
    """
    Main function to create and load the model.
    :param model_name: name of the Torchvision model to load.
    :param model_path: path to the model.
    :param num_classes: number of classes. Minimum is 2: 0 = background, 1 = object.
    :param lr_data: list containing [learning rate, learning rate momentum, learning rate decay].
    :param pretrained: whether Torch pretrained weights on COCO dataset are going to be used or not.
    :param use_gpu: whether to use GPU or CPU.
    :return:
        model: Torch model.
        optimizer: Torch optimizer.
        initial_epoch: first epoch number.
        device: torch device indicating whether to use GPU or CPU.
    """

    # Define device (GPU or CPU)
    device_name = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # Load Torchvision model
    model = torchvision_model(model_name, pretrained, num_classes).to(device)
    if use_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    # Define the optimizer
    if lr_data:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr_data[0], momentum=lr_data[1], weight_decay=lr_data[2])
    else:
        optimizer = None

    # Load trained weights, optimizer state, and initial epoch
    if os.path.isfile(model_path):
        print('  [*] Loading Torch model from {}'.format(model_path))
        model, optimizer, initial_epoch = load_model_path(model_path, model, device, optimizer)
    else:
        initial_epoch = 0
        print('Weights not found')

    return model, optimizer, initial_epoch, device

### Model parameters ###

model_name = 'faster_rcnn_v2'                                               # Torchvision model
model_path = f'/Federated_learning/VisDrone/Models/Pytorch.pt' # Path to save trained model

num_classes = 12                  # Number of classes 
resize_shape = (1280, 720)            # Resize images for faster performance. None to avoid resizing
pretrained = True                 # Use weights pre-trained on COCO dataset

# Train parameters
batch_size = 4                    # Batch size
use_gpu = True                    # Use GPU (True) or CPU (False)

# Other parameters (do not change)
num_workers = 4

# Define your model classes
CLASSES = [
    "ignored regions", "pedestrian", "people", "bicycle", "car",
    "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"
]

# Define a color map for the classes
COLORS = {
    "ignored regions": 'blue',
    "pedestrian": 'green',
    "people": 'cyan',
    "bicycle": 'magenta',
    "car": 'red',
    "van": 'yellow',
    "truck": 'black',
    "tricycle": 'orange',
    "awning-tricycle": 'purple',
    "bus": 'pink',
    "motor": 'brown',
    "others": 'gray'
}

### Load model ###
model, _, _, device = get_model(
        model_name, model_path, num_classes, None, pretrained, use_gpu
    )
    
# Set model in eval mode (no gradients)
model.eval()

### Image Preprocessing ###
transform = transforms.Compose([
    # transforms.Resize((1280, 720)),  # Adjust to the size required by your model
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize according to your model
])

### Object prediction ###
def predict_objects(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU or CPU
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Get predictions for the first element of the list (assuming you process one image at a time)
    predictions = outputs[0]
    
    # Get predicted labels and confidence scores
    labels = predictions['labels']
    scores = predictions['scores']
    boxes = predictions['boxes']
    print(labels)
    print(scores)
    print(boxes)
    
    # Get the index of the class with the highest score
    _, max_score_index = scores.max(dim=0)
    
    # Get the class label corresponding to the index
    class_index = labels[max_score_index.item()].item()
    class_label = CLASSES[class_index]

    # Filter detections with confidence greater than 0.10
    high_confidence_indices = scores > 0.7
    labels = labels[high_confidence_indices]
    scores = scores[high_confidence_indices]
    boxes = boxes[high_confidence_indices]
    
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Create the figure and axes to visualize the image and detections
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)
    
    # Track which classes have been detected
    detected_classes = set()
    
    # Iterate over the detections and draw the bounding boxes with class-specific colors
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.cpu().numpy()  
        class_name = CLASSES[label.item()]
        detected_classes.add(class_name)
        color = COLORS[class_name]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    
    # Create a legend with detected class types only
    handles = [patches.Patch(color=COLORS[class_name], label=class_name) for class_name in detected_classes]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Adjust axes to prevent labels from being cut off
    plt.axis('off')
    plt.tight_layout()
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the image with detections drawn
    base_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, os.path.splitext(base_filename)[0] + '_detection.jpg')
    plt.savefig(output_image_path, bbox_inches='tight')
    return class_label, output_image_path  # Return the class label and path to the output image

# Example
image_path = '/VisDrone/Frames/DJI_0758.jpg'  # Replace with your own image

output_dir = '/VisDrone/Detections/'  # Replace with your desired output directory
predicted_class, output_image_path = predict_objects(image_path)
print("Predicted class:", predicted_class)
print("Image with detections saved at:", output_image_path)