import Algorithmia
import torch
from PIL import Image
import json
from torchvision import models
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

CLIENT = Algorithmia.client()
SMID_ALGO = "algo://util/SmartImageDownloader/0.2.x"
LABEL_PATH = "data://AlgorithmiaSE/image_cassification_demo/imagenet_class_index.json"
MODEL_PATHS = {
    "old_model": 'data://AlgorithmiaSE/image_cassification_demo/squeezenet1_1-f364aa15.pth',
    'new_model': 'data://AlgorithmiaSE/image_cassification_demo/alexnet-owt-4df8aa71.pth',
}

def load_labels():
    local_path = CLIENT.file(LABEL_PATH).getFile().name
    with open(local_path) as f:
        labels = json.load(f)
    labels = [labels[str(k)][1] for k in range(len(labels))]
    return labels


def load_model(name):
    if name == "old_model":
        model = models.squeezenet1_1()
        weights_path = CLIENT.file(MODEL_PATHS['old_model']).getFile().name
    else:
        model = models.alexnet()
        weights_path = CLIENT.file(MODEL_PATHS['new_model']).getFile().name
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model = model.float().eval()
    return model


def load_image(image_url):
    input = {"image": image_url, "resize": {'width': 224, 'height': 224}}
    result = CLIENT.algo(SMID_ALGO).pipe(input).result["savePath"][0]
    local_path = CLIENT.file(result).getFile().name
    img_data = Image.open(local_path)
    return img_data


def infer_image(tupl):
    input, n = tupl
    if isinstance(input, str):
        image_url = input
        label = None
    else:
        image_url = input['image_url']
        label = input.get('label', None)
    image_data = load_image(image_url)
    transformed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_tensor = transformed(image_data).unsqueeze(dim=0)
    infered = model.forward(img_tensor)
    preds, indicies = torch.sort(torch.softmax(infered.squeeze(), dim=0), descending=True)
    predicted_values = preds.detach().numpy()
    indicies = indicies.detach().numpy()
    result = {'predictions': [], 'image_url': image_url}
    if label:
        result['ground_truth'] = label
    for i in range(n):
        label = labels[indicies[i]].lower().replace("_", " ")
        confidence = float(predicted_values[i])
        result['predictions'].append({"label": label, "confidence": confidence})
    return result


def calculate_topn_accuracy(results):
    accuracy = 0.0
    for result in results:
        label = result['ground_truth']
        for pred in result['predictions']:
            if label == pred['label']:
                accuracy += 1.0
                break
    accuracy /= len(results)
    return accuracy


# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages
def apply(input):
    if isinstance(input, dict):
        if "n" in input:
            n = input['n']
        else:
            n = 3
        if "data" in input:
            if isinstance(input['data'], str):
                output = infer_image((input['data'], n))
            elif isinstance(input['data'], list):
                with ThreadPoolExecutor(max_workers=5) as executor:
                    output = executor.map(infer_image, zip(input['data'], repeat(n, len(input['data']))))
                    output = list(output)
            else:
                raise Exception("'data' must be a image url or a list of image urls (with labels)")
            if "operation" in input:
                if input['operation'] == "benchmark":
                    accuracy = calculate_topn_accuracy(output)
                    return accuracy
                else:
                    return output
            else:
                return output

        else:
            raise Exception("'data' must be defined")
    elif isinstance(input, str):
        n = 3
        output = infer_image((input, n))
        return output
    else:
        raise Exception("input  must be a dictionary/json object or a string")

model = load_model("new_model")
labels = load_labels()

if __name__ == "__main__":
    input = {"data": "https://i.imgur.com/bXdORXl.jpeg"}
    result = apply(input)
    print(result)
