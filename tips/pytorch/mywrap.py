import os
print(os.getcwd())
import torch
import numpy as np
from torchvision import transforms
from tips.pytorch import image_encoder, text_encoder
from tips.scenic.utils import feature_viz
import io
from PIL import Image

IMAGE_MEAN = (0, 0, 0)
IMAGE_STD = (1.0, 1.0, 1.0)
PATCH_SIZE = 14
MAX_LEN = 64
VOCAB_SIZE = 32000

def build_models(image_checkpoint, text_checkpoint, tokenizer_path, variant="B", image_size=448):
    """buliding image & text encoder"""
    # vision encoder
    weights_image = dict(np.load(image_checkpoint, allow_pickle=False))
    for k in weights_image:
        weights_image[k] = torch.tensor(weights_image[k])
    ffn_layer = 'swiglu' if variant == 'g' else 'mlp'

    if variant == 'B':
        model_image = image_encoder.vit_base(
            img_size=image_size,
            patch_size=PATCH_SIZE,
            ffn_layer=ffn_layer,
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
    elif variant == 'L':
        model_image = image_encoder.vit_large(
            img_size=image_size,
            patch_size=PATCH_SIZE,
            ffn_layer=ffn_layer,
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
    else: raise (f'Unsupported {variant}')

    model_image.load_state_dict(weights_image)
    model_image.eval()

    # text encoder
    with open(text_checkpoint, 'rb') as fin:
        inbuffer = io.BytesIO(fin.read())
    np_weights_text = np.load(inbuffer, allow_pickle=False)
    weights_text = {k: torch.from_numpy(v) for k, v in np_weights_text.items()}
    temperature = weights_text.pop('temperature')

    def get_text_config(v):
        return {
            'hidden_size': {'S':384,'B':768,'L':1024,'So400m':1152,'g':1536}[v],
            'mlp_dim':    {'S':1536,'B':3072,'L':4096,'So400m':4304,'g':6144}[v],
            'num_heads':  {'S':6,   'B':12,  'L':16,   'So400m':16,  'g':24}[v],
            'num_layers': {'S':12,  'B':12,  'L':12,   'So400m':27,  'g':12}[v],
        }

    model_text = text_encoder.TextEncoder(
        get_text_config(variant), vocab_size=VOCAB_SIZE,
    )
    model_text.load_state_dict(weights_text)
    model_text.eval()

    tokenizer = text_encoder.Tokenizer(tokenizer_path)

    return model_image, model_text, tokenizer, temperature

def _convert_to_rgb(image):

    if isinstance(image, Image.Image):
        return image.convert('RGB')
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3: return image[[2, 1, 0], :, :]
        elif image.ndim == 4: return image[:, [2, 1, 0], :, :]
        else: raise TypeError("Input ndim be 3 or 4.")
    else:
        raise TypeError("Input must be PIL.Image or torch.Tensor")

def extract_image_features(batch_images, model_image, image_size=448, model_type='SegEarth'):
    """
    Input:
      - batch_images: torch.Tensor or np.ndarray, shape [B,H,W,C] or [B,C,H,W]
    Output:
      - image_embeddings: [B,D]
    """

    if isinstance(batch_images, np.ndarray):
        batch_images = torch.from_numpy(batch_images)

    if isinstance(batch_images, list):
        batch_images = batch_images[0]

    if batch_images.ndim == 3:
        batch_images = batch_images.unsqueeze(0)
    # [B,H,W,C] -> [B,C,H,W]
    if batch_images.shape[1] not in [1, 3]:
        batch_images = batch_images.permute(0, 3, 1, 2)

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        _convert_to_rgb,
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    batch_images = preprocess(batch_images)

    with torch.no_grad():
        output = model_image(batch_images, model_type=model_type)
        # cls_token = feature_viz.normalize(output[0][:, 0])  # Choose the first CL
        # spatial_feature = feature_viz.normalize(output[2])
        cls_token = output[0][:, 0]
        spatial_feature = output[2]

        # spatial_feature = torch.reshape(spatial_feature,
        #     (batch_images.shape[0], int(image_size / PATCH_SIZE), int(image_size / PATCH_SIZE), -1))

    return cls_token, spatial_feature


def extract_text_features(query_words, model_text, tokenizer, openai_imagenet_template=None):
    """
    输入:
      - texts: list[str]
    输出:
      - text_embeddings: [T,D]
    """
    text_ids, text_paddings = tokenizer.tokenize(query_words, max_len=MAX_LEN)
    if openai_imagenet_template:
        query_features = []
        with torch.no_grad():
            for qw in query_words:
                texts = [temp(qw) for temp in openai_imagenet_template]
                text_ids, text_paddings = tokenizer.tokenize(texts, max_len=MAX_LEN)
                text_ids = torch.from_numpy(text_ids)
                text_paddings = torch.from_numpy(text_paddings)
                text_embeddings = model_text(text_ids, text_paddings)
                text_embeddings = feature_viz.normalize(text_embeddings)
                feature = text_embeddings.mean(dim=0, keepdim=True)
                feature = feature_viz.normalize(feature)
                query_features.append(feature)
        text_embeddings = torch.cat(query_features, dim=0)

    else:
        with torch.no_grad():
            text_embeddings = model_text(torch.from_numpy(text_ids), torch.from_numpy(text_paddings))
            text_embeddings = feature_viz.normalize(text_embeddings)

    return text_embeddings


def main():
    model_image, model_text, tokenizer, temperature = build_models(
        "checkpoints/tips_oss_b14_highres_distilled_vision.npz",
        "checkpoints/tips_oss_b14_highres_distilled_text.npz",
        "checkpoints/tokenizer.model",
        variant="B",
        image_size=448
    )

    texts = ["a cat", "a dog"]
    txt_feat = extract_text_features(texts, model_text, tokenizer)

    imgs = torch.randn(4, 3, 448, 448)
    cls_token, spatial_feature = extract_image_features(imgs, model_image)
    print()

if __name__ == '__main__':
    main()