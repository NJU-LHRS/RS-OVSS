import torch
import torch.nn as nn
import sys
sys.path.append("..")

from prompts.imagenet_template import *

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F
import os

# from tips.pytorch.mywrap import build_models, extract_text_features, extract_image_features


MODEL_PATHS = {
    'FarSLIP1': {
        'ViT-B-32': "checkpoints/FarSLIP1_ViT-B-32.pt",
        'ViT-B-16': "checkpoints/FarSLIP1_ViT-B-16.pt"
    },
    'FarSLIP2': {
        'ViT-B-32': "checkpoints/FarSLIP2_ViT-B-32.pt",
        'ViT-B-16': "checkpoints/FarSLIP2_ViT-B-16.pt"
    },
    'FarSLIP2-VC': {
        'ViT-B-32': "checkpoints/FarSLIP2_VC_ViT-B-32.pt",
        'ViT-B-16': "checkpoints/FarSLIP2_VC_ViT-B-16.pt",
    },
    'RemoteCLIP': {
        'ViT-B-32': "checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt",
        'ViT-L-14': "checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-L-14.pt"
    },
    'GeoRSCLIP': {
        'ViT-B-32': "checkpoints/GeoRSCLIP-ckpt/RS5M_ViT-B-32.pt",
        'ViT-L-14': "checkpoints/GeoRSCLIP-ckpt/RS5M_ViT-L-14.pt",
        'ViT-H-14': "checkpoints/GeoRSCLIP-ckpt/RS5M_ViT-H-14.pt"
    },
    'SkyCLIP': {
        'ViT-B-32': "checkpoints/SkyCLIP_ViT_B32_top50pct/epoch_20.pt",
        'ViT-L-14': "checkpoints/SkyCLIP_ViT_L14_top50pct/epoch_20.pt"
    },
    'LRSCLIP': {
        'ViT-B-16': "checkpoints/LRSCLIP_ViT-B-16.pt"
    }
}


MODEL_CONFIG = {
    'FarSLIP1': {
        'force_quick_gelu': True,
    },
    'FarSLIP2': {
        'long_clip': 'load_from_scratch',
        'force_quick_gelu': True,
    },
    'FarSLIP2-VC': {
        'force_quick_gelu': True,
    },
    'LRSCLIP': {
        'long_clip': 'load_from_scratch',
        'force_quick_gelu': True,
    }
}


@MODELS.register_module()
class CustomizedSegmentation(BaseSegmentor):
    def __init__(self,
                 model_name,
                 vit_type,
                 name_path,
                 model_type,
                 ignore_residual=True,
                 prob_thd=0.0,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 bg_idx=0,
                 logit_scale=50,
                 is_half=False,
                 pretrained=None,
                 long_clip='load_from_scratch',
                 fqg = False
                 ):
        super().__init__()
        if model_name == 'tips':
            mean = [0., 0., 0.]
            std = [255., 255., 255.]
            rgb_to_bgr = False
        else:
            mean = [122.771, 116.746, 104.094]
            std = [68.501, 66.632, 70.323]
            rgb_to_bgr = True
        data_preprocessor = SegDataPreProcessor(
            mean=mean,
            std=std,
            rgb_to_bgr=rgb_to_bgr)
        super().__init__(data_preprocessor=data_preprocessor)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0
        self.bg_idx = bg_idx
        self.prob_thd = prob_thd
        self.model_type = model_type
        self.vit_type = vit_type
        self.is_half = is_half
        self.logit_scale = logit_scale
        self.ignore_residual = ignore_residual
        self.printed_warning = False

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(self.device)

        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        # ====== Load TIPS model ======
        if model_name == 'tips':
            self.image_size = 448
            vit_map = {
                "ViT-B-14": ("b14", "B"),
                "ViT-L-14": ("l14", "L"),
            }
            if self.vit_type not in vit_map:
                raise ValueError(f"Unrecognized vit type: {self.vit_type} for model {self.model_name}.")

            suffix, variant = vit_map[self.vit_type]
            self.model_image, self.model_text, self.tokenizer, _ = build_models(
                f"tips/pytorch/checkpoints/tips_oss_{suffix}_highres_distilled_vision.npz",
                f"tips/pytorch/checkpoints/tips_oss_{suffix}_highres_distilled_text.npz",
                f"tips/pytorch/checkpoints/tokenizer.model",
                variant=variant,
                image_size=self.image_size
            )

            self.patch_size = (self.model_image.patch_size, self.model_image.patch_size)
            self.visual_output_dim = self.model_image.embed_dim

            self.query_features = extract_text_features(query_words, self.model_text, self.tokenizer, openai_imagenet_template)  # normalized
            self.query_features = self.query_features.to(self.device)

        # ====== Load flair model ======
        elif model_name == 'flair':
            import flair
            if self.vit_type == 'ViT-B-16':
                pretrained = flair.download_weights_from_hf(model_repo='xiaorui638/flair', filename='flair-cc3m-recap.pt')
                self.model, _, self.preprocess = flair.create_model_and_transforms('ViT-B-16-FLAIR', pretrained=pretrained)
                tokenizer = flair.get_tokenizer('ViT-B-16-FLAIR')
            else: raise RuntimeError(f'Unrecognized vit type: {self.vit_type} for model {self.model_name}.')
            self.model.to(self.device)
            self.model.eval()

            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = tokenizer([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                    feature, _ = self.model.encode_text(query)
                    feature = self.model.text_post(feature)  # (B*K, D)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0)

            # self.logit_scale = self.model.logit_scale
            self.visual_output_dim = self.model.visual.output_dim
            self.patch_size = self.model.visual.patch_size
            self.image_size = 224

        # ====== Load FineCLIP model ======
        elif model_name == 'FineCLIP':
            from FineCLIP.open_clip import create_model_and_transforms, get_tokenizer
            if self.vit_type == 'ViT-B-16':
                pretrained = 'FineCLIP/checkpoints/FineCLIP_coco_vitb16.pt'
                model_name = "EVA02-CLIP-B-16"
                image_size = 224
            elif self.vit_type == 'ViT-L-14':
                pretrained = 'FineCLIP/checkpoints/FineCLIP_coco_vitl14.pt'
                model_name = "EVA02-CLIP-L-14-336"
                image_size = 336
            else: raise RuntimeError(f'Unrecognized vit type: {self.vit_type} for model {self.model_name}.')

            self.model, _, self.preprocess = create_model_and_transforms(
                model_name,
                "eva",
                "amp",
                device="cuda",
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=None,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                aug_cfg={},
                output_dict=True,
                cache_dir=pretrained,
                det_image_size=image_size,
                dataset_type="grid_distill",
            )
            self.model.to(self.device)
            self.model.eval()

            tokenizer = get_tokenizer(model_name)
            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = tokenizer([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                    feature = self.model.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0)

            # self.logit_scale = self.model.logit_scale
            self.visual_output_dim = self.model.visual.embed_dim
            self.patch_size = (int(self.vit_type.split('-')[-1]), int(self.vit_type.split('-')[-1]))
            self.image_size = image_size

        # ====== Load CLIPSelf model ======
        elif model_name == 'CLIPSelf':
            from CLIPSelf.open_clip import create_model_and_transforms, get_tokenizer
            if self.vit_type == 'ViT-B-16':
                pretrained = 'CLIPSelf/checkpoints/eva_vitb16_coco_clipself_proposals.pt'
                model_name = "EVA02-CLIP-B-16"
                image_size = 224
            elif self.vit_type == 'ViT-L-14':
                pretrained = 'CLIPSelf/checkpoints/eva_vitl14_coco_clipself_proposals.pt'
                model_name = "EVA02-CLIP-L-14-336"
                image_size = 336
            else: raise (f'Unrecognized vit type: {self.vit_type} for model {self.model_name}.')

            self.model, _, self.preprocess = create_model_and_transforms(
                model_name,
                "eva",
                "amp",
                device="cuda",
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=None,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                aug_cfg={},
                output_dict=True,
                cache_dir=pretrained,
                det_image_size=image_size,
            )
            self.model.to(self.device)
            self.model.eval()

            tokenizer = get_tokenizer(model_name)
            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = tokenizer([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                    feature = self.model.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0)

            # self.logit_scale = self.model.logit_scale
            self.visual_output_dim = self.model.visual.embed_dim
            self.patch_size = (int(self.vit_type.split('-')[-1]), int(self.vit_type.split('-')[-1]))
            self.image_size = image_size

        # ====== Load cosmos model ======
        elif model_name == 'cosmos':
            from cosmos.open_clip import create_model_and_transforms, get_tokenizer
            pretrained = 'cosmos/checkpoints/cosmos_vitb16_merged30m_2.pt'
            self.model, _, self.preprocess = create_model_and_transforms(
                "ViT-B-16",
                pretrained,
                device="cuda",
                attentional_pool=True,
                cosmos=True,
            )
            self.model.to(self.device)
            self.model.eval()

            tokenizer = get_tokenizer("ViT-B-16")
            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = tokenizer([temp(qw) for temp in openai_imagenet_template]).to(self.device)
                    feature = self.model.encode_text(query)['text_features']
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0)

            # self.logit_scale = self.model.logit_scale
            self.visual_output_dim = self.model.visual.output_dim
            self.patch_size = self.model.visual.patch_size
            self.image_size = 224

        # ====== Load OpenCLIP model ======
        else:
            print('Using Open-CLIP model with half precision.')
            from open_clip import tokenizer, create_model
            self.is_half = True
            self.context_length = 77

            # ====== Your own CLIP model or openai CLIP ======
            if model_name == 'CLIP':
                if pretrained:
                    self.model = create_model(vit_type, pretrained=pretrained, precision='fp16',
                                            long_clip=long_clip, force_quick_gelu=fqg)
                    print(f'Loading CLIP with pretrained model: {pretrained} (force_quick_gelu={fqg} long_clip={long_clip}).')
                else:
                    self.model = create_model(vit_type, pretrained='openai', precision='fp16')
                    print('Loading OpenAI CLIP...')

            elif model_name == 'MetaCLIP':
                self.model = create_model(f'{vit_type}-quickgelu', pretrained='metaclip_fullcc', precision='fp16')

            # ====== RS CLIP variants ======
            else:
                if model_name in MODEL_PATHS and vit_type in MODEL_PATHS[model_name]:
                    pretrained = MODEL_PATHS[model_name][vit_type]
                else: raise RuntimeError(f'Unrecognized vit type: {vit_type} for model {self.model_name}...')
                config = MODEL_CONFIG.get(model_name, {})

                # if enable long-clip
                long_clip = config.get('long_clip', 'disable')
                if not long_clip == 'disable': self.context_length = 248

                # if enable quick_gelu
                force_quick_gelu = config.get('force_quick_gelu', False)

                print(f'Loading {model_name} (force_quick_gelu={force_quick_gelu} long_clip={long_clip})...')
                self.model = create_model(vit_type, pretrained=pretrained, precision='fp16',
                                          long_clip=long_clip, force_quick_gelu=force_quick_gelu)


            self.model.eval().to(self.device)
            query_features = []
            with torch.no_grad():
                for qw in query_words:
                    query = tokenizer.tokenize([temp(qw) for temp in openai_imagenet_template], context_length=self.context_length).to(self.device)
                    feature = self.model.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            self.query_features = torch.cat(query_features, dim=0)

            self.patch_size = self.model.visual.patch_size
            self.image_size = self.model.visual.image_size[0]


    @torch.no_grad()
    def predict(self, inputs, data_samples, reture_image_features=False):

        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        if self.is_half: inputs = inputs.half()

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'], model_type=self.model_type)

        return self.postprocess_result(seg_logits, data_samples)    # (B, num_classes, H, W) => (B, H, W)


    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logit = seg_logits[i] * self.logit_scale
            seg_logit = seg_logit.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logit = seg_logit.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logit = (seg_logit * cls_index).max(1)[0]

            seg_pred = seg_logit.argmax(0, keepdim=True)
            seg_pred[seg_logit.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx
            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logit}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples


    def forward_feature(self, inputs, logit_size=None, model_type='SegEarth'):
        if type(inputs) == list:
            inputs = inputs[0]
        if inputs.shape[-2:] != (self.image_size , self.image_size):
            if not self.printed_warning:
                print(f"Resize input shape {inputs.shape[-2:]} to model required shape {(self.image_size , self.image_size)}.")
                self.printed_warning = True
            resized_inputs = F.interpolate(inputs, size=(self.image_size , self.image_size), mode='bilinear')
        else: resized_inputs = inputs

        # ====== TIPS ======
        if self.model_name == 'tips':
            cls_token, image_features = extract_image_features(resized_inputs, self.model_image, model_type=model_type)

            cls_token /= cls_token.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            cls_logits = cls_token @ self.query_features.T
            seg_logits = image_features @ self.query_features.T

            if self.output_cls_token:
                seg_logits = seg_logits + cls_logits.unsqueeze(1) * self.cls_token_lambda

        # ====== flair ======
        elif self.model_name == 'flair':
            cls_token, patch_tokens = self.model.get_seg_features(image=resized_inputs, model_type=model_type)

            cls_token = F.normalize(cls_token, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)

            cls_logits = cls_token @ self.query_features.t()
            seg_logits = patch_tokens @ self.query_features.t()

            if self.output_cls_token:
                seg_logits = seg_logits + cls_logits.unsqueeze(1) * self.cls_token_lambda

        # ====== FineCLIP & CLIPSelf ======
        elif self.model_name == 'FineCLIP' or self.model_name == 'CLIPSelf':
            patch_tokens = self.model.encode_dense(resized_inputs)
            patch_tokens /= patch_tokens.norm(dim=-1, keepdim=True)

            seg_logits = patch_tokens @ self.query_features.t()

        # ====== cosmos ======
        elif self.model_name == 'cosmos':
            use_csa = True
            if use_csa:
                csa_image_features, _ = self.model.visual(resized_inputs, return_all=True, csa=True)
                csa_image_features = csa_image_features @ self.model.visual.proj  # [B, L-1, C]

                image_features = csa_image_features
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                image_features, _ = self.model.visual(inputs, return_all=True, csa=use_csa)
                image_features = image_features @ self.model.visual.proj  # [B, L-1, C]
                image_features /= image_features.norm(dim=-1, keepdim=True)
            patch_tokens = image_features

            seg_logits = patch_tokens @ self.query_features.t()

        # ====== Open-CLIP series ======
        else:
            image_features = self.model.encode_image(resized_inputs, self.model_type, self.ignore_residual, self.output_cls_token)

            if self.output_cls_token:
                image_cls_token, image_features = image_features
                image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
                cls_logits = image_cls_token @ self.query_features.T

            image_features /= image_features.norm(dim=-1, keepdim=True)
            seg_logits = image_features @ self.query_features.T

            if self.output_cls_token:
                seg_logits = seg_logits + cls_logits.unsqueeze(1) * self.cls_token_lambda


        w, h = self.image_size // self.patch_size[0], self.image_size // self.patch_size[1]
        # w, h = inputs[0].shape[-2] // self.patch_size[0], inputs[0].shape[-1] // self.patch_size[1]
        out_dim = seg_logits.shape[-1]
        seg_logits = seg_logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        if logit_size == None:
            seg_logits = nn.functional.interpolate(seg_logits, size=inputs.shape[-2:], mode='bilinear')
        else:
            seg_logits = nn.functional.interpolate(seg_logits, size=logit_size, mode='bilinear')

        return seg_logits


    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))

        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img, model_type=self.model_type)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits


    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(source):
    if isinstance(source, str) and os.path.isfile(source):
        with open(source, 'r') as f:
            name_sets = f.readlines()
    elif isinstance(source, list):
        name_sets = source
    else:
        raise ValueError("Source must be .txt path or list.")
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices
