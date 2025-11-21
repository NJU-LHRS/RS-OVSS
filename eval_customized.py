import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import customized_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)


def eval_benchmark(vit_type, method, pretrained=None, model_type='SegEarth', fqg=False, long_clip='disable'):

    configs_dict = {
        'OpenEarthMapDataset': './configs/cfg_openearthmap.py',
        'LoveDADataset': './configs/cfg_loveda.py',
        'PotsdamDataset': './configs/cfg_potsdam.py',
        'ISPRSDataset': './configs/cfg_vaihingen.py',
        'UDD5Dataset': './configs/cfg_udd5.py',
        'VDDDataset': './configs/cfg_vdd.py',
        'UAVidDataset': './configs/cfg_uavid.py',
        'iSAIDDataset': './configs/cfg_iSAID.py'
    }
    configs_list = configs_dict.values()

    work_dir = f'./work_dir/{vit_type}/{method}'

    results_file = os.path.join(work_dir, "ovss_results_total.txt")
    result_dict = {}
    for config in configs_list:
        print(f'Evaluating: {config}')
        cfg = Config.fromfile(config)

        cfg.work_dir = work_dir
        cfg.test_dataloader.dataset['pipeline'] = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(448, 448), keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]

        name_path = cfg.model.name_path
        cfg.model = dict(
            type='CustomizedSegmentation',
            model_name=method,
            vit_type=vit_type,
            name_path=name_path,
            model_type=model_type,      # 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'SegEarth'
            pretrained=pretrained,
            long_clip=long_clip,
            fqg=fqg,
        )
        runner = Runner.from_cfg(cfg)
        results = runner.test()

        result_dict[cfg.dataset_type] = results['mIoU']
    result_dict["mean"] = sum(result_dict.values()) / len(result_dict)

    # Write accuracy into txt
    all_results = result_dict
    dataset_names = list(all_results.keys())
    with open(results_file, "w") as f:
        f.write(",".join(dataset_names) + "\n")
        f.write(",".join([f"{result_dict[ds]:.4f}" for ds in dataset_names]) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='OVSS evaluation for RS imagery')

    parser.add_argument('--vit-type', type=str, default='ViT-B-16',
                        help='Vision Transformer type, e.g. ViT-B-16, ViT-L-14')
    parser.add_argument('--method', type=str, default='FarSLIP2',
                        help='Method name, e.g. CLIP FineCLIP CLIPSelf tips cosmos MetaCLIP SkyCLIP LRSCLIP GeoRSCLIP RemoteCLIP FarSLIP1/2')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained open_clip model checkpoint (Only for method==CLIP)')
    parser.add_argument('--model-type', type=str, default='SegEarth',
                        help='CustAttn method for last layer.')
    parser.add_argument('--fqg', action='store_true',
                        help='Enable quick_gelu')
    parser.add_argument('--long-clip', type=str, default='disable', choices=['disable', 'load_from_scratch'],
                        help="Long CLIP mode: 'disable' for vanilla CLIP, 'load_from_scratch' to enable long CLIP")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # run evaluation for baselines (flair FineCLIP CLIPSelf tips cosmos MetaCLIP // SkyCLIP LRSCLIP GeoRSCLIP RemoteCLIP FarSLIP2
    # eval_benchmark(
    #     vit_type='ViT-B-16',
    #     method='FarSLIP2',
    # )

    # run your own open_clip checkpoints
    # eval_benchmark(
    #     vit_type='ViT-B-16',
    #     method='CLIP',
    #     pretrained='path/to/your/model'
    # )

    args = parse_args()
    eval_benchmark(
        vit_type=args.vit_type,
        method=args.method,
        pretrained=args.pretrained,
        model_type=args.model_type,
        fqg=args.fqg,
        long_clip=args.long_clip,
    )