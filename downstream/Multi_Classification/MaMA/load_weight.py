import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from MaMA.model import MaMACLIP
from argparse import ArgumentParser
import torch

def load_model(
    pretrained_model_path: str,
    eval_mode: bool = True,
    **kwargs
) -> MaMACLIP:
    default_args = {
        "batch_size": 36,
        "learning_rate": 4e-5,
        "devices": 4,
        "strategy": "ddp_find_unused_parameters_true",
        "llm_type": "gpt",
        "precision": "bf16-true",
        "peft": "lora",
        "accumulate_grad_batches": 2,
        "grad_ckpt": True,
        "weight_decay": 0.1,
        "warm_up": 4000,
        "emb_dim": 512,
        "max_steps": 40000,
        "pool_feat": True,
        "embed": True,
        "structural_cap": True,
        "slip": True,
        "img_size": 518,
        "crop_size": 518,
        "vit_grad_ckpt": True,
        "load_jpg": True,
        "num_workers": 8,
        "mask_ratio": 1.0,
        "mask_meta": 0.8,
        "symmetric_clip": True,
        "inter_side": True,
        "local_contrast": True,
        "pool_txt_feat": True,
        "late_loss": 8000,
        "symmetric_local": True,
        "num_classes": 7,
        "balanced_test": True,
        "instance_test_cap": True,
        "eval": eval_mode,
    }
    
    default_args.update(kwargs)
    
    parser = ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--llm_type",
        type=str,
        default="gpt",
        help="bert, gpt, llama, llama2, or llama3",
    )
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--rsna_mammo", action="store_true")
    
    parser = MaMACLIP.add_model_specific_args(parser)
    
    args = parser.parse_args([])  
    for k, v in default_args.items():
        setattr(args, k, v)
    
    args.deterministic = False
    
    num_cores = len(os.sched_getaffinity(0))
    if args.num_workers > num_cores:
        args.num_workers = num_cores
    
    if getattr(args, "use_flash_attention", False):
        os.environ["XFORMERS_DISABLED"] = "1"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    
    print(f"\Load pretrained weight from: {pretrained_model_path}")
    model = MaMACLIP.load_from_checkpoint(
        pretrained_model_path,
        map_location="cpu",
        strict=False,
        **args.__dict__
    )
    
    if eval_mode:
        model.eval()
    else:
        model.train()
    
    return model


# 使用示例
if __name__ == "__main__":
    model = load_model(
        pretrained_model_path="/home/jiayi/FM_downstream/Sotas/mama_embed_pretrained_40k_steps_last.ckpt",
        eval_mode=True
    )
