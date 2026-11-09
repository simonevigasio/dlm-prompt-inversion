"""Small DiffuSeq for offline tests; no pretrained weights or downloads.

``tiny_diffuseq_adapter``: a real ``DiffuSeqAdapter`` around upstream's PyTorch model and
diffusion with tiny random weights (requires the DiffuSeq checkout, not the checkpoint).
"""
from .adapter import DiffuSeqAdapter, build_model, random_parameters, read_settings, upstream_modules
from .setup import BackboneSetup


class TinyTokenizer:
    pad_token_id, cls_token_id, sep_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]

    def __len__(self):
        return 12

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(token) for token in text.split()]}

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def tiny_diffuseq_adapter(*, prompt_length=1, bottleneck=True, init_std=0.3, init_seed=0):
    """Embeddings 4 wide (16 without the bottleneck), encoder 16 wide, two layers, 8 rows, 20 steps."""
    from transformers import BertConfig

    settings = read_settings()
    upstream = upstream_modules(settings["diffuseq_root"], settings["diffuseq_commit"])
    bert = BertConfig(vocab_size=12, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                      intermediate_size=32, max_position_embeddings=16)
    training_args = {"hidden_dim": 4 if bottleneck else 16, "hidden_t_dim": 8, "seq_len": 8, "vocab_size": 12,
                     "dropout": 0.0, "use_plm_init": "no", "diffusion_steps": 20, "noise_schedule": "sqrt",
                     "timestep_respacing": "", "learn_sigma": False, "use_kl": False, "predict_xstart": True,
                     "rescale_timesteps": True, "rescale_learned_sigmas": False, "sigma_small": False}
    model, diffusion = build_model(upstream, bert, training_args)
    setup = BackboneSetup(weights="random", bottleneck=bottleneck, init="normal", init_std=init_std,
                          init_seed=init_seed)
    provenance = random_parameters(model, setup)
    return DiffuSeqAdapter(upstream=upstream, torch_model=model, diffusion=diffusion, training_args=training_args,
                           tokenizer=TinyTokenizer(), prompt_length=prompt_length, provenance=provenance)
