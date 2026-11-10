"""DiffuSeq as a victim model of the attack: weights, Stage B token model, Stage A forward map.

``DiffuSeqAdapter.from_config()`` loads the pinned DiffuSeq checkpoint described by
``configs/models/diffuseq.yaml`` (QQP paraphrase by default), after checking the DiffuSeq
checkout (a git submodule at models/diffuseq/upstream). Upstream is PyTorch: its own
model and diffusion run the sampler that produces observations (observe.py), and
forward.py ports the forward pass to JAX for the attack. A ``BackboneSetup`` can replace
the released weights with random ones, with or without the bottleneck.

DiffuSeq has no encoder: Stage A recovers the prompt's token embeddings E[prompt]
directly, so Stage B has one target representation, the prompt code: the embedding rows.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from ...backbone.knowledge import AttackerKnowledge
from ...tokens.interface import PROMPT_CODE
from ..hub import from_hub
from ..schedule import step_time
from .forward import BackboneForward, ForwardMetadata
from .setup import BackboneSetup

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "models" / "diffuseq.yaml"
# Upstream modules used here; they live in the "diffuseq" package of the checkout.
UPSTREAM_MODULES = ("transformer_model", "gaussian_diffusion", "rounding")


def upstream_modules(root: Path, expected_commit: str):
    """Import upstream's ``diffuseq`` package read-only; reject a changed or unpinned checkout.

    Only the package is imported (by its location), so the checkout's top-level scripts
    never shadow other modules.
    """
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError("DiffuSeq source missing, run 'git submodule update --init'")
    if not expected_commit:
        raise ValueError("expected commit required for consistency in experiments")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != expected_commit:
        raise ValueError(f"DiffuSeq commit mismatch: {commit} != {expected_commit}, use "
                         "'git submodule update --init' to reset the submodule commit")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "diffuseq"], text=True):
        raise ValueError("DiffuSeq tracked source differs from the pinned commit")
    package = root / "diffuseq"
    loaded = sys.modules.get("diffuseq")
    if loaded is None:
        sys.dont_write_bytecode = True  # Do not create __pycache__ in the DiffuSeq checkout.
        spec = importlib.util.spec_from_file_location("diffuseq", package / "__init__.py",
                                                      submodule_search_locations=[str(package)])
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["diffuseq"] = loaded
        spec.loader.exec_module(loaded)
    elif Path(loaded.__file__).resolve().parent != package:
        raise RuntimeError("Import name diffuseq is already occupied by another DiffuSeq or other code")
    return {name: importlib.import_module(f"diffuseq.{name}") for name in UPSTREAM_MODULES}


def read_settings(path: str | Path = DEFAULT_SETTINGS):
    """Model settings, with ``diffuseq_root`` resolved relative to the settings file."""
    path = Path(path)
    settings = yaml.safe_load(path.read_text())
    if settings.get("family") != "diffuseq":
        raise ValueError(f"{path} is not a DiffuSeq settings file (family: diffuseq)")
    settings["diffuseq_root"] = (path.parent / settings["diffuseq_root"]).resolve()
    return settings


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_directory(spec, offline):
    """The released checkpoint folder, downloaded once and verified against its pinned SHA-256.

    DiffuSeq publishes checkpoints as zip archives on Google Drive, not on the Hugging Face
    Hub. The archive is cached as ``<cache_dir>/<sha256>.zip`` and extracted next to it.
    ``offline`` follows local_files_only: True never downloads, "auto" only on a miss.
    """
    if offline not in (True, False, "auto"):
        raise ValueError('local_files_only must be true, false or "auto"')
    cache = Path(os.path.expanduser(spec["cache_dir"]))
    archive, extracted = cache / f"{spec['sha256']}.zip", cache / spec["sha256"]
    folder = extracted / spec["directory"]
    if (folder / spec["parameters"]).is_file() and offline is not False:
        return folder
    if offline is True:
        raise FileNotFoundError(f"DiffuSeq checkpoint not cached at {folder}")
    cache.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".part")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "elf-prompt-inversion"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)
    if _sha256(partial) != spec["sha256"]:
        partial.unlink()
        raise ValueError("Downloaded DiffuSeq checkpoint does not match the pinned sha256")
    partial.replace(archive)
    shutil.rmtree(extracted, ignore_errors=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(extracted)
    return folder


def load_tokenizer(path: str | Path = DEFAULT_SETTINGS, local_files_only=None):
    """The tokenizer upstream training saved with the checkpoint, without loading any weights."""
    from transformers import AutoTokenizer

    settings = read_settings(path)
    offline = settings["local_files_only"] if local_files_only is None else local_files_only
    return AutoTokenizer.from_pretrained(str(checkpoint_directory(settings["checkpoint"], offline)))


def build_model(upstream, bert_config, training_args):
    """The model and diffusion exactly as upstream basic_utils.create_model_and_diffusion builds them.

    Only difference: the BERT config is passed in (pinned) instead of being fetched by name,
    with upstream's own override hidden_dropout_prob = dropout.
    """
    a = training_args
    if a["learn_sigma"] or a["timestep_respacing"]:
        raise ValueError("Only learn_sigma=False checkpoints without respacing are supported")
    bert_config = copy.deepcopy(bert_config)
    bert_config.hidden_dropout_prob = a["dropout"]
    model = upstream["transformer_model"].TransformerNetModel(
        input_dims=a["hidden_dim"], output_dims=a["hidden_dim"], hidden_t_dim=a["hidden_t_dim"],
        dropout=a["dropout"], config=bert_config, vocab_size=a["vocab_size"], init_pretrained=a["use_plm_init"])
    gd = upstream["gaussian_diffusion"]
    steps = a["diffusion_steps"]
    diffusion = gd.SpacedDiffusion(
        use_timesteps=gd.space_timesteps(steps, [steps]),
        betas=gd.get_named_beta_schedule(a["noise_schedule"], steps),
        rescale_timesteps=a["rescale_timesteps"], predict_xstart=a["predict_xstart"],
        learn_sigmas=a["learn_sigma"], sigma_small=a["sigma_small"], use_kl=a["use_kl"],
        rescale_learned_sigmas=a["rescale_learned_sigmas"])
    return model.eval(), diffusion


def random_parameters(torch_model, setup: BackboneSetup):
    """Overwrite every parameter with N(0, init_std^2) draws; returns their provenance.

    Parameters are taken in sorted-name order and parameter i is drawn with
    fold_in(PRNGKey(init_seed), i), so the same init_seed always rebuilds the same model.
    """
    import torch

    if setup.weights != "random" or setup.init != "normal":
        raise ValueError('Random parameters need weights="random" and init="normal"')
    rng = jax.random.PRNGKey(setup.init_seed)
    named = dict(torch_model.named_parameters())  # lm_head.weight is tied to word_embedding.weight
    with torch.no_grad():
        for i, name in enumerate(sorted(named)):
            draw = setup.init_std * jax.random.normal(jax.random.fold_in(rng, i), tuple(named[name].shape), jnp.float32)
            named[name].copy_(torch.from_numpy(np.array(draw)))
    return {"weights": setup.weights, "bottleneck": setup.bottleneck, "init": setup.init,
            "init_std": setup.init_std, "init_seed": setup.init_seed}


@dataclass(frozen=True)
class PromptBatch:
    """Stage B layout: the prompt tokens alone; [CLS] and [SEP] are fixed by the format."""

    input_ids: jax.Array       # [batch, prompt_length]
    editable_mask: jax.Array   # [batch, prompt_length]; special tokens stay fixed
    prompt_length: int

    def with_prompt_ids(self, ids) -> PromptBatch:
        return replace(self, input_ids=jnp.asarray(ids),
                       editable_mask=jnp.repeat(self.editable_mask[:1], ids.shape[0], axis=0))


class DiffuSeqAdapter:
    """DiffuSeq as a VictimModel (models/interface.py) and a Stage B TokenModel."""

    family = "diffuseq"
    representations = (PROMPT_CODE,)

    def __init__(self, *, upstream, torch_model, diffusion, training_args, tokenizer, prompt_length,
                 provenance=None):
        self.upstream, self.torch_model, self.diffusion = upstream, torch_model.eval(), diffusion
        self.training_args, self.tokenizer = dict(training_args), tokenizer
        self.prompt_length, self.provenance = int(prompt_length), provenance or {}
        bert = torch_model.input_transformers.layer[0].attention.self
        self.seq_len = int(training_args["seq_len"])
        self.diffusion_steps = int(training_args["diffusion_steps"])
        # Upstream decodes with every diffusion step by default (run_decode.py --step 2000, ancestral
        # sampling); observations always use it. Time k / sampling_steps is the k-th model call.
        self.sampling_steps = self.diffusion_steps
        self.hidden_t_dim = int(training_args["hidden_t_dim"])
        self.embedding_dim = int(training_args["hidden_dim"])
        self.hidden_size = int(torch_model.hidden_size)
        self.n_layers = len(torch_model.input_transformers.layer)
        self.n_heads = int(bert.num_attention_heads)
        self.layer_norm_eps = float(torch_model.LayerNorm.eps)
        self.params = {name: jax.lax.stop_gradient(jnp.asarray(value.detach().cpu().numpy(), dtype=jnp.float32))
                       for name, value in torch_model.state_dict().items() if value.is_floating_point()}
        self.cls_token_id, self.sep_token_id = int(tokenizer.cls_token_id), int(tokenizer.sep_token_id)
        self.pad_token_id = int(tokenizer.pad_token_id)
        # alpha_bar of each timestep index, for the noise share of a later call.
        self._alphas_cumprod = jnp.asarray(diffusion.alphas_cumprod, dtype=jnp.float32)
        if self.embedding_matrix.shape[0] < len(tokenizer):
            raise ValueError("Tokenizer has more tokens than the embedding table")
        if not 0 < self.prompt_length <= self.max_prompt_length:
            raise ValueError(f"prompt_length must be from 1 to {self.max_prompt_length}")

    @classmethod
    def from_config(cls, path=DEFAULT_SETTINGS, *, local_files_only=None, setup: BackboneSetup = BackboneSetup()):
        """``setup`` picks the victim weights: released checkpoint (default) or random."""
        import torch
        from transformers import AutoConfig

        settings = read_settings(path)
        upstream = upstream_modules(settings["diffuseq_root"], settings["diffuseq_commit"])
        offline = settings["local_files_only"] if local_files_only is None else local_files_only
        spec, config_spec = settings["checkpoint"], settings["bert_config"]
        folder = checkpoint_directory(spec, offline)
        training_args = json.loads((folder / "training_args.json").read_text())
        if training_args["config_name"] != config_spec["repo"] or training_args["vocab"] != "bert":
            raise ValueError("Checkpoint BERT configuration differs from the model settings")
        bert_config = from_hub(lambda local: AutoConfig.from_pretrained(
            config_spec["repo"], revision=config_spec["revision"], local_files_only=local), offline)
        tokenizer = load_tokenizer(path, offline)  # saved by upstream training, in the same folder
        provenance = {"diffuseq_commit": settings["diffuseq_commit"], "checkpoint": spec,
                      "bert_config": config_spec, "backbone": asdict(setup)}
        if not setup.bottleneck:
            # Embeddings as wide as the encoder: upstream then builds no input_up_proj/output_down_proj.
            training_args = {**training_args, "hidden_dim": bert_config.hidden_size}
        model, diffusion = build_model(upstream, bert_config, training_args)
        if setup.weights == "checkpoint":
            parameters = folder / spec["parameters"]
            model.load_state_dict(torch.load(parameters, map_location="cpu", weights_only=True))
            provenance.update(checkpoint_path=str(parameters))
        else:
            provenance.update(random_parameters(model, setup))
        return cls(upstream=upstream, torch_model=model, diffusion=diffusion, training_args=training_args,
                   tokenizer=tokenizer, prompt_length=settings["prompt_length"], provenance=provenance)

    @property
    def max_prompt_length(self):
        """[CLS] prompt [SEP] [SEP] must leave at least one generated row."""
        return self.seq_len - 4

    def with_prompt_length(self, prompt_length: int) -> DiffuSeqAdapter:
        """Shallow copy for prompts of another length; weights are shared, not copied."""
        if not 0 < prompt_length <= self.max_prompt_length:
            raise ValueError(f"prompt_length must be from 1 to {self.max_prompt_length}")
        adapter = copy.copy(self)
        adapter.prompt_length = int(prompt_length)
        return adapter

    def noise_share(self, time):
        """sqrt(1 - alpha_bar) at the timestep index of sampler ``time`` (nearest index)."""
        index = jnp.clip(jnp.round((self.diffusion_steps - 1) - time * self.diffusion_steps), 0,
                         self.diffusion_steps - 1).astype(jnp.int32)
        return jnp.sqrt(1.0 - self._alphas_cumprod[index])

    # --- VictimModel ---

    def metadata(self, prompt_length, observation_layer, *, step=0, sampling_steps=None):
        """DiffuSeq's time of sampler call ``step`` is ``step / sampling_steps``: timestep index T - 1 - step."""
        time = step_time(step, self.sampling_steps if sampling_steps is None else sampling_steps)
        return ForwardMetadata(prompt_length, observation_layer, time=time,
                               forward_mode="sampling_pass" if step else "first_sampling_pass")

    def forward_map(self, metadata: ForwardMetadata, knowledge: AttackerKnowledge = AttackerKnowledge()):
        return BackboneForward(self, metadata, knowledge)

    def observe(self, tokens, metadata: ForwardMetadata, *, noise_seed, sampling_steps):
        from .observe import observe  # imports elf_inversion.observation, which is model-independent
        return observe(self, tokens, metadata, noise_seed=noise_seed, sampling_steps=sampling_steps)

    def source_ids(self, tokens):
        """Upstream's full input ids: [CLS] prompt [SEP] [SEP], then padding; and its input_mask."""
        tokens = np.asarray(tokens).reshape(-1)
        ids = np.full(self.seq_len, self.pad_token_id, dtype=np.int64)
        n = len(tokens)
        ids[0], ids[1:n + 1], ids[n + 1:n + 3] = self.cls_token_id, tokens, self.sep_token_id
        mask = np.ones(self.seq_len, dtype=np.int64)
        mask[:n + 3] = 0
        return ids, mask

    # --- Stage B TokenModel ---

    @property
    def embedding_matrix(self):
        return self.params["word_embedding.weight"]

    @property
    def embedding_table(self):
        return self.embedding_matrix

    def prepare_token_ids(self, prompt_ids) -> PromptBatch:
        ids = np.asarray(prompt_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] != self.prompt_length:
            raise ValueError(f"Expected [batch, {self.prompt_length}] prompt IDs")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Token IDs must be integers")
        if np.any(ids < 0) or np.any(ids >= len(self.tokenizer)):
            raise ValueError("Token ID outside tokenizer vocabulary")
        editable = ~np.isin(ids, self.tokenizer.all_special_ids)
        return PromptBatch(jnp.asarray(ids, dtype=jnp.int32), jnp.asarray(editable), self.prompt_length)

    def input_embeddings(self, batch: PromptBatch):
        return self.embedding_table[batch.input_ids]

    def encode_embeddings(self, embeddings, batch: PromptBatch, representation: str = PROMPT_CODE):
        """The prompt rows of the model input; differentiable in editable embeddings only."""
        if representation != PROMPT_CODE:
            raise ValueError(f"Unknown representation {representation!r}; DiffuSeq offers {list(self.representations)}")
        if embeddings.shape != (*batch.input_ids.shape, self.embedding_dim):
            raise ValueError("Supplied embeddings must have shape [batch, prompt_length, embedding_dim]")
        return jnp.where(batch.editable_mask[..., None], embeddings, self.input_embeddings(batch))

    def encode_token_ids(self, batch: PromptBatch, representation: str = PROMPT_CODE):
        return self.encode_embeddings(self.input_embeddings(batch), batch, representation)

    def target_dimension(self, representation) -> int:
        return self.embedding_dim
