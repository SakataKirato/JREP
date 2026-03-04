# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Colab GPU Chat UI (HF形式そのまま)
#
# - GGUF不要で、Colab GPU上でチャット推論を行うための最小構成です。
# - `MODEL_SOURCE` を切り替えることで、以下の3パターンに対応します。
#   - `base`: ベースモデル
#   - `merged`: すでにマージ済みモデル
#   - `adapter_merge`: ベース + LoRAアダプタをその場でマージ

# %% id="install_deps"
# 必要なときだけ True にして実行してください（実行後はランタイム再起動）
# vLLMチャット用途の最小セット（StructEval依存なし）
RUN_INSTALL = True

if RUN_INSTALL:
    import subprocess

    cmds = [
        # 競合しやすい主要パッケージを先に外してから最小セットを入れる
        'pip uninstall -y protobuf huggingface_hub transformers tokenizers vllm',
        'pip install --no-cache-dir "protobuf==5.29.3"',
        'pip install --no-cache-dir '
        '"torch==2.9.0" '
        '"triton==3.5.0" '
        '"transformers==4.57.6" '
        '"huggingface-hub==0.36.2" '
        '"tokenizers==0.22.1" '
        '"vllm==0.13.0" '
        '"gradio>=4.0.0" '
        '"accelerate" '
        '"peft"',
        'python3 -c "import google.protobuf, vllm, transformers, huggingface_hub, gradio; '
        "print('protobuf', google.protobuf.__version__); "
        "print('vllm', vllm.__version__); "
        "print('transformers', transformers.__version__); "
        "print('huggingface_hub', huggingface_hub.__version__)" '"',
    ]
    for c in cmds:
        subprocess.check_call(c, shell=True)
    print("✅ setup finished. Please restart runtime now.")

# %% id="config"
import os

# -----------------------------
# Config
# -----------------------------
MODEL_SOURCE = "adapter_merge"  # "base" | "merged" | "adapter_merge"

# HF形式モデルID
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MERGED_MODEL_ID = "your_id/your_merged_model"
ADAPTER_ID = "your_id/your_lora_adapter"

# 推論設定
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0

# merge一時保存先（adapter_merge時）
MERGED_LOCAL_DIR = "/content/merged_for_chat"

# %% id="hf_login"
from huggingface_hub import login

# 必要なときだけ実行
# login()

# %% id="load_model"
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import google.protobuf
import torch


def resolve_model_path_and_tokenizer():
    if MODEL_SOURCE == "base":
        model_id = BASE_MODEL_ID
        print(f"[INFO] Using base model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model_id, tokenizer

    if MODEL_SOURCE == "merged":
        model_id = MERGED_MODEL_ID
        print(f"[INFO] Using merged model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return model_id, tokenizer

    if MODEL_SOURCE == "adapter_merge":
        import torch
        from peft import PeftModel

        print(f"[INFO] Loading base model for merge: {BASE_MODEL_ID}")
        # vLLM初期化時のGPU競合を避けるため、マージはCPU上で実行
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

        print(f"[INFO] Loading adapter: {ADAPTER_ID}")
        lora_model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

        print("[INFO] Merging adapter...")
        merged_model = lora_model.merge_and_unload()

        os.makedirs(MERGED_LOCAL_DIR, exist_ok=True)
        merged_model.save_pretrained(MERGED_LOCAL_DIR)
        tokenizer.save_pretrained(MERGED_LOCAL_DIR)

        del base_model, lora_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[INFO] Merged model saved to: {MERGED_LOCAL_DIR}")
        return MERGED_LOCAL_DIR, tokenizer

    raise ValueError("MODEL_SOURCE must be one of: base, merged, adapter_merge")


model_path, tokenizer = resolve_model_path_and_tokenizer()
print(f"[INFO] Model path resolved: {model_path}")

# 本番環境合わせ: protobuf 5.29.3 を期待
pb_ver = google.protobuf.__version__
if pb_ver != "5.29.3":
    raise RuntimeError(
        f"Incompatible protobuf version: {pb_ver}. "
        "Please run install cell and restart runtime."
    )

# vLLMはimport前に環境変数を設定
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
# Colabで `UnsupportedOperation: fileno` を踏みやすい設定を明示的に上書き
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

from vllm import LLM, SamplingParams

# tokenizerはベース側を明示（mergedフォルダのtokenizer由来不整合を回避）
tokenizer_path_for_vllm = BASE_MODEL_ID if MODEL_SOURCE == "adapter_merge" else model_path

# vLLM起動フォールバック（OOM/初期化失敗を段階的に回避）
_try_cfgs = [
    {"max_model_len": 4096, "gpu_memory_utilization": 0.85},
    {"max_model_len": 3072, "gpu_memory_utilization": 0.80},
    {"max_model_len": 2048, "gpu_memory_utilization": 0.72},
]
llm = None
vllm_available = False
fallback_transformers_model = None
last_err = None
for i, cfg in enumerate(_try_cfgs, 1):
    try:
        print(
            f"[INFO] vLLM init try {i}/{len(_try_cfgs)} "
            f"(max_model_len={cfg['max_model_len']}, gpu_mem={cfg['gpu_memory_utilization']})"
        )
        llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path_for_vllm,
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=cfg["max_model_len"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            disable_log_stats=True,
        )
        print("[INFO] vLLM loaded.")
        vllm_available = True
        break
    except Exception as e:
        last_err = e
        print(f"[WARN] vLLM init failed on try {i}: {type(e).__name__}: {e}")

if llm is None:
    # Colab環境で起きる UnsupportedOperation: fileno などを救済
    print(f"[WARN] vLLM unavailable. Falling back to Transformers. last_error={last_err}")
    fallback_transformers_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(
        "[INFO] Transformers fallback model loaded on device:",
        getattr(fallback_transformers_model, "device", "unknown"),
    )

if torch.cuda.is_available():
    print("[INFO] CUDA is available:", torch.cuda.get_device_name(0))
else:
    print("[WARN] CUDA is NOT available. Running on CPU.")

# %% id="chat_ui"
import gradio as gr


def chat_fn(message, history):
    messages = []
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if vllm_available:
        sampling = SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        outs = llm.generate([prompt], sampling)
        text = outs[0].outputs[0].text.strip() if outs and outs[0].outputs else ""
    else:
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(fallback_transformers_model.device)
        with torch.no_grad():
            outputs = fallback_transformers_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0.0),
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Colab GPU Chat (HF model)",
    description="GGUF不要。Colab GPU上で直接チャット推論。",
)

# share=True で外部URL発行
demo.launch(share=True, debug=False)
