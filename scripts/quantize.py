"""Web model layout for Transformers.js: fp32 encoder + int8 (q8) merged decoder."""

import shutil
import subprocess
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

EXPORT_DIR = Path("out/onnx-fp32")
WEB_MODEL_DIR = Path("web/model")
README_PATH = Path("out/README.md")

# Quantizing the encoder cost ~7 points exact match (0.797 vs 0.863 on test[:300]) and swapped rare
# words ("Einblick" → "Überblick"). In the decoder, per-tensor q8 flipped rare participles
# ("ausgereicht" → "ausgebrochen") and even per-channel q8 on the output projection did
# ("beschlagnahmt" → "beschlaggenommen"). Keeping only lm_head fp32 matched the fp32 decoder at
# 109 MB instead of 233 MB (round 2, browser WASM, test[:300]: 0.873 exact, regression set 11/12).
KEEP_FP32_NODES = ["/lm_head/MatMul"]
QUANTIZE_JOBS = {"decoder_model_merged.onnx": "decoder_model_merged_quantized.onnx"}
FP32_FILES = ("encoder_model.onnx",)
ROOT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)


def main():
    # optimum-cli writes the .onnx files to the export root, next to config.json
    export_onnx = EXPORT_DIR
    if not (export_onnx / "decoder_model_merged.onnx").exists():
        raise FileNotFoundError(
            f"{export_onnx / 'decoder_model_merged.onnx'} is missing; "
            "rerun the export without --no-post-process so the decoder gets merged."
        )

    out_onnx = WEB_MODEL_DIR / "onnx"
    # stale files from an earlier layout would still be uploaded by the site build
    shutil.rmtree(out_onnx, ignore_errors=True)
    out_onnx.mkdir(parents=True)
    for source_name, target_name in QUANTIZE_JOBS.items():
        # The merged decoder keeps its weights inside If-node branches (with/without past); without
        # EnableSubgraph those are skipped and the decoder stays fp32 (233 MB instead of ~60 MB).
        quantize_dynamic(
            str(export_onnx / source_name),
            str(out_onnx / target_name),
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=KEEP_FP32_NODES,
            extra_options={"EnableSubgraph": True},
        )
        print(f"wrote {out_onnx / target_name}")
    for name in FP32_FILES:
        shutil.copy2(export_onnx / name, out_onnx / name)
        print(f"wrote {out_onnx / name}")

    for name in ROOT_FILES:
        shutil.copy2(EXPORT_DIR / name, WEB_MODEL_DIR / name)

    size = subprocess.run(["du", "-sh", str(WEB_MODEL_DIR)], capture_output=True, text=True, check=True).stdout.split()[0]
    line = f"web/model size: {size}"
    print(line)
    with README_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
