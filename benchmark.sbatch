#!/bin/bash
#SBATCH --job-name=kv_bench
#SBATCH --output=slurm-%j.log
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --exclude=gpu01.slurm.cpe.kmutt.ac.th
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /etc/profile
set -euo pipefail

echo "========================================="
echo "=== Host & Execution Environment      ==="
echo "========================================="
hostname
date
uname -a

echo "========================================="
echo "=== Load Cluster CUDA Module          ==="
echo "========================================="
module purge
module load cuda/12.2
module list

echo "========================================="
echo "=== GPU Information                   ==="
echo "========================================="
nvidia-smi

echo "========================================="
echo "=== Toolchain & uv Setup             ==="
echo "========================================="
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
uv sync --frozen

echo ">> Running Automated Test Suite on Compute Node..."
uv run --no-sync python -m unittest discover tests

echo "========================================="
echo "=== KV-Bench Suite (Capped at 16GB)   ==="
echo "========================================="
# The physical GPU is an RTX 4090 (24GB VRAM).
# --max-vram-gb 16.0 enforces the consumer hardware ceiling (both in PyTorch's
# allocator and vLLM's KV allocation pool) so results strictly mirror 16GB VRAM.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default to primary pure dense model (qwen3-4b); accepts aliases (phi4-mini, qwen3.5-4b, etc.)
MODEL="${1:-qwen3-4b}"
TECHNIQUE="${TECHNIQUE:-flexgen_kv}"
ENGINE="${ENGINE:-pytorch}"
RUN_FULL_SWEEP="${RUN_FULL_SWEEP:-1}"
SKIP_HARDWARE_SWEEPS="${SKIP_HARDWARE_SWEEPS:-0}"
CONTEXTS="512 1024 2048 4096 8192 16384 32768"
CONCURRENT_CONTEXTS="1024 2048 4096 8192"
CONCURRENT_USERS="1 4 8 12 16"
EVAL_CONTEXTS="512 1024 2048 4096 8192 16384"
NIAH_CONTEXTS="2048 4096"
FORCE_LONGMEM_REF="${FORCE_LONGMEM_REF:-1}"

echo "Model Selected:     $MODEL"
echo "Technique Selected: $TECHNIQUE"
echo "Engine Selected:    $ENGINE"

# ==============================================================================
# PyTorch Scientific Control Pipeline (WP1-4-1 Diagnostic Testbed)
# ==============================================================================
if [[ "$ENGINE" == "pytorch" ]]; then
    # --------------------------------------------------------------------------
    # 1. Baseline PyTorch Reference Generation (if missing or technique is baseline)
    # --------------------------------------------------------------------------
    BASELINE_MACRO_REF="results/references/baseline_macro_pytorch_ref.json"
    BASELINE_LONGMEM_REF="results/references/baseline_${MODEL}_pytorch_ref.json"
    VERIFY_REF="results/references/verify_ref_pytorch.json"

    if [[ "$TECHNIQUE" == "baseline" || ! -f "$VERIFY_REF" ]]; then
        echo ">> [PyTorch Baseline] Step 1: Verification & Verify Reference Capture..."
        uv run --no-sync main.py \
            --technique baseline \
            --engine pytorch \
            --model "$MODEL" \
            --verify \
            --max-vram-gb 16.0 \
            --save-reference "$VERIFY_REF" \
            --output results/verify_baseline_pytorch_16gb.json
    fi

    if [[ "$RUN_FULL_SWEEP" == "1" ]]; then
        if [[ ! -f "$BASELINE_MACRO_REF" ]]; then
            echo ">> [PyTorch Baseline] Step 2: Context Scaling Sweep (512 - 32768) & Macro Reference Capture..."
            uv run --no-sync main.py \
                --technique baseline \
                --engine pytorch \
                --model "$MODEL" \
                --contexts $CONTEXTS \
                --max-vram-gb 16.0 \
                --save-reference "$BASELINE_MACRO_REF" \
                --output results/baseline_pytorch_16gb.json
        fi

        if [[ "$TECHNIQUE" == "baseline" || ! -f "$BASELINE_LONGMEM_REF" || "$FORCE_LONGMEM_REF" == "1" ]]; then
            echo ">> [PyTorch Baseline] Step 3: LongMemEval Adherence & Reference Capture (512 - 16384, Select by Length)..."
            uv run --no-sync main.py \
                --technique baseline \
                --engine pytorch \
                --model "$MODEL" \
                --workload longmem \
                --select-by-length \
                --contexts $EVAL_CONTEXTS \
                --max-tokens 32 \
                --samples-per-task 10 \
                --seed 42 \
                --max-vram-gb 16.0 \
                --save-reference "$BASELINE_LONGMEM_REF" \
                --output results/longmem_baseline_pytorch_16gb.json
        fi
    fi

    if [[ "$TECHNIQUE" == "baseline" ]]; then
        echo "=== PyTorch Baseline Benchmark Run Completed Successfully ==="
        exit 0
    fi

    # --------------------------------------------------------------------------
    # 2. Technique Evaluation under PyTorch (e.g. H2O and StreamingLLM)
    # --------------------------------------------------------------------------
    TECH_LIST=("h2o" "streamingllm")
    if [[ "$TECHNIQUE" != "all" && "$TECHNIQUE" != "baseline" ]]; then
        TECH_LIST=("$TECHNIQUE")
    fi

    for CURRENT_TECH in "${TECH_LIST[@]}"; do
        echo ">> [PyTorch Technique: $CURRENT_TECH] Step 0: Verification (128 ctx, 16 tokens)..."
        uv run --no-sync main.py \
            --technique "$CURRENT_TECH" \
            --engine pytorch \
            --model "$MODEL" \
            --verify \
            --max-vram-gb 16.0 \
            --eval-reference "$VERIFY_REF" \
            --output "results/verify_${CURRENT_TECH}_pytorch_16gb.json"

        if [[ "$RUN_FULL_SWEEP" == "1" ]]; then
            if [[ "$SKIP_HARDWARE_SWEEPS" != "1" ]]; then
                echo ">> [PyTorch Technique: $CURRENT_TECH] Step 1: Scaling Sweep (512 - 32768) evaluated against Macro Reference..."
                uv run --no-sync main.py \
                    --technique "$CURRENT_TECH" \
                    --engine pytorch \
                    --model "$MODEL" \
                    --contexts $CONTEXTS \
                    --max-vram-gb 16.0 \
                    --profile \
                    --eval-reference "$BASELINE_MACRO_REF" \
                    --output "results/${CURRENT_TECH}_hardware_pytorch_16gb.json"
            fi

            echo ">> [PyTorch Technique: $CURRENT_TECH] Step 2: Diagnostic LongMemEval Adherence & Causal Layer Retention (512 - 16384, Select by Length)..."
            uv run --no-sync main.py \
                --technique "$CURRENT_TECH" \
                --engine pytorch \
                --model "$MODEL" \
                --workload longmem \
                --select-by-length \
                --contexts $EVAL_CONTEXTS \
                --max-tokens 32 \
                --samples-per-task 10 \
                --seed 42 \
                --max-vram-gb 16.0 \
                --eval-reference "$BASELINE_LONGMEM_REF" \
                --output "results/${CURRENT_TECH}_longmem_pytorch_16gb.json"

            echo ">> [PyTorch Technique: $CURRENT_TECH] Step 3: Opt-In Mechanistic 2D NIAH Sweep..."
            uv run --no-sync main.py \
                --technique "$CURRENT_TECH" \
                --engine pytorch \
                --model "$MODEL" \
                --workload niah \
                --contexts $NIAH_CONTEXTS \
                --max-tokens 16 \
                --max-vram-gb 16.0 \
                --output "results/${CURRENT_TECH}_niah_pytorch_16gb.json"
        fi

        echo "=== PyTorch Benchmark Pipeline for $CURRENT_TECH Completed Successfully ==="
    done
    exit 0
fi

if [[ "$TECHNIQUE" == "baseline" ]]; then
    echo ">> 0. Running Fast Verification & Reference Capture (128 ctx, 16 tokens, 16.0 GB VRAM cap)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model "$MODEL" \
        --verify \
        --max-vram-gb 16.0 \
        --save-reference results/references/verify_ref.json \
        --output results/verify_16gb.json

    echo ">> 0a. Running Identity Quality Check (evaluating against reference, 128 ctx)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model "$MODEL" \
        --verify \
        --max-vram-gb 16.0 \
        --eval-reference results/references/verify_ref.json \
        --output results/verify_identity_16gb.json

    echo ">> 0b. Running Balanced LongMemEval Sanity Check (1 sample per task, seed 42, 512 ctx)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model "$MODEL" \
        --workload longmem \
        --dataset-path src/data/longmemeval_subset.json \
        --contexts 512 \
        --max-tokens 32 \
        --samples-per-task 1 \
        --seed 42 \
        --max-vram-gb 16.0 \
        --output results/sanity_longmem_16gb.json

    echo ">> 0c. Running Cross-Model Divergence Calibration (evaluating qwen3-1.7b against qwen3-4b reference)..."
    uv run --no-sync main.py \
        --technique baseline \
        --model qwen3-1.7b \
        --verify \
        --max-vram-gb 16.0 \
        --eval-reference results/references/verify_ref.json \
        --output results/verify_cross_model_16gb.json

    if [[ "$RUN_FULL_SWEEP" == "1" ]]; then
        if [[ "$SKIP_HARDWARE_SWEEPS" != "1" ]]; then
            echo ">> 1. Running Baseline Single-Stream Scaling Sweep (512 - 65536)..."
            uv run --no-sync main.py \
                --technique baseline \
                --model "$MODEL" \
                --contexts 512 1024 2048 4096 8192 16384 32768 49152 65536 \
                --max-vram-gb 16.0 \
                --output results/baseline_hardware_16gb.json
            cp results/baseline_hardware_16gb.json results/baseline_16gb.json

            echo ">> 2. Running Baseline Multi-User Concurrency Grid (1, 4, 8, 12, 16 users across 1024-8192 ctx)..."
            for U in $CONCURRENT_USERS; do
                echo "   -> Running concurrency sweep for $U concurrent user(s)..."
                uv run --no-sync main.py \
                    --technique baseline \
                    --model "$MODEL" \
                    --workload concurrent \
                    --num-users "$U" \
                    --contexts $CONCURRENT_CONTEXTS \
                    --max-vram-gb 16.0 \
                    --output "results/concurrent_${U}users_16gb.json"
            done
            uv run --no-sync python scripts/consolidate_concurrency_results.py \
                --inputs results/concurrent_*users_16gb.json \
                --output results/concurrent_baseline_16gb.json
        else
            echo ">> Skipping Steps 1 & 2 (Hardware sweeps already verified and consolidated)."
        fi

        echo ">> 3. Running Baseline Long-Term Memory Adherence & Reference Capture (LongMemEval across contexts)..."
        uv run --no-sync main.py \
            --technique baseline \
            --model "$MODEL" \
            --workload longmem \
            --contexts $EVAL_CONTEXTS \
            --max-tokens 32 \
            --samples-per-task 10 \
            --seed 42 \
            --max-vram-gb 16.0 \
            --save-reference "results/references/baseline_${MODEL}_ref.json" \
            --output results/longmem_baseline_16gb.json
    fi
else
    VERIFY_REF="results/references/verify_ref.json"
    SUFFIX=""
    if [[ "$ENGINE" == "pytorch" ]]; then
        SUFFIX="_pytorch"
        if [[ -f "results/references/verify_ref_pytorch.json" ]]; then
            VERIFY_REF="results/references/verify_ref_pytorch.json"
        fi
    fi

    echo ">> 0. Running Technique Verification (${TECHNIQUE}, engine: ${ENGINE}, 128 ctx, 16 tokens)..."
    uv run --no-sync main.py \
        --technique "$TECHNIQUE" \
        --engine "$ENGINE" \
        --model "$MODEL" \
        --verify \
        --max-vram-gb 16.0 \
        --eval-reference "$VERIFY_REF" \
        --output "results/verify_${TECHNIQUE}${SUFFIX}_16gb.json"

    if [[ "$RUN_FULL_SWEEP" == "1" ]]; then
        if [[ "$SKIP_HARDWARE_SWEEPS" != "1" ]]; then
            MACRO_REF=""
            if [[ -f "results/references/baseline_macro_${ENGINE}_ref.json" ]]; then
                MACRO_REF="--eval-reference results/references/baseline_macro_${ENGINE}_ref.json"
            fi
            echo ">> 1. Running Technique Single-Stream Scaling Sweep (${TECHNIQUE}, engine: ${ENGINE}, 512 - 32768, with profiling)..."
            uv run --no-sync main.py \
                --technique "$TECHNIQUE" \
                --engine "$ENGINE" \
                --model "$MODEL" \
                --contexts $CONTEXTS \
                --max-vram-gb 16.0 \
                --profile \
                $MACRO_REF \
                --output "results/${TECHNIQUE}_hardware${SUFFIX}_16gb.json"
        fi

        LONGMEM_REF="results/references/baseline_${MODEL}_ref.json"
        if [[ "$ENGINE" == "pytorch" && -f "results/references/baseline_${MODEL}_pytorch_ref.json" ]]; then
            LONGMEM_REF="results/references/baseline_${MODEL}_pytorch_ref.json"
        fi

        echo ">> 2. Running Macro LongMemEval Adherence & Quality Evaluation (${TECHNIQUE}, engine: ${ENGINE} vs baseline reference)..."
        uv run --no-sync main.py \
            --technique "$TECHNIQUE" \
            --engine "$ENGINE" \
            --model "$MODEL" \
            --workload longmem \
            --contexts $EVAL_CONTEXTS \
            --max-tokens 32 \
            --samples-per-task 10 \
            --seed 42 \
            --max-vram-gb 16.0 \
            --eval-reference "$LONGMEM_REF" \
            --output "results/${TECHNIQUE}_longmem${SUFFIX}_16gb.json"
    fi
fi

echo "========================================="
echo "=== Benchmarks Completed Successfully ==="
echo "========================================="
date
