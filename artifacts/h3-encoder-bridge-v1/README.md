# Tiny native Encoder -> H3 bridge v1

Status: host validation passed; Hygon device validation PENDING.
Requires the already restored competition runtime and the tested MHA singleton / Qwen vision frame-boundary fixes. No framework rebuild or package installation is performed.

Upload this ZIP to /root/private_data/scripts, then in the compute Pod:

```bash
source /root/private_data/setup_env.sh &&
python -m zipfile -e "$WORK_HOME/scripts/h3-encoder-bridge-v1.zip" "$WORK_HOME/scripts" &&
cd "$WORK_HOME/scripts/h3-encoder-bridge-v1" &&
run_logged h3_encoder_bridge_package_check.log sha256sum -c SHA256SUMS &&
run_logged h3_encoder_bridge.log python -X faulthandler -u verify_h3_encoder_bridge.py
```

Three dtype model pairs; text7/image11/video9/text5 requests per pair; 12 native chained forwards. Reports, tiny synthetic checkpoints, inputs and output arrays go to LOG_DIR/h3_encoder_bridge_<timestamp>_<pid>/.

Encoder GPU output is passed into native H3 via InfiniCore; reference tensors never replace device conditions. Synthetic H3 text_dim=64 matches the complete tiny Encoder width. CPU fixture cache generation is outside this device-core test.

This does not validate official weights, the production 50-layer Encoder, processor/pipeline integration, production rotary cache generation, diffusion iterations, VAE, full videos, multi-device execution, or performance. Retain failed reports and arrays; do not relax thresholds to obtain a pass. A native crash may leave status=running: require the final PASS, all12 runs, and exit code0.

Fixture sensitivity: synthetic attention residual gate bias is offset by +1.0 for all three modalities; model equations, production weights and comparison tolerances are unchanged. Every case must reject zero and inverted conditions in both outputs under the same dtype tolerance (48 CPU mutation comparisons).
