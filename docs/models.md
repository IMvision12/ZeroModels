# Models

Every model here is a pure-Keras 3 port with weights converted from the original checkpoints. Pages are grouped by modality and listed alphabetically, and each one follows the same shape: API, preprocessing, variants, a runnable example with its measured output, data format, and how to load fine-tuned weights.

The shared machinery is documented separately, in [Main Classes](main_classes.md) and [Configuration](configuration.md).

## Text models

Encoders for embeddings and masked language modelling, and decoder LLMs, dense and mixture-of-experts.

- [BERT](bert.md)
- [DeBERTa](deberta.md)
- [DeepSeek-V2](deepseek_v2.md)
- [DeepSeek-V3](deepseek_v3.md)
- [DeepSeek-V4](deepseek_v4.md)
- [Gemma](gemma.md)
- [Gemma 2](gemma2.md)
- [GLM](glm.md)
- [GLM-4](glm4.md)
- [GLM-4 MoE](glm4_moe.md)
- [GLM-5 MoE](glm5_moe.md)
- [GPT](gpt.md)
- [GPT-2](gpt2.md)
- [GPT-OSS](gpt_oss.md)
- [Llama](llama.md)
- [Llama 2](llama2.md)
- [Llama 4](llama4.md)
- [MiniMax](minimax.md)
- [MiniMax M2](minimax_m2.md)
- [Mistral](mistral.md)
- [Mixtral](mixtral.md)
- [Qwen](qwen.md)
- [Qwen2](qwen2.md)
- [Qwen2 MoE](qwen2_moe.md)
- [Qwen3](qwen3.md)
- [Qwen3 MoE](qwen3_moe.md)
- [Qwen3.5](qwen3_5.md)
- [Qwen3-Next](qwen3_next.md)
- [RoBERTa](roberta.md)
- [XLM-RoBERTa](xlm_roberta.md)

## Vision models

Detection, segmentation, monocular depth, and self-supervised backbones.

- [Classification backbones](classification_backbones.md)
- [D-FINE](dfine.md)
- [DeepLabV3](deeplabv3.md)
- [Depth Anything V1](depth_anything_v1.md)
- [Depth Anything V2](depth_anything_v2.md)
- [DETR](detr.md)
- [DINO](dino.md)
- [DINOv2](dinov2.md)
- [DINOv3](dinov3.md)
- [EoMT](eomt.md)
- [Mask2Former](mask2former.md)
- [MaskFormer](maskformer.md)
- [MobileViT](mobilevit.md)
- [MobileViTV2](mobilevitv2.md)
- [RF-DETR](rf_detr.md)
- [RT-DETR](rt_detr.md)
- [RT-DETRv2](rt_detr_v2.md)
- [SAM](sam.md)
- [SAM 2](sam2.md)
- [SegFormer](segformer.md)

## Audio models

Speech recognition, and speech-aware language models that take audio and text together.

- [Granite Speech](granite_speech.md)
- [Granite Speech Plus](granite_speech_plus.md)
- [Moonshine](moonshine.md)
- [Speech2Text](speech2text.md)
- [Whisper](whisper.md)

## Multimodal models

Vision-language encoders, generative VLMs, grounding across detection, OCR, pointing and referring, and text-to-image diffusion.

- [CLIP](clip.md)
- [DeepSeek-VL](deepseek_vl.md)
- [DeepSeek-VL Hybrid](deepseek_vl_hybrid.md)
- [Gemma 3](gemma3.md)
- [Gemma 4](gemma4.md)
- [Gemma 4 Unified](gemma4_unified.md)
- [GLM-4V](glm4v.md)
- [GLM-4V MoE](glm4v_moe.md)
- [Grounding DINO](grounding_dino.md)
- [InternVL](internvl.md)
- [Janus](janus.md)
- [Kimi K2.5](kimi_k25.md)
- [LocateAnything](locateanything.md)
- [MetaCLIP 2](metaclip2.md)
- [MiniMax M3-VL](minimax_m3_vl.md)
- [Mistral 3](mistral3.md)
- [OneFormer](oneformer.md)
- [OWL-ViT](owlvit.md)
- [OWLv2](owlv2.md)
- [Qwen2-VL](qwen2_vl.md)
- [Qwen2.5-VL](qwen2_5_vl.md)
- [Qwen3-VL](qwen3_vl.md)
- [Qwen3-VL-MoE](qwen3_vl_moe.md)
- [Qwen3.5-MoE](qwen3_5_moe.md)
- [SAM 3](sam3.md)
- [SigLIP](siglip.md)
- [SigLIP 2](siglip2.md)
- [Stable Diffusion](stable_diffusion.md)
