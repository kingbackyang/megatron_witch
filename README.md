# Megatron Witch

Build a large model from scratch: a hexagon warrior that unites algorithms and infrastructure.  
Not a single-point breakthrough, but an end-to-end system challenge: model structure, parallel strategies, and training/inference engineering all connected.

## Vision
- **Algorithmic depth**: from vanilla Transformers to MoE and parallel training strategies, build a solid algorithm stack.
- **Engineering strength**: reusable, scalable, and maintainable training and inference infrastructure.
- **System mindset**: make research and engineering resonate, turning experiments into runnable systems.

## Structure (current plan)
- `native_transformers/`  
  Start with PyTorch-native implementations; keep models readable, hackable, and experiment-friendly.
- `megatron_transformers/`  
  Explore Megatron-style parallel training model/runner structures.
- `moe_megatron/`  
  MoE (Mixture-of-Experts) architectures and parallel strategy exploration.
- `fused_transformers/`  
  Performance and throughput experiments with fused operators/engineering optimizations.
- `mini_megatron_witch/`  
  A minimal runnable version for fast validation and teaching demos.
- `main.py`  
  Entry/experiment script (placeholder for now, will be replaced with real flows).

## Milestones (planned)
1. **Minimal runnable training loop**: data → model → train → validate.
2. **Reproducible experiment scripts**: stable runs across different model/parallel setups.
3. **MoE + parallel training fusion**: verify feasibility at a controlled scale.
4. **Inference and deployment path**: measurable and usable inference baseline.

## Style and principles
- **From scratch**: fewer black boxes, own the critical path.
- **Readability first**: keep core implementations straightforward; add heavy optimizations later.
- **Extensible**: each component can be swapped or upgraded without lock-in.

## Usage
This project is under rapid construction. Once the first batch of runnable scripts is stable, it will include:
- Environment dependencies and setup
- Minimal training/inference examples
- Experiment reproduction steps

## Contributing
Discussions and contributions are welcome, including:
- Code structure and engineering suggestions
- Parallel training strategy and performance optimizations
- Experiment logs and reproduction scripts

---
"Building large models is not just algorithms, and not just engineering, but both taken to the extreme."
