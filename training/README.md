# SmolVLA training environment

Do not install ML dependencies into the ROS 2 Jazzy Python environment. Current
official LeRobot 0.6.1 requires Python 3.12 and PyTorch 2.7–2.11. Create a
separate environment on a GPU-capable machine, then install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-vla.txt
```

SmolVLA (`lerobot/smolvla_base`) is a 450M base model and requires fine-tuning
on a compatible custom UR3e dataset before it can be evaluated here. The
native API predicts action chunks of shape `(batch, chunk_size, action_dim)`;
NS-VLAS accepts only a checkpoint trained with the documented 8-D state,
`observation.images.vla_camera`, and 7-D project action schema.

No training or checkpoint download was attempted on the 8 GB development PC.
