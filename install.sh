pip install vllm==0.4.2
pip uninstall flash-attn -y
MAX_JOBS=32 pip install flash-attn==2.6.1 --no-build-isolation
pip install openrlhf==0.3.7