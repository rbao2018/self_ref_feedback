import gc

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.logging import init_logger

import vllm
from vllm.distributed.parallel_state import destroy_model_parallel

logger = init_logger(__name__)


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):

        assert vllm.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.4.1":
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = "openrlhf.trainer.ray.vllm_worker_wrap"
                    kwargs["worker_class_name"] = "WorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)
        self.prompt_token_ids = []
        self.output = []

    def get_output(self, index, total_group):
        avg_length = len(self.output) // total_group
        remainder = len(self.output) % total_group
        start_index = avg_length * index + min(index, remainder)
        end_index = start_index + avg_length + (1 if index < remainder else 0)
        return self.output[start_index:end_index]

    def add_request(self, prompt_token_ids):
        self.prompt_token_ids.extend(prompt_token_ids)
        return len(self.prompt_token_ids)

    def generate(self, **kwargs):
        self.output = self.llm.generate(prompt_token_ids=self.prompt_token_ids, **kwargs)
        self.prompt_token_ids = []
        return len(self.output)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def destroy(self):
        if self.llm is not None:
            destroy_model_parallel()
            del self.llm.llm_engine.model_executor.driver_worker
            del self.llm  # Isn't necessary for releasing memory, but why not
            gc.collect()
            torch.cuda.empty_cache()
        self.llm = None


def create_vllm_engines(num_engines: int, tensor_parallel_size: int, pretrain: str, seed: int):
    vllm_engines = []
    for _ in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 8}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, 
                placement_group_capture_child_tasks=True, 
                placement_group_bundle_index=0
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=8,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size
            )
        )

    return vllm_engines


if __name__ == "__main__":
    llm = LLMRayActor.remote(
        "/root/Llama-2-7b-chat-hf", 
        tensor_parallel_size=1
        )
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
