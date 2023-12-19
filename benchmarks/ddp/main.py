# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
import uuid

import torch
import torch.distributed as dist
import torchsnapshot
from torch.nn.parallel import DistributedDataParallel


class Model(torch.nn.Module):
    def __init__(self, param_size: int, num_params: int) -> None:
        super().__init__()
        for i in range(num_params):
            self.register_parameter(
                f"param_{i}",
                torch.nn.Parameter(
                    torch.rand(int(param_size / 4), device=torch.cuda.current_device())
                ),
            )


def rank_0_print(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg)

import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="tmp")
    parser.add_argument("--param-size", type=int, default=int(100_000_000))
    parser.add_argument("--num-params", type=int, default=200)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")

    model = Model(param_size=args.param_size, num_params=args.num_params)
    model = DistributedDataParallel(model, gradient_as_bucket_view=True)

    sz = sum(t.nelement() * t.element_size() for t in model.parameters())
    rank_0_print(f"Model size: {sz / 1_000_000_000.0} GB")

    # rank_0_print("Saving the model with torchsnapshot...")
    # t_begin = time.monotonic()
    # app_state = {"model": model}
    # snapshot = torchsnapshot.Snapshot.take(
    #     path=f"{args.work_dir}/{uuid.uuid4()}",
    #     app_state=app_state,
    #     replicated=["**"],
    # )
    # os.sync()
    # rank_0_print(f"Snapshot path: {snapshot.path}")
    # rank_0_print(f"Took {time.monotonic() - t_begin} seconds with torchsnapshot")
    # dist.barrier()

    _patch_model_state_dict(model)
    rank_0_print("Saving the model with DCP...")
    for num_threads in range(32, 33):
        checkpointer = DCP.FileSystemCheckpointer(
            f"{args.work_dir}/{uuid.uuid4()}",
            thread_count=num_threads
        )

        begin_ts = time.monotonic()
        checkpointer.save(state_dict={"model": model})
        end_ts = time.monotonic()
        rank_0_print(f"{num_threads}, {time.monotonic() - begin_ts}")

        dist.barrier()
        if dist.get_rank() == 0:
            import shutil
            # Delete a directory and all its contents
            shutil.rmtree(args.work_dir)
            shutil.os.makedirs(args.work_dir)

        dist.barrier()
