# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.distributed.collectives import DeterministicCollective


def test_direct_staging_view_does_not_specialize_dynamic_token_count():
    collective = object.__new__(DeterministicCollective)
    collective._direct_staging_views = {((4, 8), torch.float32): torch.zeros(4, 8)}
    assert collective.direct_staging_view((4, 8), dtype=torch.float32) is not None

    def project(x):
        staging = collective.direct_staging_view((x.shape[0], 8), dtype=x.dtype)
        assert staging is None
        return x + 1

    compiled = torch.compile(project, backend="eager", fullgraph=True)
    for rows in (4, 8):
        x = torch.zeros(rows, 8)
        torch._dynamo.mark_dynamic(x, 0)
        torch.testing.assert_close(compiled(x), torch.ones_like(x))
