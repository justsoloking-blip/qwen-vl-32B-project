"""Exercise the real ForwardBatch padding seam, without loading model weights."""
import unittest
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MropePaddingTest(unittest.TestCase):
    def check_padding(self, original_tokens, padded_tokens, original_bs, padded_bs):
        batch = object.__new__(ForwardBatch)
        batch.input_ids = torch.arange(original_tokens)
        batch.req_pool_indices = torch.arange(original_bs)
        batch.lora_ids = [None] * original_bs
        batch.seq_lens = torch.full((original_bs,), max(original_tokens, 1))
        batch.seq_lens_cpu = batch.seq_lens.clone()
        batch.seq_lens_sum = int(batch.seq_lens.sum())
        batch.out_cache_loc = torch.arange(original_tokens)
        batch.encoder_lens = None
        batch.positions = torch.arange(original_tokens)
        original = torch.arange(3 * original_tokens).reshape(3, original_tokens)
        batch.mrope_positions = original.clone()
        batch.extend_seq_lens = None
        batch.spec_info = None
        runner = SimpleNamespace(attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 1))
        batch._pad_inputs_to_size(runner, padded_tokens, padded_bs)
        self.assertEqual(tuple(batch.mrope_positions.shape), (3, padded_tokens))
        torch.testing.assert_close(batch.mrope_positions[:, :original_tokens], original)
        self.assertTrue(torch.all(batch.mrope_positions[:, original_tokens:] == 0))
        self.assertEqual(batch.mrope_positions.dtype, original.dtype)

    def test_prefill_original_failure(self):
        self.check_padding(78, 78, 1, 1)

    def test_prefill_token_padding(self):
        self.check_padding(5, 8, 2, 2)

    def test_decode_padding(self):
        self.check_padding(1, 4, 1, 4)

    def test_idle_rank_padding(self):
        self.check_padding(0, 4, 0, 4)

    def test_three_requests_preserve_axes(self):
        self.check_padding(7, 10, 3, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
