import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.model_executor.model_runner import ModelRunner  # Standard SGLang import order.
from sglang.srt.models import qwen3_vl as model


class VisionDPTest(unittest.TestCase):
    def test_dp_attention_replicates_vision_weights(self):
        cfg = SimpleNamespace(vision_config=SimpleNamespace(), tie_word_embeddings=False,
                              vocab_size=16, hidden_size=4, rope_scaling={'mrope_section':[1,1,1]})
        args = SimpleNamespace(mm_enable_dp_encoder=False)
        language = MagicMock()
        with patch.object(model, 'is_dp_attention_enabled', return_value=True, create=True), \
             patch.object(model, 'get_global_server_args', return_value=args), \
             patch.object(model, 'Qwen3VLMoeVisionModel') as vision, \
             patch.object(model, 'Qwen3LLMModel', language), \
             patch.object(model, 'ParallelLMHead'), \
             patch.object(model, 'LogitsProcessor'), \
             patch.object(model, 'Pooler'):
            model.Qwen3VLForConditionalGeneration(cfg, language_model_cls=language)
            self.assertTrue(vision.call_args.kwargs['use_data_parallel'])

    def check_dispatch(self, modality, dp_attention):
        visual = MagicMock()
        visual.dtype = torch.float32
        local_result = torch.full((2,4), 3.)
        gathered_result = torch.full((2,4), 7.)
        visual.return_value = local_result
        instance = SimpleNamespace(visual=visual, use_data_parallel=True)
        item = SimpleNamespace(feature=torch.ones(2,8), image_grid_thw=torch.tensor([[1,2,2]]),
                               video_grid_thw=torch.tensor([[2,2,2]]))
        with patch.object(model, 'is_dp_attention_enabled', return_value=dp_attention, create=True), \
             patch.object(model, 'run_dp_sharded_mrope_vision_model', return_value=gathered_result) as sharded:
            method = getattr(model.Qwen3VLForConditionalGeneration, f'get_{modality}_feature')
            result = method(instance, [item])
            if dp_attention:
                visual.assert_called_once()
                sharded.assert_not_called()
                self.assertIs(result, local_result)
            else:
                visual.assert_not_called()
                sharded.assert_called_once()
                self.assertIs(result, gathered_result)

    def test_image_stays_local_with_dp_attention(self):
        self.check_dispatch('image', True)

    def test_video_stays_local_with_dp_attention(self):
        self.check_dispatch('video', True)

    def test_existing_image_sharding_preserved(self):
        self.check_dispatch('image', False)

    def test_existing_video_sharding_preserved(self):
        self.check_dispatch('video', False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
