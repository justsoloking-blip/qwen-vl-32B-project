"""Instrument only the disposable TP1 overlay; never modify the serving overlay."""
from pathlib import Path
import shutil
B=Path('/deployment/qwen3-vl-32b');d=B/'performance/tp1';ov=d/'overlay'
assert ov.is_symlink();ov.unlink();shutil.copytree(B/'overlay',ov)
p=ov/'sglang/srt/multimodal/processors/qwen_vl.py';s=p.read_text();helper='''
# Temporary profiling instrumentation in the disposable performance instance.
def _profile_mm_cpu(fn):
    async def wrapped(*args, **kwargs):
        from pathlib import Path
        import torch
        root = Path("/deployment/qwen3-vl-32b/performance")
        if not (root / "mm-profile.flag").exists():
            return await fn(*args, **kwargs)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
            with torch.profiler.record_function("mm_frontend_pipeline"):
                result = await fn(*args, **kwargs)
        out = root / "traces/mm_frontend"
        out.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(out / f"mm-cpu-{time.time()}.trace.json.gz"))
        return result
    return wrapped

'''
s=s.replace('# Compatible with Qwen-VL & Qwen-Omni Series',helper+'# Compatible with Qwen-VL & Qwen-Omni Series');s=s.replace('    async def process_mm_data_async(', '    @_profile_mm_cpu\n    async def process_mm_data_async(');s=s.replace('logger.debug(\n            f"[QwenVLProcessor Perf]', 'logger.info(\n            f"[QwenVLProcessor Perf]');p.write_text(s)
