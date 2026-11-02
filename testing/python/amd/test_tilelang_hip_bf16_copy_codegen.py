"""BF16 staging copies must be native values, with a conservative fallback."""

import re
from pathlib import Path
import subprocess

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _copy_module(
    dtype="bfloat16",
    lanes=8,
    src_scope="global",
    dst_scope="local",
    src_stride=1,
    dst_stride=1,
    src_offset=8,
    dst_offset=8,
    converted=False,
    volatile=None,
    predicate=None,
    predicate_side=None,
    byte_pointer=False,
    src_name="src",
    dst_name="dst",
    symbol="copy_bf16",
    after_copy_var=None,
    after_copy_call=None,
    consumer="shared",
):
    src = tirx.decl_buffer((64,), dtype, name=src_name, scope=src_scope)
    dst = tirx.decl_buffer((64,), dtype, name=dst_name, scope=dst_scope)
    if byte_pointer:
        src = tirx.decl_buffer((64,), dtype, name="src", data=tirx.Var("src", tvm.ir.PointerType(tvm.ir.PrimType("uint8"), "global")))
    src_index = tirx.Ramp(src_offset, src_stride, lanes)
    dst_index = tirx.Ramp(dst_offset, dst_stride, lanes)
    load = tirx.BufferLoad(src, [src_index], predicate=predicate if predicate_side == "load" else None)
    value = tirx.Cast(f"{dtype}x{lanes}", tirx.Cast(f"float32x{lanes}", load)) if converted else load
    body = tirx.BufferStore(dst, value, [dst_index], predicate=predicate if predicate_side == "store" else None)
    if consumer is not None:
        sink = tirx.decl_buffer((64,), dtype, name="sink", scope="shared")
        # Keep consumer accesses valid independently of producer alignment and
        # stride, so negative producer cases exercise the original boundary.
        consumer_index = tirx.Ramp(8, 1, lanes)
        read = tirx.BufferLoad(dst, [consumer_index])
        if consumer == "converted":
            read = tirx.Cast(f"{dtype}x{lanes}", tirx.Cast(f"float32x{lanes}", read))
        if consumer == "unrelated":
            other = tirx.decl_buffer((64,), dtype, name="other", scope="local")
            body = tirx.SeqStmt(
                [tirx.AllocBuffer(other), body, tirx.BufferStore(other, tirx.Broadcast(tirx.const(0, dtype), lanes), [consumer_index])]
            )
            read = tirx.BufferLoad(other, [consumer_index])
        body = tirx.SeqStmt([tirx.AllocBuffer(sink), body, tirx.BufferStore(sink, read, [consumer_index])])
        if consumer == "mixed":
            body = tirx.SeqStmt([body, tirx.BufferStore(src, tirx.BufferLoad(dst, [dst_index]), [src_index])])
        if consumer == "escaped":
            body = tirx.SeqStmt([body, tirx.Evaluate(tirx.call_extern("int32", "consume", dst.data))])
        if consumer in ("init_load", "init_escape", "init_unrelated"):
            scalar = tirx.decl_buffer((1,), "int32", name="init", scope="local.var")
            init = (
                tirx.call_extern("int32", "consume", dst.data)
                if consumer == "init_escape"
                else tirx.Cast("int32", tirx.BufferLoad(src if consumer == "init_unrelated" else dst, [8]))
            )
            body = tirx.SeqStmt(
                [
                    body,
                    tirx.AllocBuffer(scalar, annotations={"tl.local_var_init": init}),
                    tirx.Evaluate(tirx.call_extern("int32", "consume_init", tirx.BufferLoad(scalar, [0]))),
                ]
            )
    if after_copy_var:
        body = tirx.SeqStmt([body, tirx.Bind(tirx.Var(after_copy_var, "int32"), tirx.const(1, "int32"))])
    if after_copy_call:
        body = tirx.SeqStmt([body, tirx.Evaluate(tirx.call_extern("int32", after_copy_call, 1))])
    params = [src.data, dst.data]
    if volatile == "dst":
        body = tirx.SeqStmt([tirx.AllocBuffer(dst, annotations={"tirx.volatile": True}), body])
        params = [src.data]
    if not isinstance(src_offset, int):
        params += list(tirx.analysis.undefined_vars(tirx.Evaluate(src_offset)))
    func = tirx.PrimFunc(params, body)
    func = func.with_attr("global_symbol", symbol)
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm.IRModule({symbol: func})


def _copy_source(target="gfx1200", **kwargs):
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(
        _copy_module(**kwargs), tvm.target.Target({"kind": "rocm", **({"mcpu": target} if target is not None else {})})
    ).inspect_source()


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("byte_pointer", [False, True])
def test_bf16_staging_uses_native_vector_value(byte_pointer):
    source = _copy_source(byte_pointer=byte_pointer)
    assignment = next(line for line in source.splitlines() if " = " in line and "src" in line)
    assert len(re.findall(r"unsigned int __attribute__\(\(vector_size\(16\), may_alias\)\)", assignment)) == 2, assignment
    assert "src" in assignment and "dst" in assignment
    assert "[8]" in assignment, assignment
    assert "uint4" not in assignment


@tilelang.testing.requires_rocm
def test_bf16_staging_symbolically_aligned_offset():
    offset = tirx.Var("offset", "int32")
    assert "vector_size(16)" in _copy_source(src_offset=offset * 8)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("consumer", [None, "unrelated", "converted", "mixed", "escaped"])
def test_bf16_native_copy_requires_exclusive_shared_staging(consumer):
    assert "vector_size(16)" not in _copy_source(consumer=consumer)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "consumer, initializer, native",
    [
        ("init_load", "int init = ((int)dst[8]);", False),
        ("init_escape", "int init = consume(dst);", False),
        ("init_unrelated", "int init = ((int)src[8]);", True),
    ],
)
def test_bf16_staging_checks_local_var_initializer(consumer, initializer, native):
    source = _copy_source(consumer=consumer)
    assert initializer in source
    assert "consume_init(init);" in source
    assert ("vector_size(16)" in source) == native


@tilelang.testing.requires_rocm
def test_bf16_staging_consumer_state_is_per_function():
    first = _copy_module()["copy_bf16"]
    # Reuse the same Var objects so this catches a stale buffer-identity set.
    body = tirx.BufferStore(
        tirx.decl_buffer((64,), "bfloat16", data=first.params[1]),
        tirx.BufferLoad(tirx.decl_buffer((64,), "bfloat16", data=first.params[0]), [tirx.Ramp(8, 1, 8)]),
        [tirx.Ramp(8, 1, 8)],
    )
    second = tirx.PrimFunc(first.params, body).with_attr("global_symbol", "copy_only")
    second = second.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile")
    source = build(
        tvm.IRModule({"copy_bf16": first, "copy_only": second}), tvm.target.Target({"kind": "rocm", "mcpu": "gfx1200"})
    ).inspect_source()
    assert source.count("vector_size(16)") == 2


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "kwargs",
    [
        {"src_name": "tl_copy_u32x4"},
        {"dst_name": "tl_copy_u32x4"},
        {"symbol": "tl_copy_u32x4"},
        {"src_name": "tl:copy_u32x4"},
        {"src_name": "tl-copy_u32x4"},
        {"after_copy_var": "tl:copy_u32x4"},
        {"after_copy_call": "tl_copy_u32x4"},
    ],
)
def test_bf16_staging_helper_name_does_not_collide(kwargs, tmp_path):
    from tilelang.contrib import hipcc

    try:
        compiler = hipcc.find_hipcc()
    except RuntimeError:
        pytest.skip("HIP frontend is required for the generated-name regression")
    source = _copy_source(**kwargs)
    if "after_copy_call" in kwargs:
        source = source.replace('extern "C" __global__', 'extern "C" __device__ int tl_copy_u32x4(int);\nextern "C" __global__')
    assert "vector_size(16)" in source
    path = tmp_path / "copy.hip"
    path.write_text(source)
    result = subprocess.run(
        [
            compiler,
            "-x",
            "hip",
            "-std=c++17",
            "--offload-arch=gfx1200",
            "--cuda-device-only",
            "-fsyntax-only",
            "-I" + str(Path(tilelang.__file__).resolve().parents[1] / "src"),
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, source + "\n" + result.stdout + result.stderr


@tilelang.testing.requires_rocm
def test_bf16_staging_default_target_uses_detected_arch():
    target = tvm.target.Target("rocm")
    source = _copy_source(target=None)
    assert ("vector_size(16)" in source) == (target.attrs.get("mcpu") == "gfx1200")


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "gfx950"},
        {"target": "gfx1201"},
        {"target": "gfx12000"},
        {"src_scope": "shared"},
        {"dst_scope": "shared"},
        {"dtype": "float16"},
        {"lanes": 4},
        {"src_stride": 2, "lanes": 4},
        {"dst_stride": 2, "lanes": 4},
        {"dst_offset": 1},
        {"converted": True},
        {"volatile": "dst"},
    ],
)
def test_bf16_staging_keeps_unmatched_copy_path(kwargs):
    source = _copy_source(**kwargs)
    assert "vector_size(16)" not in source
    assert "src" in source and "dst" in source


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("kwargs", [{"src_offset": 1}, {"src_stride": 2}, {"dst_stride": 2}])
def test_bf16_staging_does_not_bypass_unsupported_ramp(kwargs):
    with pytest.raises(tvm.error.InternalError, match="Ramp of more than 4 lanes"):
        _copy_source(**kwargs)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("side", ["load", "store"])
def test_bf16_staging_does_not_bypass_predicate_rejection(side):
    with pytest.raises(tvm.error.InternalError, match="Predicated buffer (load|store) is not supported"):
        _copy_source(predicate_side=side, predicate=tirx.Broadcast(tirx.const(True, "bool"), 8))


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("elements,threads,shared_staging", [(256, 32, True), (256, 32, False), (4096, 128, False)])
def test_bf16_staging_preserves_bits_and_copy_boundaries(elements, threads, shared_staging):
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1200":
        pytest.skip("gfx1200 native BF16 copy path")

    @T.prim_func
    def kernel(A: T.Tensor((elements + 16,), "bfloat16"), B: T.Tensor((elements + 16,), "bfloat16")):
        with T.Kernel(1, threads=threads):
            fragment = T.alloc_fragment((elements,), "bfloat16")
            T.copy(A[8 : elements + 8], fragment)
            if shared_staging:
                shared = T.alloc_shared((elements,), "bfloat16")
                T.copy(fragment, shared)
                T.sync_threads()
                T.copy(shared, B[8 : elements + 8])
            else:
                T.copy(fragment, B[8 : elements + 8])

    compiled = tilelang.compile(kernel, target="hip")
    assert ("vector_size(16)" in compiled.get_kernel_source()) == shared_staging
    # Include signed zeros, infinities, NaNs and subnormals; this is a bit copy.
    bits = torch.arange(elements + 16, dtype=torch.int32).mul(241).to(torch.int16)
    bits[8:16] = torch.tensor([0, -32768, 32640, -128, 32704, -64, 1, -32767], dtype=torch.int16)
    source = bits.view(torch.bfloat16).to("cuda")
    output_bits = torch.full((elements + 16,), 0x1234, dtype=torch.int16, device="cuda")
    compiled(source, output_bits.view(torch.bfloat16))
    torch.cuda.synchronize()
    expected = torch.full((elements + 16,), 0x1234, dtype=torch.int16)
    expected[8 : elements + 8] = bits[8 : elements + 8]
    torch.testing.assert_close(output_bits.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source.view(torch.int16).cpu(), bits, rtol=0, atol=0)


def _make_bf16_prefetch_gemm(k_size):
    @T.prim_func
    def kernel(A: T.Tensor((16, k_size), "bfloat16"), B: T.Tensor((k_size, 32), "bfloat16"), C: T.Tensor((16, 32), "bfloat16")):
        with T.Kernel(1, threads=32):
            a_shared = T.alloc_shared((16, 32), "bfloat16")
            b_shared = T.alloc_shared((32, 32), "bfloat16")
            a_next = T.alloc_fragment((16, 32), "bfloat16")
            b_next = T.alloc_fragment((32, 32), "bfloat16")
            accum = T.alloc_fragment((16, 32), "float32")
            T.clear(accum)
            T.copy(A[:, 0:32], a_next)
            T.copy(B[0:32, :], b_next)
            T.copy(a_next, a_shared)
            T.copy(b_next, b_shared)
            T.sync_threads()
            for k in T.serial(k_size // 32 - 1):
                T.copy(A[:, (k + 1) * 32 : (k + 2) * 32], a_next)
                T.copy(B[(k + 1) * 32 : (k + 2) * 32, :], b_next)
                T.gemm(a_shared, b_shared, accum)
                T.sync_threads()
                T.copy(a_next, a_shared)
                T.copy(b_next, b_shared)
                T.sync_threads()
            T.gemm(a_shared, b_shared, accum)
            T.copy(accum, C)

    return kernel


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("k_size", [32, 512])
def test_bf16_prefetch_gemm_correctness(k_size):
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1200":
        pytest.skip("gfx1200 native BF16 copy path")
    compiled = tilelang.compile(_make_bf16_prefetch_gemm(k_size), target="hip")
    assert "vector_size(16)" in compiled.get_kernel_source()
    generator = torch.Generator().manual_seed(17)
    a_int = torch.randint(-3, 4, (16, k_size), generator=generator, dtype=torch.int64)
    b_int = torch.randint(-3, 4, (k_size, 32), generator=generator, dtype=torch.int64)
    for tile_signature in (False, True):
        if tile_signature:
            a_int = (torch.arange(k_size) // 32 + 1).expand(16, -1).to(torch.int64)
            b_int = torch.ones((k_size, 32), dtype=torch.int64)
        expected = ((a_int @ b_int).double() / 4).to(torch.bfloat16)
        a = (a_int.float() / 2).to(device="cuda", dtype=torch.bfloat16)
        b = (b_int.float() / 2).to(device="cuda", dtype=torch.bfloat16)
        storage = torch.full((528,), 7, device="cuda", dtype=torch.bfloat16)
        output = storage[8:520].view(16, 32)
        output.fill_(float("nan"))
        compiled(a, b, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        assert torch.all(storage[:8] == 7) and torch.all(storage[520:] == 7)
        torch.testing.assert_close(a.cpu(), (a_int.float() / 2).to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), (b_int.float() / 2).to(torch.bfloat16), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
