# gfx1200 BF16 WMMA linear-layout reproduction

This branch hosts the reproduction package accompanying the TileLang feature
PR. It is independent of the source branch and is not intended to be merged.

- Source commit: https://github.com/gwokhou/tilelang/commit/cb87202f1ca0c458a5f0c1c194babf994ab827e6
- Tested base: `0176ba1465b12758f41802bf838ddd68d43c9ee9`.
- Package: [gfx1200-linear-layout-reproduction.zip](gfx1200-linear-layout-reproduction.zip).
- Package SHA-256: `a04eade76894fd0e53a16bba07df06e1b0950c06f3c52ad2bc648efed5a764e2`.

The archive contains the measured patch, follow-up patch, validation and
measurement summaries, reproduction scripts, and setup instructions. Raw
traces and process logs are omitted. The launcher cleanup passes 14 CPU-only
fault-injection tests; the packaged launcher has not been rerun on a GPU or
verified on a fresh machine. See the packaged README for evidence boundaries.
