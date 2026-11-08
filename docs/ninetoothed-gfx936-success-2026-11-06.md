# NineToothed gfx936 installation/JIT result

Run20261106T124640Z-1947749 completed with JIT_VERIFIED, complete, state_published=true. All9 expected add cases passed (FP32/FP16/BF16,1/513/strided513), inputs and output guards unchanged. Actual Triton target HIP/gfx936/warp64 on GPU0. NineToothed0.26.0, SymPy1.14.0,mpmath1.3.0,privateNumPy1.26.4 andvendorTriton3.5.1+das.opt1.dtk2604.torch290 checked. Protected Torch2.9.0/HIP6.3.26093 unchanged; baseNumPy1.25.0 remains separate. Elapsed402.252s includes installation/staging/compile/tests and is not an operator performance measurement.

Candidate AFS path: python-envs/ninetoothed-gfx936/20261106T124640Z-1947749. Verified state published at state/ninetoothed-gfx936.json per report/log. State file itself was not supplied locally. Triton and compiler caches were job-local; later jobs must restage vendor Triton and its compatibility patch using the recorded recipe. Do not assume global activation or persistence of /dev/shm paths.

Both cache compatibility changes passed this actual run: NineToothed generation.py cache-location patch SHA17f908e7ee7ddf9e87f2a2cb51d302d6d2b32211da7735998fbced276b8cbf4e; vendor cache.py same-directory publication patch SHAf99e24bc0b1a7eea0aeeb7cbe6ba5cee41b778d9a2f96368c6b5b96cd5b5ca0a. Three generated source hashes recorded. Generated source/cache contents and NPZ were not separately returned, so no independent local recomputation of device arrays claimed.

Installation and standalone NineToothed JIT milestone passed for these cases on GPU0. InfiniCore integration remains NOT_TESTED, quality UNASSESSED, no8rank or fullH3/optimization acceptance. Existing Attention numerical investigation remains pending actual all-rank capture result. No numerical implementation change in this result-analysis turn.

Original twoJSON/full log/checksums archived at artifacts/fl2va-quality-evidence-2026-11-04/ninetoothed-success-20261106T124640Z-1947749/. Local validator checked exact9-case protocol, actualarch, protectedidentity and statepublication fields.
