
# Scarlet Chess Engine

Scarlet is an open source chess engine that implements most of the UCI protocol.  The engine includes an efficient bitboard representation of the gamestate based on the pext and pdep instructions, meaning that zen2 is unsupported.

## Strength

Scarlet v1 is rated 2640 on the CCRL blitz rating list as of August 9th, 2026.\
Scarlet v2 is rated 3381 CCRL blitz (#148) as of October 17th, 2026.

## Engine Features

- Principle Variation Search
- Quiescence Search
- Transposition Table (move ordering and cutoffs)
- Internal iterative reductions if no TT move
- Time Management with Hard/Soft bounds
- Null move pruning
- Reverse futility pruning
- Futility pruning
- Late move pruning
- Late move reductions
- Static Exchange Evaluation (SEE) pruning and move ordering
- Singular Extensions (SE) with multicut
- Improving heuristic
- History heuristic using [type][from][to] w/ gravity, penalties, and reductions for LMR
- Capture history
- Continuation history (1+2 ply)
- Killer moves heuristic
- Hard/Soft time management
- Check extension
- Aspiration windows
- Early cycle/repetition detection
- Efficiently updatable neural network (NNUE) evaluation using 768->512x2->1 architecture, trained on 2GB of selfplay data in viriformat.

## Usage

Find and download the binary from releases most fitting your system, then run that executable. <br>
Alternatively, you can build the project yourself. Ensure both rust and cargo are installed, enter the main directory of the project, and run
`cargo build --bin scarlet --release`.  After this, the executable for the chess engine will be in target/release. 

## Android ARM64 build

The repository includes a GitHub Actions workflow (`Build Scarlet Android ARM64`) and a local build script:

```bash
bash scripts/build-android-arm64.sh
```

Set `ANDROID_NDK_HOME` to a Linux Android NDK installation. The script targets `aarch64-linux-android` (API 24), keeps the baseline binary on generic ARM64 + NEON, compiles the AArch64 test configuration, and checks the final ELF/disassembly. It emits only `dist/scarlet-android-arm64`.

The sliding-attack PEXT/PDEP operations retain BMI2 assembly on x86-64 and use an exact portable bit-extraction/deposition fallback on AArch64. NNUE accumulator updates and the non-Dotprod output evaluator use NEON. The bundled network's output-layer weights are signed-byte representable; an exact SDOT output-layer kernel is compiled separately and used only when runtime CPU detection reports Dotprod and the network weight-range check passes. Other networks/CPUs fall back to NEON. The feature-weight layer remains 16-bit and does not use SDOT.

The training/data-generation binaries and CUDA-backed `bullet` dependency are opt-in through `--features training`; the Android engine build does not enable that feature.
