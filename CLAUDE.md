# mmDar Project Instructions

## Second Opinion Workflow

When to invoke a second opinion:
- Design plans and implementation plans -- get another model's opinion before finalizing
- When the user explicitly asks to "ask Codex/Gemini" or wants a second opinion
- Architecture decisions, experiment design, physics/math reasoning checks

Workflow:
1. Formulate a clear prompt with full context (paste the plan, code snippet, or question)
2. Run the second-opinion tool in read-only mode (it can read the repo but not modify it)
3. Compare Claude's opinion with the second opinion
4. If they agree -- proceed with the merged best approach
5. If they disagree -- present both opinions to the user with tradeoffs and let them decide

Do NOT use second-opinion tools for: actual code generation or file edits -- purely for interpretation, feedback, and second opinions.

### Codex (OpenAI gpt-5.4)

Codex CLI is installed via npm (`@openai/codex`) and configured at `~/.codex/config.toml` with `model = "gpt-5.4"` and `model_reasoning_effort = "xhigh"`.

**Non-interactive invocation (read-only, no file changes):**
```bash
codex exec -s read-only -C <PROJECT_ROOT> --ephemeral "PROMPT" 2>/dev/null
```

**Key flags:**
- `-s read-only` -- Codex reads but never writes (opinion-only use)
- `--ephemeral` -- don't persist throwaway sessions
- `-C <PROJECT_ROOT>` -- set working directory (e.g. `/git/mmDar` or `$PWD`)
- `-o <file>` -- capture clean output to file (also prints to stdout)
- Stdin pipe for long prompts: `echo "PROMPT" | codex exec -s read-only --ephemeral - 2>/dev/null`

### Gemini (Google Gemini CLI)

Gemini CLI is installed via npm (`@google/gemini-cli`). Uses OAuth authentication (run `gemini` interactively once to authenticate).

**Non-interactive invocation (read-only, no file changes):**
```bash
cd <PROJECT_ROOT> && gemini -p "PROMPT" --approval-mode plan 2>/dev/null
```

**Key flags:**
- `-p "PROMPT"` -- non-interactive (headless) mode
- `--approval-mode plan` -- read-only mode (can read files, cannot write)
- `-m <model>` -- override model (optional, uses default if omitted)
- `-s` -- run in sandbox (extra isolation, optional)
- Stdin pipe for long prompts: `echo "PROMPT" | gemini -p "" --approval-mode plan 2>/dev/null`

**Differences from Codex:**
- Gemini requires `cd` into project dir (no `-C` flag for working directory)
- No `--ephemeral` equivalent -- Gemini saves session history by default
- Auth is OAuth-based (may need periodic re-verification via browser)

### Choosing Between Them

| Scenario | Prefer |
|----------|--------|
| Code review, implementation feedback | Codex (stronger at code-level detail) |
| Architecture, system design | Either -- compare both when stakes are high |
| Physics/math/science reasoning | Both -- cross-validate for correctness |
| Quick sanity check | Codex (faster startup, ephemeral sessions) |
| When one tool is down/rate-limited | Use the other |

For high-stakes decisions (experiment design, major refactors), run both and present the user with a comparison.

## Documentation Rules

- **README.md must have a "Changes From Original RadarHD" section** that documents every modification from the upstream repo. When making changes to training, inference, evaluation, or infrastructure code, update this section immediately. Include: what changed, the original behavior, the new behavior, and impact/rationale.
- **Lessons learned from experiments belong in the README** under "Lessons Learned", not just in planning docs. These are permanent project knowledge.
- **results/README.md must stay in sync** with any new evaluation runs. Update the comparison tables when new results are produced.

## Running Experiments

### Trigger: "sweep"
When the user says **"sweep"** (e.g., "sweep learning rates", "sweep batch sizes"), enter autonomous experiment mode:
1. Design the first experiment based on context
2. Launch it in background via Docker
3. When notified of completion, analyze results, decide next experiment adaptively, launch it
4. Keep going until results converge or the user says stop
5. Maintain a running comparison table, report concisely after each run

### Execution model
- Training runs inside Docker: `docker compose run --rm mmdar python3 <script> 2>&1`
- Launch as background tasks so the user isn't blocked. They can edit code while training runs.
- After each experiment completes, **analyze results immediately**, decide the next experiment adaptively, and launch it. Don't wait for the user unless a decision point is ambiguous.
- Report results concisely with a running comparison table after each experiment.

### Time budget awareness
Know where time goes and estimate accurately before committing to a plan:
- **Data loading:** ~15 min per fresh Docker container (loads all trajectories into RAM)
- **Training:** ~1.6 min/epoch at batch=12 bf16, ~1.15 min/epoch at batch=24 bf16, ~2.2 min/epoch at batch=6 fp32
- **Inference:** ~9 min per checkpoint at batch=1 on full test set (18,575 samples)
- **Eval (metrics computation):** ~2-3 min per checkpoint
- A "quick" experiment with 50 epochs + 3-checkpoint sweep = ~2h total including container overhead

### Adaptive sweep protocol
1. Don't pre-commit to a fixed grid. Run one experiment, analyze, then decide the next.
2. After each run, re-evaluate: is the trend clear? Should we go higher/lower/stop?
3. Start with the most informative experiment (e.g., middle of the range), then binary search.
4. If the user gives a time budget, estimate how many experiments fit and prioritize accordingly.

### Checkpoint selection
- **Never select checkpoints by training loss.** BCE+Dice loss in polar space diverges from Chamfer/mod-Hausdorff after early epochs.
- Save checkpoints every 10 epochs during training.
- After training, sweep a small number of checkpoints (e.g., epochs 10, 20, 30) by running inference + eval with Chamfer distance.
- The sweet spot for this architecture is typically epoch 10-30 depending on batch size and LR.
- Use `run_experiment.py --sweep-epochs 10,20,30` to limit the expensive sweep phase.

### Efficiency rules
- **Minimize checkpoint sweeps.** Only evaluate 3-5 epochs per run, not all. Expand only if the best is at the boundary.
- **Train only as many epochs as needed.** The sweet spot is epoch 10-30 for most configs. Don't train 100+ epochs unless investigating late convergence.
- **Reuse containers when possible.** Each Docker launch costs ~15 min for data loading.
- **Inference batch size != training batch size.** Inference at batch=16-32 is safe (BatchNorm uses frozen running stats) and would cut eval time from 9 min to ~1-2 min.

### Current best config (RTX 5090)

**Gaussian model (PhysicsGaussianModel):**
- batch=4, lr=1e-4, 50 epochs, window=41, K=96, N_az=64, mixed split, augment
- sigma_r_prior=0.3, huber_range_weight=0.1
- Test mod-H 0.205 (thresh=0.3), Chamfer 0.280
- Near-field (0-2m): 0.162m, worst-case traj_max: 0.617
- Low-ID test: mod-H 0.112, High-ID test: mod-H 0.250
- Train: `python3 train/train.py --train --split mixed --augment`

**Baseline U-Net (for reference):**
- batch=12, lr=7e-5, fp32, epoch 10 → Chamfer 0.295m, mod-H 0.189m (test-selected, unfair)
- Honest (val-selected): mod-H 0.296

**Loss tuning lessons (2026-04-02 sweep):**
- Relaxing sigma_r_prior from 0.1 to 0.3 improved calibration
- Huber range loss (w=0.1) reduced range bias, especially mid/far-field
- FPS prototypes (replacing K-Means) hurt — cluster centers are better regression targets
- Huber > 0.2 overwhelms NLL and degrades angular quality

## Git Commit Rules

- Never mention AI, Claude, Copilot, Codex, Gemini, or any AI tool in commit messages, PR descriptions, branch names, or any git metadata
- No Co-Authored-By, Signed-off-by, or similar trailers referencing AI
- Commits must appear as sole-author human work
