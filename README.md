# silent-deploy

Production deploy artifacts for **silent** — a JEPA predator that hunts
the human player by audio. 4-channel cardioid mel-spectrograms, ViT-Tiny
encoder + AR causal predictor + DexWM joint state head (λ=10), CEM
planner in latent space, post-hoc head decodes positions for cost.

Runs on the JEPA inference VPS at `jepa.waweapps.win/silent/` behind a
Caddy reverse proxy ([jepa-vps-proxy](https://github.com/SotoAlt/jepa-vps-proxy)).

## What's here

A self-contained code subset of the private [aura-new-games](https://github.com/SotoAlt/aura)
worktree — only what's needed to run the WebSocket inference server.
No training data, no notebooks, no GPU pipeline.

- `Dockerfile` — CPU PyTorch + librosa + pymunk
- `docker-compose.yml` — joins the shared external `web` Docker network
- `world_model/`, `envs/`, `scripts/`, `client/silent/` — minimum source
- `checkpoints/` — gitignored; populate via scp

## Five model variants wired in (so you can A/B in the public demo)

| Slot         | Checkpoint                                | Head                                |
| ------------ | ----------------------------------------- | ----------------------------------- |
| canonical    | `silent_v1_3e_ep030.pt` (3E ep30)         | `3e_ep030_head_uniform.pt`          |
| jepa_baseline| `silent_BASELINE_ep010_joint.pt`          | `silent_BASELINE_head_uniform.pt`   |
| jepa_test1   | `silent_v1_3f_ep010.pt` (3F effective 40) | `3f_ep010_head_uniform.pt`          |
| jepa_test2   | `silent_v1_3f_ep015.pt` (3F effective 45) | `3f_ep015_head_uniform.pt`          |
| jepa_test3   | `silent_v1_3f_ep020.pt` (3F effective 50) | `3f_ep020_head_uniform.pt`          |

## Deploy on a fresh VPS

Prereqs: Caddy proxy on the `web` network with `/silent/*` → `silent:8801`
routing (see jepa-vps-proxy repo).

```bash
ssh jepa-vps
cd /srv/silent
git clone https://github.com/SotoAlt/silent-deploy.git .
docker compose up -d --build
```

Then from your Mac, scp the checkpoints over (one-time):

```bash
LOCAL=/Users/rodrigosoto/repos/aura-new-games/checkpoints
scp $LOCAL/silent_full/silent_v1_3e_ep030.pt              jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/3e_ep030_head_uniform.pt           jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/silent_v1_3f_ep010.pt              jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/3f_ep010_head_uniform.pt           jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/silent_v1_3f_ep015.pt              jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/3f_ep015_head_uniform.pt           jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/silent_v1_3f_ep020.pt              jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/silent_full/3f_ep020_head_uniform.pt           jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/BASELINE_DO_NOT_DELETE/silent_BASELINE_ep010_joint.pt    jepa-vps:/srv/silent/checkpoints/
scp $LOCAL/BASELINE_DO_NOT_DELETE/silent_BASELINE_head_uniform.pt   jepa-vps:/srv/silent/checkpoints/
docker compose restart silent
```

## Architecture

```
browser  →  https://jepa.waweapps.win/silent/   (Caddy)
                       ↓ reverse_proxy /silent/* silent:8801
                  silent container (port 8801)
                       ↓ FastAPI + websockets
                  world_model.infer_silent_env  (env loop, JEPA hunts)
```

Article: [sotoalt.dev/experiments/silent.html](https://sotoalt.dev/experiments/silent.html)

Source for training the model + the full research history lives in the
private [aura](https://github.com/SotoAlt/aura) repo.
