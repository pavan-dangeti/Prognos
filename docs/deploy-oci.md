# Deploying to Oracle Always Free A1

The demo host. Everything in this document is either free or already running; nothing
here creates a billable resource.

## What is already settled

**The image works on `linux/arm64`.** This was the open risk and it is now measured:
the stack was built and run on `linux/arm64` locally, all six containers healthy,
5/5 Prometheus targets up, `/cost` serving 13 policies.

```
numpy 2.4.1   pyarrow 26.0.0   h5py   lightgbm 4.6.0   torch 2.10.0+cpu
```

The pin to `torch==2.10.0` (for a segfault against LightGBM on macOS) resolves correctly
on aarch64 Linux from the CPU index. A1 is Oracle Ampere A1, so the wheels match.

**The Parquet tables ship instead of the raw archive.** N-CMAPSS is 14.6 GB of HDF5 from
NASA; converting needs that archive plus five minutes of a free ARM core. The converted
tables are 1.3 GB and `prognos convert` has already produced them, with a manifest and
checksums committed.

## 1. Create the instance — you do this, not me

I cannot create resources in your OCI account. It needs your credentials, and those are
yours to place.

1. **Sign up** at <https://www.oracle.com/cloud/free/>. A1 is the Ampere A1 shape: 4
   OCPUs and 24 GB RAM, free forever, in a home region that must be near you.
2. **Create the instance.** Oracle Linux 9 (aarch64), shape `VM.Standard.A1.Flex`,
   4 OCPU / 24 GB. Add your SSH public key — the script below needs key auth, not a
   password. Attach a boot volume of **200 GB** (the free tier allows 10× the instance's
   RAM in block storage; 1.3 GB of Parquet plus the image would not fit in the 50 GB
   default).
3. **Open the ports.** Ingress rules in the VCN security list, source restricted to your
   own address:

   | Port | Purpose |
   | --- | --- |
   | 22 | SSH, from your IP only |
   | 8084 | dashboard |
   | 9090 | Prometheus |
   | 3000 | Grafana |

   **Do not open 8081–8083.** Those are internal only and have no reason to face a
   network.

4. **Record the public IP** and put it in your shell:

   ```bash
   export PROGNOS_HOST=203.0.113.24   # your instance's public IP
   ```

## 2. Prepare this machine

```bash
TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')
echo "PROGNOS_API_TOKEN=$TOKEN" >> .env
printf '%s' "$TOKEN" > ops/prometheus-token.txt
```

Both files are gitignored. The token file is required, not optional: Prometheus scrapes
with `bearer_token_file`, and without it all four prognos targets read DOWN with 401
while every container still reports healthy.

Confirm the stack works here before pushing it anywhere:

```bash
docker compose up -d --build
ops/provision-oci.sh verify http://127.0.0.1
```

## 3. Push and start

```bash
ops/provision-oci.sh push
```

This rsyncs the source, configs, `ops/`, the Parquet tables and the model artifacts,
writes the token and `.env` with mode 600, then runs `docker compose up -d --build`
remotely. Expect a few minutes; most of it is the 1.3 GB transfer.

## 4. Verify

```bash
ops/provision-oci.sh verify "http://$PROGNOS_HOST"
```

The same command checks a laptop and the instance, so a pass means the same thing in
both places. It asserts the dashboard routes, that an absent and a wrong token are both
refused, that the cost panel has content, and that all five Prometheus targets are up.

## If it fails

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ssh: Permission denied` | Wrong key or user | `ssh -i ~/.oci/prognos_key opc@$PROGNOS_HOST` |
| `no model artifacts` | Never trained | `uv run prognos stream --model-dir artifacts/model` |
| Prometheus 1/5 up, four DOWN | Token file missing on the host | Re-run `push`; the file is written with mode 600 |
| `inference` never becomes ready | `artifacts/model` absent or empty | Inference stays unready without it, by design |
| Port unreachable | Security list, not the app | The container binds `0.0.0.0`; the host firewall is separate |

## What is deliberately not here

- **No TLS.** The token is a shared secret over plain HTTP. It tells you the caller was
  holding the right string, not who they were. Put a TLS terminator in front before this
  faces anything but your own IP.
- **No per-user identity.** One token, everyone. For a demo on a restricted source that
  is a defensible choice; for anything else it is not.
- **No automated redeploy.** `ops/provision-oci.sh push` is the whole process. There is
  no CI runner on the instance, because provisioning one is not free.
- **No cost figures beyond the placeholder.** The failure-to-planned cost ratio is 25×,
  grounded in published turbofan economics rather than numbers you have confirmed. The
  sweep over 1/5/10/25/50/100 is committed so the conclusion's dependence on that
  assumption is visible rather than hidden.

## Cost

Free, and it stays free while it stays inside the Always Free allowance: 4 OCPU / 24 GB
compute and up to 200 GB block storage. The images and containers are the only storage
besides the Parquet tables. Nothing in the compose file pulls from a paid registry, and
the three base images are pinned by digest.

If Oracle emails about idle resource reclamation, that is the known behaviour of their
free tier and not a failure of this deployment.