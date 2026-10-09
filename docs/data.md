# Dataset

N-CMAPSS is redistributed by NASA and is **not** committed here. Only checksums
and derived Parquet (which is also gitignored) refer to it.

Download (14.68 GiB, ~25 min on a 100 Mbit link):

```bash
curl -L -o ~/data/ncmapss/17.Turbofan2.zip \
  "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
```

Verify against the checksum below before extracting:

```bash
shasum -a 256 ~/data/ncmapss/17.Turbofan2.zip
# d1271732485dc1ed354e8c7d950edc196aa78ec48d78f38f2d9681dc82fb61c2
```

The archive is a zip64 wrapper around a nested `data_set.zip`. Extract the
sub-datasets in scope:

```bash
uv run python - <<'PY'
import io, pathlib, zipfile

archive = pathlib.Path.home() / "data/ncmapss/17.Turbofan2.zip"
out = pathlib.Path.home() / "data/ncmapss/extracted"
out.mkdir(parents=True, exist_ok=True)

inner = "17. Turbofan Engine Degradation Simulation Data Set 2/data_set.zip"
want = {
    "data_set/N-CMAPSS_DS01-005.h5",
    "data_set/N-CMAPSS_DS02-006.h5",
    "data_set/N-CMAPSS_DS06.h5",
}

inner_zip = out / "data_set.zip"
if not inner_zip.exists():
    with zipfile.ZipFile(archive) as outer:
        with outer.open(inner) as src, inner_zip.open("wb") as dst:
            while chunk := src.read(8 << 20):
                dst.write(chunk)

with zipfile.ZipFile(inner_zip) as zf:
    for name in want:
        zf.extract(name, out)
PY
```

Requires ~16 GiB of free disk for the archive plus the extracted HDF5. Delete
`17.Turbofan2.zip` and `extracted/data_set.zip` once extraction succeeds.

## Why only these sub-datasets

Full archive contents, measured 2026-10-09:

| File | Size | In scope |
| --- | --- | --- |
| `N-CMAPSS_DS01-005.h5` | 2,873 MB | yes — single failure mode, single flight class |
| `N-CMAPSS_DS02-006.h5` | 2,450 MB | yes — primary; the most-cited in the literature |
| `N-CMAPSS_DS06.h5` | 2,549 MB | yes — five failure modes |
| `N-CMAPSS_DS03-012.h5` | 3,693 MB | no |
| `N-CMAPSS_DS04.h5` | 3,753 MB | no |
| `N-CMAPSS_DS05.h5` | 2,599 MB | no |
| `N-CMAPSS_DS07.h5` | 2,715 MB | no |
| `N-CMAPSS_DS08a-009.h5` | 3,237 MB | no |
| `N-CMAPSS_DS08c-008.h5` | 2,413 MB | no |
| `N-CMAPSS_DS08d-010.h5` | 2,885 MB | no — its HDF5 is unreadable in published benchmark work |

## Convert

```bash
uv run prognos convert --config configs/base.yaml
```

29 units, 20,938,714 rows, ~13 s on an Apple M5, 1.3 GiB of Parquet. Writes
`data/manifest.json` with a SHA-256 per input and per output file.

Then:

```bash
uv run pytest -m integration
```

## Citation

M. Chao, C. Kulkarni, K. Goebel, O. Fink, *Aircraft Engine Run-to-Failure
Dataset under real flight conditions*, NASA Prognostics Center of Excellence Data
Repository, NASA Ames Research Center, 2021.
https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

The repository asks that publications acknowledge its assistance and the data
donors. Terms of use: no restriction on use, no warranty, used at your own risk.