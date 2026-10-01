"""Settings, from the environment. Mirrors the audio app's shape."""
import os
import pathlib


class S:
    root = pathlib.Path(os.environ.get("IMAGEAI_ROOT",
                        pathlib.Path(__file__).resolve().parent.parent))
    app = pathlib.Path(__file__).resolve().parent
    sd_bin = os.environ.get("SD_BIN", "")          # resolved in registry if empty
    out = pathlib.Path(os.environ.get("IMAGEAI_OUT", root / "outputs"))
    data_dir = pathlib.Path(os.environ.get("IMAGEAI_DATA", root / "data"))
    db_path = data_dir / "app.db"
    # Creating costs a GPU, so it is gated. Empty password => creating is allowed
    # from private IPs only, so an unconfigured box is never open to the internet.
    owner_password = os.environ.get("OWNER_PASSWORD", "")


def settings():
    s = S()
    S.out.mkdir(parents=True, exist_ok=True)
    S.data_dir.mkdir(parents=True, exist_ok=True)
    return s
