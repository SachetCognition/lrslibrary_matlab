import os
from pathlib import Path

class LRSConfig:
    def __init__(self):
        self.lrs_dir = Path(__file__).parent.parent.absolute()
        self.algorithms_path = self.lrs_dir / "lrslibrary" / "algorithms"
        
        self.rpca_path = self.algorithms_path / "rpca"
        self.st_path = self.algorithms_path / "st"
        self.mc_path = self.algorithms_path / "mc"
        self.lrr_path = self.algorithms_path / "lrr"
        self.ttd_path = self.algorithms_path / "ttd"
        self.nmf_path = self.algorithms_path / "nmf"
        self.ntf_path = self.algorithms_path / "ntf"
        self.td_path = self.algorithms_path / "td"

_config = None

def get_config():
    global _config
    if _config is None:
        _config = LRSConfig()
    return _config
