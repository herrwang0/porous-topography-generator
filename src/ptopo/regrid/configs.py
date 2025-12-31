"""
configs.py

Pure helper classes for various configurations
"""

from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional
from enum import Enum

@dataclass
class RefineConfig:
    """A container for GMesh.GMesh.refine() options"""
    fixed_refine_level : int = -1
    resolution_limit : bool = False
    use_center : bool = True
    work_in_3d : bool = False
    singularity_radius : float = 0.25
    max_mb : float = 32000
    max_stages : int = 32

    def to_kwargs(self):
        return asdict(self)

    def print_options(self, indent=0):
        pad = " " * indent
        options = asdict(self)
        max_len = max(len(key) for key in options.keys())
        print(f"{pad}RefineConfig Options:")
        for key, value in options.items():
            print(f"{pad}  {key.ljust(max_len)} : {value}")

@dataclass
class CalcConfig:
    """A container for regridding options"""
    calc_cell_stats: bool = True
    _thinwalls: bool = True
    _effective_tw: bool = False
    thinwalls_interp: str = 'max'
    calc_roughness: bool = False
    calc_gradient: bool = False
    save_hits : bool = False

    def __post_init__(self):
        if not self.calc_thinwalls and self.thinwalls_interp is not None:
            self.thinwalls_interp = None

    @property
    def calc_thinwalls(self):
        return self._thinwalls and self.calc_cell_stats

    @property
    def calc_effective_tw(self):
        if self.calc_thinwalls:
            return self.calc_thinwalls and self._effective_tw
        else:
            return None

    def print_options(self, indent=0):
        pad = " " * indent

        order = [
            "calc_cell_stats",
            "calc_thinwalls",
            "thinwalls_interp",
            "calc_effective_tw",
            "calc_roughness",
            "calc_gradient",
            "save_hits"
        ]

        # Build a dict of all values, respecting conditions
        values = {
            "calc_cell_stats": self.calc_cell_stats,
            "calc_thinwalls": self.calc_thinwalls,
            "thinwalls_interp": self.thinwalls_interp,
            "calc_effective_tw": self.calc_effective_tw if self.calc_thinwalls else None,
            "calc_roughness": self.calc_roughness,
            "calc_gradient": self.calc_gradient,
            "save_hits": self.save_hits
        }

        max_len = max(len(k) for k in order)
        print(f"{pad}CalcConfig Options:")
        for key in order:
            # Skip thinwalls_interp and calc_effective_tw if calc_thinwalls is False
            if key in ["thinwalls_interp", "calc_effective_tw"] and not self.calc_thinwalls:
                continue
            print(f"{pad}  {key.ljust(max_len)} : {values[key]}")

@dataclass
class TileConfig:
    """Configuration for domain tiling / decomposition"""

    # Tiling
    pelayout : Tuple[int, int] = (1, 1)
    tgt_halo : int = 0
    symmetry : Tuple[bool, bool] = (False, True)
    norm_lon : bool | None = None

    # Source data
    subset_eds: bool = True
    src_halo: int = 0

    # Stitch option
    bnd_tol_level: int = 2

    def __post_init__(self):
        # Normalize pelayout
        self.pelayout = tuple(self.pelayout)
        if len(self.pelayout) != 2:
            raise ValueError("pelayout must be a 2-tuple (px, py)")

        # Basic validation
        if self.tgt_halo < 0 or self.src_halo < 0:
            raise ValueError("tgt_halo and src_halo must be >= 0")

        if self.bnd_tol_level < 0:
            raise ValueError("bnd_tol_level must be >= 0")

    @property
    def ntiles(self) -> int:
        """Total number of tiles"""
        return self.pelayout[0] * self.pelayout[1]

    def print_options(self, indent=0):
        pad = " " * indent

        order = [
            "pelayout",
            "tgt_halo",
            "symmetry",
            "subset_eds",
            "src_halo",
            "bnd_tol_level",
        ]

        values = {
            "pelayout": self.pelayout,
            "tgt_halo": self.tgt_halo,
            "symmetry": self.symmetry,
            "subset_eds": self.subset_eds,
            "src_halo": self.src_halo,
            "bnd_tol_level": self.bnd_tol_level,
        }

        max_len = max(len(k) for k in order)
        print(f"{pad}TileConfig Options:")
        for key in order:
            print(f"{pad}  {key.ljust(max_len)} : {values[key]}")

class NorthPoleMode(str, Enum):
    MASK_ONLY = "mask"
    RING_UPDATE = "ring"

@dataclass
class NorthPoleConfig:
    """Configuration for North Pole handling"""

    mode: NorthPoleMode = NorthPoleMode.MASK_ONLY

    # Shared
    lat_start: float = 85.0

    # Ring-update parameters (only used if mode == RING_UPDATE)
    lat_stop: Optional[float] = None
    lat_step: Optional[float] = None
    pole_halo : int = 0
    tile_cfg: TileConfig = field(default_factory=TileConfig)

    def __post_init__(self):
        if self.mode == NorthPoleMode.MASK_ONLY:
            # Enforce unused parameters are unset
            self.lat_stop = None
            self.lat_step = None

        elif self.mode == NorthPoleMode.RING_UPDATE:
            if self.lat_stop is None or self.lat_step is None:
                raise ValueError(
                    "lat_stop and lat_step must be set for ring update mode"
                )
            if self.lat_step <= 0:
                raise ValueError("lat_step must be positive")
            if self.lat_start >= self.lat_stop:
                raise ValueError("lat_start must be < lat_stop")

    def print_options(self, indent=0):
        pad = " " * indent
        print(f"{pad}NorthPoleConfig Options:")
        print(f"{pad}  mode       : {self.mode.value}")
        print(f"{pad}  lat_start  : {self.lat_start}")

        if self.mode == NorthPoleMode.RING_UPDATE:
            print(f"{pad}  lat_stop   : {self.lat_stop}")
            print(f"{pad}  lat_step   : {self.lat_step}")
            print(f"{pad}  pole_halo  : {self.pole_halo}")
            # Nested TileConfig
            if self.tile_cfg is not None:
                self.tile_cfg.print_options(indent=indent + 2)