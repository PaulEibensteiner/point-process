"""Poisson rate model for the Züri wie neu reports with a view-dependent optimal
basis, and a background job manager that refits it for the current map view."""

from dataclasses import dataclass
import threading
import time
import traceback
from typing import Callable

import numpy as np
import torch
from nmf.nmf_models import NMFBatchHALS
from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator

from utils import get_zueri_wie_neu_data, lonlat_to_mercator, mercator_to_lonlat

# iteration cap of the NMF in stpy's add_new_functions (batch_max_iter)
NMF_MAX_ITER = 100

# run_nmf has no callback, so hook into its HALS iterations to report progress and
# to cancel. Only the worker thread runs NMF, so a module level hook is enough.
_nmf_iteration_hook: Callable[[], None] | None = None
_hals_update_W = NMFBatchHALS._update_W


def _hals_update_W_with_hook(self):
    _hals_update_W(self)
    if _nmf_iteration_hook is not None:
        _nmf_iteration_hook()


NMFBatchHALS._update_W = _hals_update_W_with_hook

# length of the edge of the square cell that "expected reports per year" refers to
CELL_SIZE_M = 1000.0
EARTH_RADIUS_M = 6371008.8


def pixel_centers(a: float, b: float, n: int) -> np.ndarray:
    edges = np.linspace(a, b, n + 1)
    return (edges[:-1] + edges[1:]) / 2


class Cancelled(Exception):
    pass


@dataclass
class RateImage:
    """Rate on a regular Web Mercator grid with pixel edges (x0, y0, x1, y1), sampled
    at the pixel centers, Z[row, col] with row along increasing y."""

    job_id: int
    x0: float
    y0: float
    x1: float
    y1: float
    Z: np.ndarray  # expected reports per year per CELL_SIZE_M x CELL_SIZE_M cell


class GraffitiModel:
    def __init__(
        self,
        service_code: str | None = "Graffiti",
        gamma: float = 0.2,
        kappa: float = 1.0,
        m: int = 25,
        levels: int = 8,
        roi_resolution: int = 30,
        roi_num_basis_functions: int = 9,
        render_resolution: int = 200,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.roi_resolution = roi_resolution
        self.roi_num_basis_functions = roi_num_basis_functions
        self.render_resolution = render_resolution

        obs, self.dt, self.gdf = get_zueri_wie_neu_data(
            None, dtype=np.float64, service_code=service_code
        )
        self.obs = obs.to(device)
        self.left, self.down, self.right, self.up = self.gdf.total_bounds
        D = BorelSet(2, bounds=torch.tensor([[-1.0, 1.0], [-1, 1]]).double())
        self.data = [(D, self.obs, self.dt)]

        self._estimator_kwargs = dict(gamma=gamma, kappa=kappa, m=m, levels=levels)
        self.estimator: PoissonRateEstimator | None = None

        # rate is per normalized area and per day -> reports per year per cell
        mid_lat = np.deg2rad((self.down + self.up) / 2)
        width_m = np.deg2rad(self.right - self.left) * EARTH_RADIUS_M * np.cos(mid_lat)
        height_m = np.deg2rad(self.up - self.down) * EARTH_RADIUS_M
        cell_area_normalized = CELL_SIZE_M**2 * (2 / width_m) * (2 / height_m)
        self.to_reports_per_year = 365.25 * cell_area_normalized

        mx0, my0 = lonlat_to_mercator(self.left, self.down)
        mx1, my1 = lonlat_to_mercator(self.right, self.up)
        self.mercator_bounds = (mx0, my0, mx1, my1)

    def _normalize(self, lon, lat):
        x = 2 * (lon - self.left) / (self.right - self.left) - 1
        y = 2 * (lat - self.down) / (self.up - self.down) - 1
        return x, y

    def fit_initial(self, stage: Callable[[str], None]):
        """Build the global optimal basis (slow) and fit it to all data."""
        p = self._estimator_kwargs
        stage("build")
        self.estimator = PoissonRateEstimator(
            HierarchicalBorelSets(d=2, interval=[(-1, 1), (-1, 1)], levels=p["levels"]),
            d=2,
            kernel=KernelFunction(
                kernel_name="squared_exponential",
                gamma=p["gamma"],
                kappa=p["kappa"],
                d=2,
            ),
            max_intensity=1e7,
            min_intensity=0.0,
            basis_size_per_dim=p["m"],
            langevine_sampling_steps=200,
            optimization_library="torch",
            use_anchors=False,
            basis="optimal-positive",
            samples_nystrom=500,
            roi=self.obs,
            memory_limit=1,
            device=self.device,
        )

        stage("fit")
        self.estimator.load_data(self.data)
        self.estimator.fit_gp()

    def refine(
        self,
        view: tuple[float, float, float, float],
        stage: Callable[[str], None],
        callback: Callable[[], None],
        nmf_iteration: Callable[[], None] | None = None,
    ) -> tuple[float, float, float, float, np.ndarray]:
        """Add basis functions for the Web Mercator view (x0, y0, x1, y1), refit and
        evaluate the rate there. ``stage`` is told the stage name before each stage,
        ``callback`` is called often and may raise Cancelled."""
        bx0, by0, bx1, by1 = self.mercator_bounds
        x0, y0, x1, y1 = view
        x0, x1 = max(x0, bx0), min(x1, bx1)
        y0, y1 = max(y0, by0), min(y1, by1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError("view does not overlap the data")
        x0, y0, x1, y1 = self.snap_to_initial_grid(x0, y0, x1, y1)

        (lon0, lon1), (lat0, lat1) = mercator_to_lonlat(
            np.array([x0, x1]), np.array([y0, y1])
        )
        (nx0, nx1), (ny0, ny1) = self._normalize(
            np.array([lon0, lon1]), np.array([lat0, lat1])
        )

        stage("basis")
        # stay off the domain boundary, points on it give degenerate triangles in
        # the basis interpolation
        inset = 1e-3
        gx, gy = torch.meshgrid(
            torch.linspace(
                max(nx0, -1 + inset), min(nx1, 1 - inset), self.roi_resolution
            ),
            torch.linspace(
                max(ny0, -1 + inset), min(ny1, 1 - inset), self.roi_resolution
            ),
            indexing="ij",
        )
        roi = torch.stack([gx.flatten(), gy.flatten()], dim=-1).to(self.device)
        global _nmf_iteration_hook
        _nmf_iteration_hook = nmf_iteration
        try:
            self.estimator.packing.add_new_functions(roi, self.roi_num_basis_functions)
        finally:
            _nmf_iteration_hook = None
        self.estimator.m = self.estimator.packing.m
        callback()

        stage("data")
        self.estimator.load_data(self.data)
        callback()

        stage("fit")
        self.estimator.fit_gp(callback=callback)

        stage("render")
        return (x0, y0, x1, y1, self.evaluate(x0, y0, x1, y1))

    def snap_to_initial_grid(self, x0, y0, x1, y1):
        """Grow the rectangle to the pixel edges of the initial full-area image, so
        the view image replaces whole pixels of it without gaps or overlaps."""
        bx0, by0, bx1, by1 = self.mercator_bounds
        n = self.render_resolution
        px, py = (bx1 - bx0) / n, (by1 - by0) / n
        return (
            bx0 + np.floor((x0 - bx0) / px) * px,
            by0 + np.floor((y0 - by0) / py) * py,
            bx0 + np.ceil((x1 - bx0) / px) * px,
            by0 + np.ceil((y1 - by0) / py) * py,
        )

    def evaluate(self, x0, y0, x1, y1) -> np.ndarray:
        """Rate at the pixel centers of an n x n image with edges (x0, y0, x1, y1)."""
        n = self.render_resolution
        mx, my = np.meshgrid(pixel_centers(x0, x1, n), pixel_centers(y0, y1, n))
        lon, lat = mercator_to_lonlat(mx, my)
        nx, ny = self._normalize(lon, lat)
        pts = torch.from_numpy(np.stack([nx.ravel(), ny.ravel()], axis=1)).to(
            self.device
        )
        with torch.no_grad():
            rate = self.estimator.rate_value(pts)
        return rate.reshape(n, n).cpu().numpy() * self.to_reports_per_year


# share of the refit progress bar for the basis stage, advanced per NMF iteration;
# the other stages share the rest by their measured durations
BASIS_SHARE = 1 / 3
# initial guess of the stage durations in seconds, replaced by measurements
STAGE_SECONDS = {"build": 30.0, "basis": 5.0, "data": 1.0, "fit": 1.0, "render": 0.2}
PROGRESS_INTERVAL_S = 0.2


class FitJobManager:
    """Runs fits of a GraffitiModel on one worker thread. A new ``submit`` cancels
    the running job at its next checkpoint and starts over.

    Progress is estimated from the elapsed time of each stage relative to how long
    that stage took before, since the NMF inside the basis stage cannot report
    progress. Listeners are called from background threads."""

    def __init__(self, model: GraffitiModel):
        self.model = model
        self.job_id = 0
        self._pending: tuple | None = None
        self._cond = threading.Condition()
        self._listeners: list = []
        self._seconds = dict(STAGE_SECONDS)
        self._stages: list[str] = ["build", "fit"]
        self._stage: str | None = None
        self._stage_start = 0.0
        self._nmf_iters = 0
        self._done_seconds = 0.0
        self._running_job = 0
        self.last_result: RateImage | None = None
        # density of the global basis only over the whole data area, job_id 0
        self.initial_result: RateImage | None = None
        self.progress = (0, 0.0, "starting")  # (job_id, fraction, stage)
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._tick, daemon=True).start()

    def subscribe(self, on_progress, on_result):
        self._listeners.append((on_progress, on_result))

    def unsubscribe(self, on_progress, on_result):
        self._listeners.remove((on_progress, on_result))

    def submit(self, view: tuple[float, float, float, float]) -> int:
        with self._cond:
            self.job_id += 1
            self._pending = (self.job_id, view)
            self._cond.notify()
        self._report(self.job_id, 0.0, "queued")
        return self.job_id

    def _report(self, job_id, fraction, stage):
        if job_id != self.job_id:
            return
        if self.progress == (job_id, round(fraction, 2), stage):
            return
        self.progress = (job_id, round(fraction, 2), stage)
        for on_progress, _ in list(self._listeners):
            on_progress(*self.progress)

    def _fraction(self) -> float:
        if self._stage is None:
            return 0.0
        if self._stage == "basis":
            return BASIS_SHARE * min(self._nmf_iters, NMF_MAX_ITER) / NMF_MAX_ITER
        offset, share = (
            (BASIS_SHARE, 1 - BASIS_SHARE) if "basis" in self._stages else (0, 1)
        )
        total = sum(self._seconds[s] for s in self._stages if s != "basis")
        elapsed = time.time() - self._stage_start
        current = min(elapsed, 0.95 * self._seconds[self._stage])
        return offset + share * (self._done_seconds + current) / total

    def _tick(self):
        while True:
            time.sleep(PROGRESS_INTERVAL_S)
            job_id, stage = self._running_job, self._stage
            if stage is not None:
                self._report(job_id, self._fraction(), stage)

    def _tracker(self, job_id, stages):
        self._running_job = job_id
        self._stages = stages
        self._stage = None
        self._done_seconds = 0.0
        self._nmf_iters = 0
        measured = {}

        def end_stage():
            if self._stage is not None:
                measured[self._stage] = time.time() - self._stage_start
                if self._stage != "basis":
                    self._done_seconds += self._seconds[self._stage]

        def stage(name):
            end_stage()
            self._stage, self._stage_start = name, time.time()
            self._report(job_id, self._fraction(), name)

        def callback():
            if job_id != self.job_id:
                raise Cancelled()

        def nmf_iteration():
            callback()
            self._nmf_iters += 1

        def finish():
            end_stage()
            self._stage = None
            for name, seconds in measured.items():
                self._seconds[name] = 0.5 * self._seconds[name] + 0.5 * seconds

        return stage, callback, finish, nmf_iteration

    def _run(self):
        # the default device is thread local
        torch.set_default_device(self.model.device)
        torch.set_default_dtype(torch.float64)

        # initial global fit, counts as job 0; its duration is not representative
        stage, _, finish, _ = self._tracker(0, ["build", "fit"])
        self.model.fit_initial(stage)
        bounds = self.model.mercator_bounds
        self.initial_result = RateImage(0, *bounds, self.model.evaluate(*bounds))
        finish()
        for _, on_result in list(self._listeners):
            on_result(self.initial_result)
        self._report(0, 1.0, "ready")

        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
                job_id, view = self._pending
                self._pending = None
            stage, callback, finish, nmf_iteration = self._tracker(
                job_id, ["basis", "data", "fit", "render"]
            )
            start = time.time()
            try:
                x0, y0, x1, y1, Z = self.model.refine(
                    view, stage, callback, nmf_iteration
                )
            except Cancelled:
                self._stage = None
                continue
            except ValueError:
                self._stage = None
                self._report(job_id, 1.0, "outside data")
                continue
            except Exception:
                traceback.print_exc()
                self._stage = None
                self._report(job_id, 1.0, "failed, see server log")
                continue
            finish()
            if job_id != self.job_id:
                continue
            self.last_result = RateImage(job_id, x0, y0, x1, y1, Z)
            print(f"job {job_id} took {time.time() - start:.1f}s")
            for _, on_result in list(self._listeners):
                on_result(self.last_result)
            self._report(job_id, 1.0, "done")


_job_manager: FitJobManager | None = None
_job_manager_lock = threading.Lock()


def get_job_manager() -> FitJobManager:
    """The job manager shared by the whole process. The first call loads the data
    and starts the initial global fit on the worker thread."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # the default device is thread local, set it for the calling thread
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float64)
            _job_manager = FitJobManager(
                GraffitiModel(
                    service_code="Graffiti",
                    gamma=0.08,
                    kappa=6.0,
                    m=50,
                    roi_num_basis_functions=18,
                )
            )
        return _job_manager
