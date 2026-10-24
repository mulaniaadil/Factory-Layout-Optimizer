"""
Factory Layout Optimization Engine.
Implements Simulated Annealing heuristic and Random Search baseline with strict
geometric constraint feasibility gating and convergence tracking.
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from src.layout import FactoryLayout, Machine
from src.constraints import is_layout_feasible, validate_layout, FeasibilityReport
from src.distance import compute_layout_objective


@dataclass
class OptimizationHistoryPoint:
    iteration: int
    temperature: float
    current_cost: float
    best_cost: float
    is_feasible: bool
    accepted: bool


@dataclass
class OptimizationResult:
    algorithm_name: str
    status: str  # 'SUCCESS', 'INFEASIBLE_INPUT', 'NO_FEASIBLE_FOUND'
    initial_layout: FactoryLayout
    best_layout: FactoryLayout
    initial_cost: float
    best_cost: float
    improvement_percent: float
    iterations_run: int
    runtime_seconds: float
    seed: int
    distance_metric: str
    feasibility_report: FeasibilityReport
    history: List[OptimizationHistoryPoint] = field(default_factory=list)
    message: str = ""


class SimulatedAnnealingOptimizer:
    """
    Simulated Annealing heuristic for Quadratic Assignment / Facility Layout Problem (FLP).
    Explores continuous 2D coordinate space snapped to discrete grid resolution.
    """
    def __init__(
        self,
        initial_temperature: float = 500.0,
        cooling_rate: float = 0.985,
        min_temperature: float = 0.1,
        max_iterations: int = 1500,
        step_size: float = 1.0,
        random_seed: int = 42,
        distance_metric: str = "manhattan",
        enforce_clearance: bool = False
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.random_seed = random_seed
        self.distance_metric = distance_metric
        self.enforce_clearance = enforce_clearance

    def _snap_to_grid(self, val: float, resolution: float) -> float:
        return round(round(val / resolution) * resolution, 3)

    def _generate_candidate(self, layout: FactoryLayout, rng: random.Random) -> Optional[FactoryLayout]:
        """Propose a valid perturbation: move a movable machine or swap two machines."""
        candidate = layout.clone()
        movable = [m for m in candidate.machines if not m.is_fixed]
        if not movable:
            return None

        W = candidate.factory_width
        H = candidate.factory_height
        res = candidate.grid_resolution

        # Strategy 1 (80% probability): Perturb single machine position
        # Strategy 2 (20% probability): Coordinate swap between two movable machines
        move_type = rng.random()

        if move_type < 0.8 or len(movable) < 2:
            m = rng.choice(movable)
            max_x = max(0.0, W - m.width)
            max_y = max(0.0, H - m.height)

            # Draw random displacement within [-step_size, +step_size]
            dx = rng.choice([-self.step_size, 0.0, self.step_size, -2 * self.step_size, 2 * self.step_size])
            dy = rng.choice([-self.step_size, 0.0, self.step_size, -2 * self.step_size, 2 * self.step_size])

            new_x = min(max(0.0, m.x + dx), max_x)
            new_y = min(max(0.0, m.y + dy), max_y)

            m.x = self._snap_to_grid(new_x, res)
            m.y = self._snap_to_grid(new_y, res)
        else:
            # Swap coordinates of two movable machines
            m1, m2 = rng.sample(movable, 2)
            max_x1 = max(0.0, W - m1.width)
            max_y1 = max(0.0, H - m1.height)
            max_x2 = max(0.0, W - m2.width)
            max_y2 = max(0.0, H - m2.height)

            new_x1 = min(max(0.0, m2.x), max_x1)
            new_y1 = min(max(0.0, m2.y), max_y1)
            new_x2 = min(max(0.0, m1.x), max_x2)
            new_y2 = min(max(0.0, m1.y), max_y2)

            m1.x = self._snap_to_grid(new_x1, res)
            m1.y = self._snap_to_grid(new_y1, res)
            m2.x = self._snap_to_grid(new_x2, res)
            m2.y = self._snap_to_grid(new_y2, res)

        return candidate

    def optimize(
        self,
        initial_layout: FactoryLayout,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> OptimizationResult:
        start_time = time.time()
        rng = random.Random(self.random_seed)

        # Record fixed machine original coordinates
        orig_fixed = {m.id: (m.x, m.y) for m in initial_layout.machines if m.is_fixed}

        # Check initial layout feasibility
        init_feasibility = validate_layout(
            initial_layout,
            original_fixed_positions=orig_fixed,
            enforce_safety_clearance=self.enforce_clearance
        )
        initial_cost, _ = compute_layout_objective(initial_layout, self.distance_metric)

        current_layout = initial_layout.clone()
        current_cost = initial_cost
        best_layout = initial_layout.clone()
        best_cost = initial_cost if init_feasibility.is_feasible else float("inf")

        temperature = self.initial_temperature
        history: List[OptimizationHistoryPoint] = []

        for it in range(1, self.max_iterations + 1):
            candidate = self._generate_candidate(current_layout, rng)
            accepted = False

            if candidate is not None:
                # Strict feasibility gating: candidate must not violate physical boundaries or overlap
                candidate_feasible = is_layout_feasible(candidate, enforce_clearance=self.enforce_clearance)

                if candidate_feasible:
                    candidate_cost, _ = compute_layout_objective(candidate, self.distance_metric)
                    delta = candidate_cost - current_cost

                    # Metropolis criterion
                    if delta < 0:
                        accepted = True
                    else:
                        # Acceptance probability P = exp(-delta / T)
                        p = math.exp(-delta / max(temperature, 1e-6))
                        if rng.random() < p:
                            accepted = True

                    if accepted:
                        current_layout = candidate
                        current_cost = candidate_cost

                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_layout = candidate.clone()

            # Record history snapshot
            history.append(
                OptimizationHistoryPoint(
                    iteration=it,
                    temperature=temperature,
                    current_cost=current_cost,
                    best_cost=best_cost if best_cost < float("inf") else current_cost,
                    is_feasible=best_cost < float("inf"),
                    accepted=accepted
                )
            )

            # Temperature decay
            temperature = max(self.min_temperature, temperature * self.cooling_rate)

            # Optional UI callback
            if progress_callback and (it % 25 == 0 or it == self.max_iterations):
                progress_callback(it, self.max_iterations, current_cost, best_cost if best_cost < float("inf") else current_cost)

        elapsed = time.time() - start_time

        # Validate final solution
        final_feasibility = validate_layout(
            best_layout,
            original_fixed_positions=orig_fixed,
            enforce_safety_clearance=self.enforce_clearance
        )

        if not final_feasibility.is_feasible:
            status = "NO_FEASIBLE_FOUND"
            imp_pct = 0.0
            msg = "Could not find a feasible layout under given constraints."
        else:
            status = "SUCCESS"
            if init_feasibility.is_feasible and initial_cost > 0:
                imp_pct = ((initial_cost - best_cost) / initial_cost) * 100.0
            else:
                imp_pct = 0.0
            msg = f"Feasible layout found with {imp_pct:.2f}% improvement."

        return OptimizationResult(
            algorithm_name="Simulated Annealing",
            status=status,
            initial_layout=initial_layout,
            best_layout=best_layout,
            initial_cost=initial_cost,
            best_cost=best_cost if best_cost < float("inf") else initial_cost,
            improvement_percent=imp_pct,
            iterations_run=self.max_iterations,
            runtime_seconds=elapsed,
            seed=self.random_seed,
            distance_metric=self.distance_metric,
            feasibility_report=final_feasibility,
            history=history,
            message=msg
        )


class RandomSearchOptimizer:
    """Baseline heuristic: generates random feasible layouts and selects the minimum cost arrangement."""
    def __init__(
        self,
        max_iterations: int = 500,
        random_seed: int = 42,
        distance_metric: str = "manhattan"
    ):
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.distance_metric = distance_metric

    def optimize(
        self,
        initial_layout: FactoryLayout,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> OptimizationResult:
        start_time = time.time()
        rng = random.Random(self.random_seed)

        orig_fixed = {m.id: (m.x, m.y) for m in initial_layout.machines if m.is_fixed}
        init_feasibility = validate_layout(initial_layout, original_fixed_positions=orig_fixed)
        initial_cost, _ = compute_layout_objective(initial_layout, self.distance_metric)

        best_layout = initial_layout.clone()
        best_cost = initial_cost if init_feasibility.is_feasible else float("inf")
        history: List[OptimizationHistoryPoint] = []

        W = initial_layout.factory_width
        H = initial_layout.factory_height
        res = initial_layout.grid_resolution

        for it in range(1, self.max_iterations + 1):
            candidate = initial_layout.clone()
            for m in candidate.machines:
                if not m.is_fixed:
                    max_x = max(0.0, W - m.width)
                    max_y = max(0.0, H - m.height)
                    m.x = round(round(rng.uniform(0.0, max_x) / res) * res, 3)
                    m.y = round(round(rng.uniform(0.0, max_y) / res) * res, 3)

            accepted = False
            if is_layout_feasible(candidate):
                cost, _ = compute_layout_objective(candidate, self.distance_metric)
                if cost < best_cost:
                    best_cost = cost
                    best_layout = candidate.clone()
                    accepted = True

            history.append(
                OptimizationHistoryPoint(
                    iteration=it,
                    temperature=0.0,
                    current_cost=best_cost if best_cost < float("inf") else initial_cost,
                    best_cost=best_cost if best_cost < float("inf") else initial_cost,
                    is_feasible=best_cost < float("inf"),
                    accepted=accepted
                )
            )

            if progress_callback and (it % 25 == 0 or it == self.max_iterations):
                progress_callback(it, self.max_iterations, best_cost, best_cost)

        elapsed = time.time() - start_time
        final_feasibility = validate_layout(best_layout, original_fixed_positions=orig_fixed)

        if not final_feasibility.is_feasible:
            status = "NO_FEASIBLE_FOUND"
            imp_pct = 0.0
            msg = "Random baseline could not find a feasible arrangement."
        else:
            status = "SUCCESS"
            imp_pct = ((initial_cost - best_cost) / initial_cost * 100.0) if (init_feasibility.is_feasible and initial_cost > 0) else 0.0
            msg = f"Random search found feasible layout with {imp_pct:.2f}% improvement."

        return OptimizationResult(
            algorithm_name="Random Search Baseline",
            status=status,
            initial_layout=initial_layout,
            best_layout=best_layout,
            initial_cost=initial_cost,
            best_cost=best_cost if best_cost < float("inf") else initial_cost,
            improvement_percent=imp_pct,
            iterations_run=self.max_iterations,
            runtime_seconds=elapsed,
            seed=self.random_seed,
            distance_metric=self.distance_metric,
            feasibility_report=final_feasibility,
            history=history,
            message=msg
        )
