#!/usr/bin/env python3
"""The policy contract, plus reference policies that calibrate the metric.

A consumer plugs in a driving model by implementing `Policy`:

    class MyPlanner:
        name = "my-planner"
        def plan(self, obs: Observation) -> list[tuple[float, float]]:
            # obs.ego_history / obs.agent_history contain data up to t0 ONLY
            return [(x, y), ...]   # obs.n_steps points, EGO FRAME at t0,
                                   # +x forward, +y left, at t0+(k+1)*dt

That is the whole interface: history in, ego-frame trajectory out. No dataset
knowledge, no map, no sensor handling.

The three reference policies exist to make a score readable. A metric harness with
no calibration is untrustworthy — you cannot tell 0.62 from a bug. Expected shape:
  ReplayHuman   ~1.0  (it IS the recorded future; anything less is a harness bug)
  Stationary    NC=1, EP~0   (safe and useless — the degenerate corner)
  ConstantVel   in between, collides in interactive scenes
"""
from __future__ import annotations

import math
from typing import List, Tuple

from scenario import Observation


class Policy:
    name = "policy"
    is_oracle = False
    needs_scenario = False      # oracles only; see ReplayHuman
    needs_sensors = False       # True restricts decision times to sensor coverage

    def plan(self, obs: Observation) -> List[Tuple[float, float]]:
        raise NotImplementedError


class Stationary(Policy):
    """Do nothing. Should score NC=1, EP~0 — the safe-and-useless corner."""
    name = "stationary"

    def plan(self, obs):
        return [(0.0, 0.0) for _ in range(obs.n_steps)]


class ConstantVelocity(Policy):
    """Hold the current speed and heading. The honest naive baseline."""
    name = "constant_velocity"

    def plan(self, obs):
        e = obs.ego_state
        speed = math.hypot(e.vx, e.vy)
        dt = obs.dt_s
        return [(speed * dt * (k + 1), 0.0) for k in range(obs.n_steps)]


class ConstantTurnRate(Policy):
    """Hold current speed and the yaw rate observed over the recent history."""
    name = "constant_turn_rate"

    def plan(self, obs):
        e = obs.ego_state
        speed = math.hypot(e.vx, e.vy)
        hist = obs.ego_history[-5:]
        yaw_rate = 0.0
        if len(hist) >= 2:
            dt_h = max(1e-3, (hist[-1].t_us - hist[0].t_us) / 1e6)
            d = (hist[-1].yaw - hist[0].yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = d / dt_h
        out, x, y, th = [], 0.0, 0.0, 0.0
        for _ in range(obs.n_steps):
            th += yaw_rate * obs.dt_s
            x += speed * obs.dt_s * math.cos(th)
            y += speed * obs.dt_s * math.sin(th)
            out.append((x, y))
        return out


class ReactiveIDM(Policy):
    """Rule-based car-follower: hold the observed yaw rate, brake for the lead agent.

    The reference ladder was missing its most informative rung. `constant_velocity`
    is naive because it ignores agents entirely, and the learned models sit only
    slightly above it — but that comparison cannot separate "the model perceives
    agents" from "the metric barely rewards it", because nothing in between existed.
    This is a competent non-learned planner: it sees the same agent tracks the
    metric scores against and reacts to them with IDM-style longitudinal control.

    Map-free by construction — it uses only the ego corridor and agent positions, so
    it runs on any dataset the adapter supports.
    """

    name = "reactive_idm"

    # IDM-ish constants (nuPlan-flavoured, deliberately unremarkable)
    DESIRED_GAP_S = 1.5      # time headway
    MIN_GAP_M = 5.0
    MAX_DECEL = 3.0          # m/s^2, comfortable
    CORRIDOR_HALF_W = 1.8    # m, lateral half-width of the "my lane" test

    def plan(self, obs):
        e = obs.ego_state
        speed = math.hypot(e.vx, e.vy)

        hist = obs.ego_history[-5:]
        yaw_rate = 0.0
        if len(hist) >= 2:
            dt_h = max(1e-3, (hist[-1].t_us - hist[0].t_us) / 1e6)
            d = (hist[-1].yaw - hist[0].yaw + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = d / dt_h

        # nearest agent ahead, in the ego frame at t0
        c, s = math.cos(-e.yaw), math.sin(-e.yaw)
        gap = float("inf")
        for track in obs.agent_history.values():
            b = track[-1]
            dx = c * (b.x - e.x) - s * (b.y - e.y)
            dy = s * (b.x - e.x) + c * (b.y - e.y)
            if dx > 0 and abs(dy) < self.CORRIDOR_HALF_W + b.width / 2:
                gap = min(gap, dx - b.length / 2 - obs.ego_length / 2)

        target = speed
        if gap < float("inf"):
            desired = self.MIN_GAP_M + self.DESIRED_GAP_S * speed
            if gap < desired:
                # close the gap deficit over the horizon, bounded by comfort
                deficit = desired - gap
                target = max(0.0, speed - min(self.MAX_DECEL * obs.horizon_s,
                                              deficit / max(0.5, obs.horizon_s)))

        out, x, y, th, v = [], 0.0, 0.0, 0.0, speed
        n = obs.n_steps
        for k in range(n):
            v += (target - speed) / n            # linear ramp to the target speed
            th += yaw_rate * obs.dt_s
            x += v * obs.dt_s * math.cos(th)
            y += v * obs.dt_s * math.sin(th)
            out.append((x, y))
        return out


class ReplayHuman(Policy):
    """ORACLE. Replays the recorded future — deliberately bypasses the Observation.

    This is the harness's calibration reference and a self-test: it must score
    close to 1.0. It is not a baseline to compare models against, and the harness
    marks it `is_oracle` so it cannot be mistaken for one in results.
    """
    name = "replay_human"
    is_oracle = True
    needs_scenario = True       # the ONLY policy granted the full Scenario

    def plan(self, obs, scenario):
        return scenario.future_egoframe(obs.t0_us, obs.n_steps, obs.dt_s)


BUILTIN = {p.name: p for p in (Stationary, ConstantVelocity, ConstantTurnRate,
                               ReactiveIDM, ReplayHuman)}
