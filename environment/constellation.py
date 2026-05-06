"""
Walker Delta constellation model.
Computes satellite positions (ECI / ECEF) and relative velocities.

Reference: Section III-B of the paper.
"""

import math
import numpy as np
from typing import Dict, Tuple, List

import config as cfg

R_EARTH = 6_371_000.0   # [m]
MU      = 3.986004418e14 # [m^3/s^2]
OMEGA_E = 7.2921159e-5   # Earth rotation rate [rad/s]
C_LIGHT = 299_792_458.0  # [m/s]


class Constellation:
    """
    Walker Delta constellation.

    Parameters
    ----------
    n_planes : int
        Number of orbital planes.
    n_sats : int
        Satellites per plane.
    altitude_m : float
        Orbital altitude [m].
    inclination_deg : float
        Orbital inclination [degrees].
    """

    def __init__(
        self,
        n_planes: int        = cfg.TRAIN_N_PLANES,
        n_sats: int          = cfg.TRAIN_N_SATS_PER_PLANE,
        altitude_m: float    = cfg.ALTITUDE_M,
        inclination_deg: float = cfg.INCLINATION_DEG,
    ):
        self.n_planes    = n_planes
        self.n_sats      = n_sats
        self.altitude_m  = altitude_m
        self.inc         = math.radians(inclination_deg)
        self.n_total     = n_planes * n_sats

        self.semi_major  = R_EARTH + altitude_m
        self.n_mean      = math.sqrt(MU / self.semi_major ** 3)  # mean motion [rad/s]

        # Satellite ID mapping: sat_id = plane * n_sats + slot
        self.sat_ids: List[int] = list(range(self.n_total))

    # ── Position computation ────────────────────────────────────────────────────

    def _eci_pos(self, plane: int, slot: int, t: float) -> np.ndarray:
        """ECI Cartesian position of one satellite at time t [s]."""
        raan  = 2 * math.pi * plane / self.n_planes
        phase = 2 * math.pi * slot  / self.n_sats
        M     = self.n_mean * t + phase

        x_orb = self.semi_major * math.cos(M)
        y_orb = self.semi_major * math.sin(M)

        cosO, sinO = math.cos(raan), math.sin(raan)
        cosi, sini = math.cos(self.inc), math.sin(self.inc)

        x =  x_orb * cosO - y_orb * cosi * sinO
        y =  x_orb * sinO + y_orb * cosi * cosO
        z =  y_orb * sini
        return np.array([x, y, z], dtype=np.float64)

    def _eci_to_ecef(self, eci: np.ndarray, t: float) -> np.ndarray:
        """Rotate ECI vector to ECEF."""
        theta  = OMEGA_E * t
        ct, st = math.cos(theta), math.sin(theta)
        return np.array([
             ct * eci[0] + st * eci[1],
            -st * eci[0] + ct * eci[1],
             eci[2],
        ], dtype=np.float64)

    def get_positions_eci(self, t: float) -> np.ndarray:
        """
        Returns ECI positions for all satellites (vectorized).

        Returns
        -------
        pos : ndarray, shape (n_total, 3)
        """
        pp = np.arange(self.n_planes)   # (P,)
        ss = np.arange(self.n_sats)     # (S,)

        raan  = (2 * math.pi / self.n_planes) * pp  # (P,)
        phase = (2 * math.pi / self.n_sats) * ss    # (S,)

        # Broadcast: M[p, s] = n_mean*t + phase[s]  → shape (P, S)
        M = self.n_mean * t + phase[np.newaxis, :]   # (1, S) broadcasts to (P, S)

        x_orb = self.semi_major * np.cos(M)   # (P, S)
        y_orb = self.semi_major * np.sin(M)   # (P, S)

        cosO = np.cos(raan).reshape(self.n_planes, 1)   # (P, 1)
        sinO = np.sin(raan).reshape(self.n_planes, 1)   # (P, 1)
        cosi = math.cos(self.inc)
        sini = math.sin(self.inc)

        x = x_orb * cosO - y_orb * cosi * sinO    # (P, S)
        y = x_orb * sinO + y_orb * cosi * cosO    # (P, S)
        z = np.broadcast_to(y_orb * sini, (self.n_planes, self.n_sats)).copy()  # (P, S)

        # Flatten to (n_total, 3)
        pos = np.empty((self.n_total, 3), dtype=np.float64)
        pos[:, 0] = x.ravel()
        pos[:, 1] = y.ravel()
        pos[:, 2] = z.ravel()
        return pos

    def get_velocities_eci(self, t: float, dt: float = 0.5) -> np.ndarray:
        """
        Numerical ECI velocities via finite difference.

        Returns
        -------
        vel : ndarray, shape (n_total, 3)
        """
        p1 = self.get_positions_eci(t)
        p2 = self.get_positions_eci(t + dt)
        return (p2 - p1) / dt

    # ── Link availability ───────────────────────────────────────────────────────

    def check_isl(
        self,
        pos_u: np.ndarray,
        pos_v: np.ndarray,
        max_dist: float = cfg.ISL_RANGE_M,
    ) -> Tuple[bool, float]:
        """
        Check ISL availability between two satellites.
        Implements Eq. (3) from the paper (LoS + range).

        Returns
        -------
        (available, distance)
        """
        d = float(np.linalg.norm(pos_u - pos_v))
        if d > max_dist:
            return False, d

        # LoS: line must not intersect the Earth
        nu = float(np.linalg.norm(pos_u))
        nv = float(np.linalg.norm(pos_v))
        if nu <= R_EARTH or nv <= R_EARTH:
            return False, d

        los_limit = (
            math.sqrt(max(nu ** 2 - R_EARTH ** 2, 0.0))
            + math.sqrt(max(nv ** 2 - R_EARTH ** 2, 0.0))
        )
        return d <= los_limit, d

    def elevation_angle(
        self,
        sat_pos_ecef: np.ndarray,
        gs_pos_ecef: np.ndarray,
    ) -> float:
        """
        Elevation angle of satellite seen from a ground station.
        Implements Eq. (4) from the paper.
        """
        v   = sat_pos_ecef - gs_pos_ecef
        dot = float(np.dot(sat_pos_ecef, v))
        ns  = float(np.linalg.norm(sat_pos_ecef))
        nv  = float(np.linalg.norm(v))
        if ns < 1e-9 or nv < 1e-9:
            return 0.0
        return math.degrees(math.asin(dot / (ns * nv)))

    # ── SNR / data-rate ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_snr(dist_m: float) -> float:
        """
        Free-space SNR using Eq. (5):
            γ = P_tx · G_tx · G_rx / (N0 · B · L_fs)
        Returns SNR in linear scale.
        """
        if dist_m <= 0:
            return 1e12
        # Free-space path loss
        fs_loss_db = (
            20 * math.log10(dist_m)
            + 20 * math.log10(cfg.CARRIER_FREQ_HZ)
            + 20 * math.log10(4 * math.pi / C_LIGHT)
        )
        snr_db = (
            10 * math.log10(cfg.TX_POWER_W)
            + cfg.TX_GAIN_DBI
            + cfg.RX_GAIN_DBI
            - cfg.NOISE_PSD_DBW_HZ
            - 10 * math.log10(cfg.BANDWIDTH_HZ)
            - fs_loss_db
        )
        return 10 ** (snr_db / 10.0)

    @staticmethod
    def compute_rate(dist_m: float, available: bool = True) -> float:
        """
        Shannon-Hartley data rate [bits/s]. Eq. (6).
        """
        if not available or dist_m <= 0:
            return 0.0
        snr = Constellation.compute_snr(dist_m)
        snr_min = 10 ** (cfg.SINR_THRESHOLD_DB / 10.0)
        if snr < snr_min:
            return 0.0
        return cfg.BANDWIDTH_HZ * math.log2(1.0 + snr)

    @staticmethod
    def compute_per(dist_m: float, rel_vel_ms: float = 0.0,
                    is_inter_plane: bool = False) -> float:
        """
        Per-hop Packet Error Rate (PER) model combining three physical effects:

          1. Distance component — longer ISL → higher free-space path loss
               per_dist = PER_DIST_MAX * (dist_m / ISL_RANGE_M)^2

          2. Velocity component — high relative velocity → Doppler shift /
             pointing instability → increased link degradation
               per_vel  = PER_VEL_MAX  * (rel_vel_ms / V_MAX_MS)^2

          3. Inter-plane penalty — inter-plane ISLs suffer additional
             APT (acquisition, pointing, tracking) errors due to relative
             orbital plane motion, beam handover, and geometry changes.
             This effect is independent of instantaneous velocity and
             represents the structural instability of cross-plane links.
        """
        d_ratio = min(dist_m / cfg.ISL_RANGE_M, 1.0)
        v_ratio = min(abs(rel_vel_ms) / cfg.V_MAX_MS, 1.0)
        per = cfg.PER_DIST_MAX * (d_ratio ** 2) + cfg.PER_VEL_MAX * (v_ratio ** 2)
        # Inter-plane links have additional APT-related packet errors
        if is_inter_plane:
            per += cfg.PER_INTERPLANE
        return float(min(per, 0.30))   # hard cap at 30 %

    # ── Topology snapshot ───────────────────────────────────────────────────────

    def build_topology(
        self,
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the ISL topology at time t (vectorized 4-connected Walker Delta).

        Returns
        -------
        positions : ndarray (n_total, 3)   ECI positions
        velocities: ndarray (n_total, 3)   ECI velocities
        adj_avail : ndarray (n_total, n_total) bool  — link available
        adj_dist  : ndarray (n_total, n_total) float — link distance [m]
        """
        pos = self.get_positions_eci(t)
        vel = self.get_velocities_eci(t)

        n  = self.n_total
        P  = self.n_planes
        S  = self.n_sats
        adj_avail = np.zeros((n, n), dtype=bool)
        adj_dist  = np.zeros((n, n), dtype=np.float64)

        # Satellite IDs as (P, S) grid
        ids = np.arange(n).reshape(P, S)   # ids[p, s] = p*S + s

        # ── Intra-orbit links: (p, s) ↔ (p, s+1) ──────────────────────────────
        u_ids = ids                                  # (P, S)
        v_ids = np.roll(ids, -1, axis=1)             # (P, S) shifted by +1 slot
        u_flat = u_ids.ravel()
        v_flat = v_ids.ravel()
        mask   = u_flat < v_flat                     # avoid double-counting
        self._check_and_set_isl(pos, adj_avail, adj_dist, u_flat[mask], v_flat[mask])

        # ── Inter-orbit links: (p, s) ↔ (p+1, s) ─────────────────────────────
        u_ids2 = ids                                 # (P, S)
        v_ids2 = np.roll(ids, -1, axis=0)            # (P, S) shifted by +1 plane
        u_flat2 = u_ids2.ravel()
        v_flat2 = v_ids2.ravel()
        mask2   = u_flat2 < v_flat2
        self._check_and_set_isl(pos, adj_avail, adj_dist, u_flat2[mask2], v_flat2[mask2])

        return pos, vel, adj_avail, adj_dist

    def _check_and_set_isl(
        self,
        pos: np.ndarray,
        adj_avail: np.ndarray,
        adj_dist: np.ndarray,
        u_arr: np.ndarray,
        v_arr: np.ndarray,
        max_dist: float = cfg.ISL_RANGE_M,
    ):
        """Vectorized ISL check for a set of (u, v) pairs."""
        pu = pos[u_arr]   # (K, 3)
        pv = pos[v_arr]   # (K, 3)
        diff = pu - pv
        d = np.sqrt((diff * diff).sum(axis=1))   # (K,)

        # Range constraint
        in_range = d <= max_dist

        # LoS: line must not intersect Earth
        nu = np.sqrt((pu * pu).sum(axis=1))
        nv = np.sqrt((pv * pv).sum(axis=1))
        los_limit = (
            np.sqrt(np.maximum(nu ** 2 - R_EARTH ** 2, 0.0))
            + np.sqrt(np.maximum(nv ** 2 - R_EARTH ** 2, 0.0))
        )
        los_ok = (nu > R_EARTH) & (nv > R_EARTH) & (d <= los_limit)

        ok = in_range & los_ok
        u_ok = u_arr[ok]
        v_ok = v_arr[ok]
        d_ok = d[ok]

        adj_avail[u_ok, v_ok] = True
        adj_avail[v_ok, u_ok] = True
        adj_dist[u_ok, v_ok]  = d_ok
        adj_dist[v_ok, u_ok]  = d_ok

    @staticmethod
    def compute_rate_matrix(
        dist_m: np.ndarray,
        avail: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized Shannon-Hartley data rate matrix [bits/s].
        Only computes rates for available links.
        """
        n = dist_m.shape[0]
        rates = np.zeros((n, n), dtype=np.float64)
        if not avail.any():
            return rates

        safe_d = np.where(avail & (dist_m > 0), dist_m, 1.0)
        fs_loss_db = (
            20 * np.log10(safe_d)
            + 20 * math.log10(cfg.CARRIER_FREQ_HZ)
            + 20 * math.log10(4 * math.pi / C_LIGHT)
        )
        snr_db = (
            10 * math.log10(cfg.TX_POWER_W)
            + cfg.TX_GAIN_DBI
            + cfg.RX_GAIN_DBI
            - cfg.NOISE_PSD_DBW_HZ
            - 10 * math.log10(cfg.BANDWIDTH_HZ)
            - fs_loss_db
        )
        snr = 10 ** (snr_db / 10.0)
        snr_min = 10 ** (cfg.SINR_THRESHOLD_DB / 10.0)
        valid = avail & (dist_m > 0) & (snr >= snr_min)
        rates = np.where(valid, cfg.BANDWIDTH_HZ * np.log2(1.0 + snr), 0.0)
        rates[~np.isfinite(rates)] = 0.0
        return rates

    def gs_ecef(self, lat_deg: float, lon_deg: float) -> np.ndarray:
        """ECEF position of a ground station."""
        la, lo = math.radians(lat_deg), math.radians(lon_deg)
        return np.array([
            R_EARTH * math.cos(la) * math.cos(lo),
            R_EARTH * math.cos(la) * math.sin(lo),
            R_EARTH * math.sin(la),
        ], dtype=np.float64)

    def find_gateway(
        self,
        sat_positions_eci: np.ndarray,
        t: float,
        gs_ecef: np.ndarray,
        elev_mask_deg: float = cfg.ELEV_MASK_DEG,
    ) -> int:
        """Return the index of the best gateway satellite (nearest above mask)."""
        theta = OMEGA_E * t
        ct, st = math.cos(theta), math.sin(theta)
        best_idx, best_d = -1, float("inf")
        for i in range(self.n_total):
            eci = sat_positions_eci[i]
            ecef = np.array([
                 ct * eci[0] + st * eci[1],
                -st * eci[0] + ct * eci[1],
                 eci[2],
            ])
            elev = self.elevation_angle(ecef, gs_ecef)
            if elev < elev_mask_deg:
                continue
            d = float(np.linalg.norm(ecef - gs_ecef))
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx
