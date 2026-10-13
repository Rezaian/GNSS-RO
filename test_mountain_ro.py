"""
Synthetic end-to-end test for the mountain-top pipeline (steps 5 -> 6 -> 7).

Truth: dry atmosphere, T = 288.15 - 6.5 K/km * h, hydrostatic P, N = 77.6 P/T,
spherical about the occultation's centre of curvature. Station at 3.9 km.

For each impact parameter a and branch, the ray is built exactly:
  receiver direction from Bouguer (n_r), total bending from the forward
  integrals, transmitter on the incoming asymptote at GPS radius,
  Doppler = v_t . (k_t - k0) / lambda.
The CSV is written in the step-4 format and run through the real
BendingAngleRetriever, AbelInversion and retrieve_atmospheric_profile.

Pass criteria (fixed before running):
  1. recovered partial bending vs truth: max |error| < 1e-6 rad
  2. refractivity:  max |N_ret - N_true| < 0.5 N-units
  3. pressure:      max |P_ret - P_true| < 0.5 hPa
"""
import os
import sys
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import tempfile
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ground_gnss_ro_pipeline as gp

LAT, LON, H_STA, AZ0 = 35.95, 51.42, 3900.0, 250.0
R_GPS = 26_560e3
F1, F2 = 1575.42e6, 1227.60e6
V_SAT = np.array([600.0, -2700.0, 1500.0])   # in-plane x, in-plane y, out-of-plane (m/s)


def truth_profile():
    """Dry hydrostatic atmosphere on a fine grid of height above the R_c sphere."""
    h = np.linspace(-500.0, 80_000.0, 160_001)
    T = np.maximum(288.15 - 6.5e-3 * h, 216.65)
    P = np.empty_like(h)
    i0 = np.argmin(np.abs(h - H_STA))
    P[i0] = 630.0                                        # station pressure (hPa), arbitrary
    m_d, R = 28.97e-3, 8.314462
    for i in range(i0 - 1, -1, -1):                      # same hypsometric rule as the pipeline
        g = gp.compute_gravity(0.5 * (h[i] + h[i + 1]), LAT)
        P[i] = P[i + 1] * np.exp(g * m_d * (h[i + 1] - h[i]) / (R * 0.5 * (T[i] + T[i + 1])))
    for i in range(i0 + 1, len(h)):
        g = gp.compute_gravity(0.5 * (h[i] + h[i - 1]), LAT)
        P[i] = P[i - 1] * np.exp(-g * m_d * (h[i] - h[i - 1]) / (R * 0.5 * (T[i] + T[i - 1])))
    N = 77.6 * P / T
    return h, T, P, N, float(P[i0])


def main(noise_hz=0.0, n_events=1, check_step7=True, label='', single_freq=False, allow_single=False, seed=7,
         below_only=False):
    """below_only: the satellite never rises above the apparent horizon -> alpha_P from the model."""
    gp.ALLOW_SINGLE_FREQ = allow_single
    gp.ALPHA_P_MODEL = 'fill'
    rng = np.random.default_rng(seed)
    st_xyz = gp.geodetic_to_ecef(LAT, LON, H_STA)
    c0, R_c = gp.curvature_centre(st_xyz, LAT, LON, H_STA, AZ0)
    h, T, P, N, P_sta = truth_profile()
    r = R_c + h
    n = 1.0 + N * 1e-6
    x = n * r
    lnn_of_x = interp1d(x, np.log(n))
    dlnn_dx = np.gradient(np.log(n), x)
    g_of_x = interp1d(x, dlnn_dx)
    r_r = R_c + H_STA
    n_r = float(np.interp(r_r, r, n))
    x_r = n_r * r_r
    x_top = x[-1]

    def integral(a, lo, hi, npts=4000):
        """Int_lo^hi g(x)/sqrt(x^2-a^2) dx via u = sqrt(x^2-a^2) (removes the singularity)."""
        u = np.linspace(np.sqrt(max(lo * lo - a * a, 0.0)), np.sqrt(hi * hi - a * a), npts)
        xx = np.sqrt(a * a + u * u)
        return gp._trapz(g_of_x(xx) / xx, u)

    # 2D plane -> ECEF basis: y = geodetic up at station, x = away from the satellite
    up = gp.geodetic_up(LAT, LON)
    la, lo = np.radians(LAT), np.radians(LON)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    toward = np.cos(np.radians(AZ0)) * north + np.sin(np.radians(AZ0)) * east
    ex, ey = -toward, up
    ez = np.cross(ex, ey)

    def rot(v, ang):
        c, s = np.cos(ang), np.sin(ang)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    # impact parameters: tangent heights 0.2 .. 3.85 km
    r_tp = R_c + np.linspace(200.0, 3850.0, 200)
    a_vals = np.interp(r_tp, r, n) * r_tp

    rows, truth, tbranch = [], [], {}
    t0 = pd.Timestamp('2026-07-01T12:00:00')
    k_ep = 0
    seq = [(+1, a) for a in a_vals] + [(-1, a) for a in a_vals[::-1]]   # setting: P first, then N
    if below_only:
        seq = [(-1, a) for a in a_vals[::-1]]
    for ev in range(n_events):
        k_ep = ev * 3600                                              # events 1 h apart
        for branch, a in seq:
            up_part = -a * integral(a, x_r, x_top)
            alpha = up_part + (-2 * a * integral(a, a, x_r) if branch == -1 else 0.0)
            psi = np.arcsin(a / (n_r * r_r))
            k_r = np.array([np.sin(psi), np.cos(psi) * (1 if branch == -1 else -1)])
            k_t = rot(k_r, alpha)
            nh = rot(k_t, np.pi / 2)
            tx2 = a * nh - np.sqrt(R_GPS ** 2 - a ** 2) * k_t
            rx2 = np.array([0.0, r_r])
            k0 = (rx2 - tx2) / np.linalg.norm(rx2 - tx2)
            v2 = V_SAT[:2]
            tx3 = c0 + tx2[0] * ex + tx2[1] * ey
            v3 = V_SAT[0] * ex + V_SAT[1] * ey + V_SAT[2] * ez
            el, _ = gp.elevation_azimuth(tx3, st_xyz, LAT, LON)
            utc = (t0 + pd.Timedelta(seconds=k_ep)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            tbranch[utc] = (branch, a)
            for sig, f in ((('L1C/A', F1),) if single_freq else (('L1C/A', F1), ('L2CL', F2))):
                dop = float(v2 @ (k_t - k0)) * f / gp.SPEED_OF_LIGHT + rng.normal(0, noise_hz) * (noise_hz > 0)
                # SP3 gives the position at RECEIVE time; the signal left tau earlier (at tx3)
                rx3 = tx3 + v3 * (np.linalg.norm(tx3 - st_xyz) / gp.SPEED_OF_LIGHT)
                rows.append({'gnssId': 'GPS', 'svId': 7, 'sigID': sig, 'timestamp': float(k_ep),
                             'utc': utc, 'accurate_elevation': el, 'atmos_doppler': dop,
                             'atmos_dopp_poli': dop,
                             'interp_x': rx3[0], 'interp_y': rx3[1], 'interp_z': rx3[2],
                             'interp_vel_x': v3[0], 'interp_vel_y': v3[1], 'interp_vel_z': v3[2]})
            if branch == -1 and ev == 0:
                truth.append((a, alpha + a * integral(a, x_r, x_top)))   # true partial bending
            k_ep += 1

    tmp = tempfile.mkdtemp(prefix='mro_')
    print(f"\n=== {label or 'clean'}: noise {noise_hz} Hz, {n_events} event(s) ===")
    step4 = os.path.join(tmp, 'step4.csv')
    pd.DataFrame(rows).to_csv(step4, index=False)
    els = pd.DataFrame(rows)['accurate_elevation']
    print(f"synthetic: {k_ep} epochs, elevation {els.max():+.2f} .. {els.min():+.2f} deg, "
          f"N_station={(n_r-1)*1e6:.1f}, R_c={R_c/1e3:.3f} km")

    station = gp.StationConfig(latitude=LAT, longitude=LON, altitude=H_STA,
                               surface_N=(n_r - 1) * 1e6, surface_pressure_hPa=P_sta)

    # ---- step 5
    res5 = gp.retrieve_bending_angles(step4, station, gp.PipelineConfig(), os.path.join(tmp, 'bending'))
    print('step5:', res5.message, res5.metadata.get('skipped'))
    ids = list(res5.data['sat_id']) if not res5.data.empty else []
    if single_freq and not allow_single:
        ok = not ids and not gp.evaluate_ro_status(pd.read_csv(step4)).get('GPS_7', False)
        print(f"  single-freq data with the mode OFF: not an RO candidate, no profile -> {'PASS' if ok else 'FAIL'}")
        return ok
    modes = list(res5.data['freq_mode']) if ids else []
    want = 'single' if single_freq else 'dual'
    ok_events = len(ids) == n_events and all(m == want for m in modes)
    print(f"  freq_mode: {modes} (expected {want})")
    print(f"  events found: {ids} -> {'PASS' if ok_events else 'FAIL'}")
    ta, tp = np.array(truth).T
    o = np.argsort(ta)
    ok1 = ok2 = ok_events
    for ev_id in ids:
        bcsv = os.path.join(tmp, 'bending', f'{ev_id}_bending.csv')
        b = pd.read_csv(bcsv)
        err_alpha = b['bending_angle_rad'].values - np.interp(b['impact_parameter_m'].values, ta[o], tp[o])
        # branch labels vs truth, excluding epochs within 3 sigma_a of the apparent horizon
        ep = pd.read_csv(os.path.join(tmp, 'bending', f'{ev_id}_epochs.csv'))
        tb = np.array([tbranch[u][0] for u in ep.utc]); ta_ep = np.array([tbranch[u][1] for u in ep.utc])
        sig_a = 1.6e3 * max(noise_hz, 1e-3)                       # ~1.6 km of a per Hz
        far = np.abs(ta_ep - x_r) > 3 * sig_a
        acc = float((ep.branch.values[far] == tb[far]).mean()) if far.any() else 1.0
        rcsv = os.path.join(tmp, f'refr_{ev_id}.csv')
        res6 = gp.retrieve_refractivity(bcsv, rcsv)
        rf = res6.data
        N_true = np.interp(rf['height_km'].values * 1e3, h, N)
        err_N = rf['refractivity_N'].values - N_true
        if below_only:
            src = set(b['alpha_P_src']) if 'alpha_P_src' in b.columns else set()
            a_ok = src == {'model'}
            n_ok = np.sqrt(np.mean(err_N ** 2)) < 2.0
            crit = "alpha_P all from the model, rms dN<2"
            nstat = (f"rms dN {np.sqrt(np.mean(err_N ** 2)):.2f}, max |dN| {np.max(np.abs(err_N)):.2f}, "
                     f"alpha_P source {sorted(src)}")
        elif noise_hz == 0:
            # exact geometry applies to measured-alpha_P levels; model-filled edge levels
            # (v4.7) are tested by the below-horizon-only scenario
            if 'alpha_P_src' in b.columns:
                err_alpha = err_alpha[b['alpha_P_src'].values == 'measured']
            a_ok = np.nanmax(np.abs(err_alpha)) < 1e-6
            n_ok = np.max(np.abs(err_N)) < 0.5
            crit = "max|dalpha|<1e-6, max|dN|<0.5"
            nstat = f"max |dN| {np.max(np.abs(err_N)):.3f}"
        else:
            a_ok = acc >= 0.95
            n_ok = np.sqrt(np.mean(err_N ** 2)) < 5.0
            crit = "branch acc>=95%, rms dN<5"
            nstat = f"rms dN {np.sqrt(np.mean(err_N ** 2)):.2f}, max |dN| {np.max(np.abs(err_N)):.2f}"
        ok1 &= a_ok; ok2 &= n_ok
        print(f"  {ev_id}: branch accuracy {acc*100:.1f}% (away from horizon), max |dalpha| {np.nanmax(np.abs(err_alpha)):.2e} rad, "
              f"{nstat}, levels {len(rf)}, {rf['height_km'].min():.2f}-{rf['height_km'].max():.2f} km "
              f"-> {'PASS' if (a_ok and n_ok) else 'FAIL'} ({crit})")
    rcsv = os.path.join(tmp, f'refr_{ids[0]}.csv') if ids else None

    # ---- tangent-point geometry vs an independent numerical ray trace (clean run only)
    if ids and noise_hz == 0 and not single_freq and not below_only:
        rf0 = pd.read_csv(rcsv)
        lv = rf0[(rf0['height_km'] < (H_STA - 300.0) / 1e3)].iloc[::max(1, len(rf0) // 6)]
        dn_dr = np.gradient(n, r)
        def trace(a):
            # RK4 on d(pos)/ds = p/n, d(p)/ds = grad n, from the receiver back along the ray
            psi = np.arcsin(a / (n_r * r_r))
            pos = np.array([0.0, r_r]); t = np.array([-np.sin(psi), -np.cos(psi)])
            p_ = float(np.interp(r_r, r, n)) * t
            def f(st):
                q, pp = st[:2], st[2:]
                rr = np.hypot(*q); nn = np.interp(rr, r, n); g = np.interp(rr, r, dn_dr) * q / rr
                return np.concatenate([pp / nn, g])
            stt = np.concatenate([pos, p_]); ds = 20.0; best = (np.inf, 0.0)
            for _ in range(40000):
                k1 = f(stt); k2 = f(stt + ds / 2 * k1); k3 = f(stt + ds / 2 * k2); k4 = f(stt + ds * k3)
                stt = stt + ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                rr = np.hypot(*stt[:2])
                if rr < best[0]:
                    best = (rr, np.arctan2(-stt[0], stt[1]))
                elif rr > best[0] + 5.0:
                    break
            return best[1]
        rel = []
        for _, row in lv.iterrows():
            th_trace = trace(row['impact_parameter_m'])
            th_pipe = row['tp_distance_km'] * 1e3 / R_c
            rel.append(abs(th_pipe - th_trace) / th_trace)
        # lat/lon consistency: haversine distance and bearing from the station
        la0, lo0 = np.radians(LAT), np.radians(LON)
        la1, lo1 = np.radians(rf0['tp_lat'].values), np.radians(rf0['tp_lon'].values)
        hav = 2 * np.arcsin(np.sqrt(np.sin((la1 - la0) / 2) ** 2 + np.cos(la0) * np.cos(la1) * np.sin((lo1 - lo0) / 2) ** 2)) * R_c / 1e3
        brg = np.degrees(np.arctan2(np.sin(lo1 - lo0) * np.cos(la1),
                                    np.cos(la0) * np.sin(la1) - np.sin(la0) * np.cos(la1) * np.cos(lo1 - lo0))) % 360
        m = rf0['tp_distance_km'] > 1.0
        d_err = np.max(np.abs(hav[m] - rf0['tp_distance_km'].values[m]))
        b_err = np.max(np.abs(brg[m] - rf0['tp_azimuth_deg'].values[m]))
        ok_geo = max(rel) < 0.01 and d_err < 0.01 and b_err < 0.01
        print(f"  tangent points: {rf0['tp_distance_km'].min():.0f}-{rf0['tp_distance_km'].max():.0f} km; "
              f"vs RK4 ray trace max rel. error {max(rel)*100:.3f}% (straight-line would be "
              f"{np.mean([abs(np.arccos(rw['impact_parameter_m']/x_r) - rw['tp_distance_km']*1e3/R_c)/(rw['tp_distance_km']*1e3/R_c) for _, rw in lv.iterrows()])*100:.0f}% short); "
              f"lat/lon distance err {d_err*1e3:.1f} m, bearing err {b_err:.4f} deg -> {'PASS' if ok_geo else 'FAIL'}")
        ok1 = ok1 and ok_geo
    if not check_step7 or not ids:
        return ok1 and ok2

    # ---- step 7 with a synthetic ERA5 file (exact T at pressure levels, q = 0)
    import xarray as xr
    plev = np.array([1000, 925, 850, 800, 700, 650, 600, 500], float)
    h_lev = np.interp(-plev, -P, h)
    T_lev = np.interp(h_lev, h, T)
    lats, lons = np.arange(LAT - 3.0, LAT + 3.01, 0.5), np.arange(LON - 3.0, LON + 3.01, 0.5)
    times = pd.to_datetime(['2026-07-01T06:00', '2026-07-01T12:00'])
    shp = (len(times), len(plev), len(lats), len(lons))
    ds = xr.Dataset(
        {'t': (('valid_time', 'pressure_level', 'latitude', 'longitude'),
               np.broadcast_to(T_lev[None, :, None, None], shp).copy()),
         'q': (('valid_time', 'pressure_level', 'latitude', 'longitude'), np.zeros(shp)),
         'z': (('valid_time', 'pressure_level', 'latitude', 'longitude'),
               np.broadcast_to((9.80665 * h_lev)[None, :, None, None], shp).copy())},
        coords={'valid_time': times, 'pressure_level': plev, 'latitude': lats, 'longitude': lons})
    era5 = os.path.join(tmp, 'era5.nc')
    ds.to_netcdf(era5)

    res7 = gp.retrieve_atmospheric_profile(rcsv, era5, LAT, LON, os.path.join(tmp, 'atm.csv'),
                                           when=res5.data['utc_mid'].iloc[0], station=station)
    print('step7:', res7.message)
    co = pd.read_csv(os.path.join(tmp, 'atm.csv'))['era5_colocated']
    print(f"  ERA5 sampled at the tangent point for {int(co.sum())}/{len(co)} levels")
    ok3_colo = bool(co.sum() >= len(co) - 1)            # the station row itself has distance 0
    at = res7.data
    P_true = np.interp(at['height_km'].values * 1e3, h, P)
    err_P = at['pressure_hPa'].values - P_true
    ok3 = np.max(np.abs(err_P)) < 0.5 and ok3_colo
    print(f"  P: {at['pressure_hPa'].min():.1f} .. {at['pressure_hPa'].max():.1f} hPa, "
          f"max |dP| = {np.max(np.abs(err_P)):.3f} hPa, max Pw = {at['water_vapor_hPa'].max():.3f} hPa"
          f"  -> {'PASS' if ok3 else 'FAIL'}")

    print('\nALL PASS' if (ok1 and ok2 and ok3) else '\nFAILED')
    return bool(ok1 and ok2 and ok3)


def noisy_multi_seed(seeds=range(1, 9)):
    """
    One noise draw swings the profile error between ~2.5 and ~8 N for ANY branch
    method, so a single seed is not a test. Criterion: mean RMS dN < 4 and every
    seed < 6 N over 8 seeds (v4.3 labelling: mean 3.75, worst 7.6).
    """
    import io, contextlib, re
    vals = []
    for sd in seeds:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(noise_hz=0.05, check_step7=False, label=f'noisy seed {sd}', seed=sd)
        m = re.search(r'rms dN ([\d.]+)', buf.getvalue())
        vals.append(float(m.group(1)) if m else np.inf)
    ok = np.mean(vals) < 4.0 and max(vals) < 6.0
    print(f"\n=== noise 0.05 Hz over {len(vals)} seeds: rms dN {np.round(vals, 2).tolist()}, "
          f"mean {np.mean(vals):.2f}, worst {max(vals):.2f} -> {'PASS' if ok else 'FAIL'} (mean<4, worst<6)")
    return ok


if __name__ == '__main__':
    results = {
        'clean, 1 event':           main(),
        'clean, 2 events':          main(n_events=2, check_step7=False, label='two events'),
        'noise 0.05 Hz, 8 seeds':   noisy_multi_seed(),
        'single-freq, mode on':     main(single_freq=True, allow_single=True, check_step7=False, label='single frequency, mode ON'),
        'single-freq, mode off':    main(single_freq=True, allow_single=False, check_step7=False, label='single frequency, mode OFF'),
        'below horizon only (model alpha_P)': main(below_only=True, check_step7=False, label='below the horizon only'),
    }
    print("\n" + "\n".join(f"{'PASS' if v else 'FAIL'}  {k}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)
