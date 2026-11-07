#!/usr/bin/env python3
"""GeoLife 1.3 -> trips resampled at 10 s, local metres, with transport-mode labels.

Trip = maximal run of points of one user with gaps <= 600 s, inside the Beijing box,
duration >= 600 s.  Positions are linearly interpolated at a 10 s grid.  Output npz:
  trip_user[int], trip_mode[str], trip_len[int], trip_off[int] (into xy), xy[float32, N x 2]
Users are split 70/30 into train/test by a seeded permutation of user ids.
"""
import io, json, math, sys, zipfile, hashlib, random
import numpy as np

ZIP = sys.argv[1]; OUT = sys.argv[2]
LAT0, LON0 = 39.95, 116.35
BOX = (39.75, 40.10, 116.10, 116.65)
KX = 111320.0 * math.cos(math.radians(LAT0)); KY = 110540.0
DT = 10.0; GAP = 600.0; MIN_DUR = 600.0

z = zipfile.ZipFile(ZIP)
names = z.namelist()
plts = {}
labels = {}
for n in names:
    p = n.split('/')
    if n.endswith('.plt'):
        u = int(p[-3]); plts.setdefault(u, []).append(n)
    elif n.endswith('labels.txt'):
        labels[int(p[-2])] = n


def days_to_s(d):
    return (d - 25569.0) * 86400.0  # Excel/Delphi day number -> unix seconds


def parse_labels(n):
    out = []
    import datetime as dt
    for line in z.read(n).decode().splitlines()[1:]:
        a = line.strip().split('\t')
        if len(a) < 3: continue
        f = lambda s: dt.datetime.strptime(s, '%Y/%m/%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
        out.append((f(a[0]), f(a[1]), a[2].strip()))
    return out


trips = []
stats = {'users': 0, 'points_in_box': 0, 'trips': 0}
for u in sorted(plts):
    pts = []
    for n in plts[u]:
        lines = z.read(n).decode(errors='ignore').splitlines()[6:]
        for ln in lines:
            a = ln.split(',')
            if len(a) < 7: continue
            lat, lon, d = float(a[0]), float(a[1]), float(a[4])
            if BOX[0] <= lat <= BOX[1] and BOX[2] <= lon <= BOX[3]:
                pts.append((days_to_s(d), (lon - LON0) * KX, (lat - LAT0) * KY))
    if not pts: continue
    stats['users'] += 1; stats['points_in_box'] += len(pts)
    pts.sort()
    lab = parse_labels(labels[u]) if u in labels else []
    seg = [pts[0]]

    def flush(seg):
        if len(seg) < 30 or seg[-1][0] - seg[0][0] < MIN_DUR: return
        t = np.array([p[0] for p in seg]); x = np.array([p[1] for p in seg]); y = np.array([p[2] for p in seg])
        t, idx = np.unique(t, return_index=True); x = x[idx]; y = y[idx]
        g = np.arange(t[0], t[-1], DT)
        xi = np.interp(g, t, x); yi = np.interp(g, t, y)
        mode = 'unknown'
        if lab:
            votes = {}
            for (a0, a1, m) in lab:
                ov = max(0.0, min(a1, t[-1]) - max(a0, t[0]))
                if ov > 0: votes[m] = votes.get(m, 0) + ov
            if votes:
                m, ov = max(votes.items(), key=lambda kv: kv[1])
                if ov >= 0.5 * (t[-1] - t[0]): mode = m
        sp = np.hypot(np.diff(xi), np.diff(yi)) / DT
        trips.append(dict(user=u, mode=mode, xy=np.stack([xi, yi], 1).astype(np.float32), t0=float(t[0]),
                          dur=float(t[-1] - t[0]), v_med=float(np.median(sp)) if len(sp) else 0.0))

    for p in pts[1:]:
        if p[0] - seg[-1][0] > GAP:
            flush(seg); seg = [p]
        else:
            seg.append(p)
    flush(seg)

users = sorted({t['user'] for t in trips})
rng = random.Random(20261103); perm = users[:]; rng.shuffle(perm)
test_users = set(perm[int(0.7 * len(perm)):])
off = np.cumsum([0] + [len(t['xy']) for t in trips])
np.savez_compressed(OUT,
                    trip_user=np.array([t['user'] for t in trips]), trip_mode=np.array([t['mode'] for t in trips]),
                    trip_len=np.array([len(t['xy']) for t in trips]), trip_off=off[:-1],
                    trip_test=np.array([t['user'] in test_users for t in trips]),
                    trip_vmed=np.array([t['v_med'] for t in trips]), trip_t0=np.array([t['t0'] for t in trips]),
                    xy=np.concatenate([t['xy'] for t in trips]))
stats['trips'] = len(trips)
stats['users_with_trips'] = len(users); stats['test_users'] = len(test_users)
durs = np.array([t['dur'] for t in trips]); vm = np.array([t['v_med'] for t in trips])
stats['trip_dur_s'] = {q: float(np.quantile(durs, q)) for q in (0.1, 0.5, 0.9)}
stats['trip_vmed_mps'] = {q: float(np.quantile(vm, q)) for q in (0.1, 0.5, 0.9)}
modes = {}
for t in trips: modes[t['mode']] = modes.get(t['mode'], 0) + 1
stats['modes'] = modes
stats['zip_sha256'] = hashlib.sha256(open(ZIP, 'rb').read()).hexdigest()
json.dump(stats, open(OUT + '.stats.json', 'w'), indent=1)
print(json.dumps(stats, indent=1))
