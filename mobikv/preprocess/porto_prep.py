#!/usr/bin/env python3
"""A7 / E6: Porto taxi trajectories (ECML/PKDD 2015, UCI 339) -> trips npz in the GeoLife format of geolife_prep.py.

Trip = one taxi ride (POLYLINE, one GPS point per 15 s), no missing data, every point inside the Porto box, duration
>= 600 s, no step faster than 40 m/s.  Positions in local metres, linearly interpolated at a 10 s grid.  A seeded
reservoir sample of 30,000 qualifying trips is kept.  Taxis are split 70/30 into train/test by a seeded permutation.
Usage: porto_prep.py <uci zip> <out.npz>
"""
import csv, hashlib, io, json, math, random, sys, zipfile
import numpy as np

ZIP, OUT = sys.argv[1], sys.argv[2]
LAT0, LON0 = 41.16, -8.62
BOX = (41.05, 41.30, -8.75, -8.45)
KX = 111320.0 * math.cos(math.radians(LAT0)); KY = 110540.0
DT = 10.0; STEP = 15.0; MIN_PTS = 41; VMAX = 40.0; K = 30000
csv.field_size_limit(1 << 30)


def open_train(z):
    for n in z.namelist():
        b = n.lower().rsplit('/', 1)[-1]
        if b.startswith('train') and b.endswith('.csv'):
            return io.TextIOWrapper(z.open(n), encoding='utf-8')
    for n in z.namelist():
        b = n.lower().rsplit('/', 1)[-1]
        if b.startswith('train') and b.endswith('.zip'):
            return open_train(zipfile.ZipFile(io.BytesIO(z.read(n))))
    raise SystemExit('no train csv in ' + str(z.namelist()))


rng = random.Random(20261103); res = []; seen = 0; stats = dict(rows=0, missing=0, short=0, outside=0, jump=0)
f = open_train(zipfile.ZipFile(ZIP))
for row in csv.DictReader(f):
    stats['rows'] += 1
    if row['MISSING_DATA'].strip().lower() == 'true': stats['missing'] += 1; continue
    pts = json.loads(row['POLYLINE'])
    if len(pts) < MIN_PTS: stats['short'] += 1; continue
    p = np.array(pts, float)
    if not (np.all((p[:, 1] >= BOX[0]) & (p[:, 1] <= BOX[1]) & (p[:, 0] >= BOX[2]) & (p[:, 0] <= BOX[3]))): stats['outside'] += 1; continue
    x = (p[:, 0] - LON0) * KX; y = (p[:, 1] - LAT0) * KY
    if np.max(np.hypot(np.diff(x), np.diff(y))) / STEP > VMAX: stats['jump'] += 1; continue
    seen += 1
    rec = (int(row['TAXI_ID']), int(row['TIMESTAMP']), x.astype(np.float32), y.astype(np.float32))
    if len(res) < K: res.append(rec)
    else:
        j = rng.randrange(seen)
        if j < K: res[j] = rec
stats['qualifying'] = seen
trips = []
for taxi, t0, x, y in res:
    t = np.arange(len(x)) * STEP; g = np.arange(0.0, t[-1], DT)
    xi = np.interp(g, t, x); yi = np.interp(g, t, y)
    sp = np.hypot(np.diff(xi), np.diff(yi)) / DT
    trips.append(dict(user=taxi, xy=np.stack([xi, yi], 1).astype(np.float32), t0=float(t0), dur=float(t[-1]), v_med=float(np.median(sp))))
trips.sort(key=lambda d: (d['user'], d['t0']))
users = sorted({d['user'] for d in trips}); perm = users[:]; random.Random(20261103).shuffle(perm)
test_users = set(perm[int(0.7 * len(perm)):])
off = np.cumsum([0] + [len(d['xy']) for d in trips])
np.savez_compressed(OUT, trip_user=np.array([d['user'] for d in trips]), trip_mode=np.array(['taxi'] * len(trips)),
                    trip_len=np.array([len(d['xy']) for d in trips]), trip_off=off[:-1],
                    trip_test=np.array([d['user'] in test_users for d in trips]),
                    trip_vmed=np.array([d['v_med'] for d in trips]), trip_t0=np.array([d['t0'] for d in trips]),
                    xy=np.concatenate([d['xy'] for d in trips]))
durs = np.array([d['dur'] for d in trips]); vm = np.array([d['v_med'] for d in trips])
stats.update(trips=len(trips), taxis=len(users), test_taxis=len(test_users),
             trip_dur_s={q: float(np.quantile(durs, q)) for q in (0.1, 0.5, 0.9)},
             trip_vmed_mps={q: float(np.quantile(vm, q)) for q in (0.1, 0.5, 0.9)},
             zip_sha256=hashlib.sha256(open(ZIP, 'rb').read()).hexdigest())
json.dump(stats, open(OUT + '.stats.json', 'w'), indent=1); print(json.dumps(stats, indent=1))
