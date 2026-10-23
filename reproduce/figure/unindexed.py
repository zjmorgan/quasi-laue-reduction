import os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
WORK_DIR = _os.environ.get("QLR_WORK", _os.path.join(_REPO, "work"))
FIGURE_DIR = _os.environ.get("QLR_FIGURES", ".")
import numpy as np
d = np.load("t4l_1816.npz"); kf = np.load("../peaks_1816_detcal.npz")["kf_ki_dir"]
UB, hkl, lam, idx, coords = d["UB"], d["hkl"], d["lam"], d["indexed"], d["coords"]
bank = coords[:, 0].astype(int)
u = kf / np.linalg.norm(kf, axis=1, keepdims=True)
s2 = np.linalg.norm(kf, axis=1)            # |kf^-ki^| = 2 sin(theta)
tt = np.rad2deg(2 * np.arcsin(s2 / 2))
# all hkl to d_min 1.2 A
r = np.arange(-60, 61); H = np.array(np.meshgrid(r, r, np.arange(-90, 91), indexing="ij")).reshape(3, -1).T
q = H @ UB.T; qn = np.linalg.norm(q, axis=1); keep = (qn > 0) & (qn < 1 / 1.2)
H, q, qn = H[keep], q[keep], qn[keep]
qu = q / qn[:, None]
best_ang, best_lam, best_d = [], [], []
for i in range(len(u)):
    c = qu @ u[i]
    ang = np.rad2deg(np.arccos(np.clip(c, -1, 1)))
    l = s2[i] / qn                           # wavelength needed
    ok = (l > 1.0) & (l < 20) & (ang < 2)
    if not ok.any(): best_ang.append(9); best_lam.append(np.nan); best_d.append(np.nan); continue
    # lowest-order (largest d) among those within 0.35, else nearest
    j = np.flatnonzero(ok)[np.argmin(ang[ok])]
    best_ang.append(ang[j]); best_lam.append(l[j]); best_d.append(1 / qn[j])
best_ang, best_lam, best_d = map(np.array, (best_ang, best_lam, best_d))
un = ~idx
print("unindexed", un.sum(), "| 2theta median unindexed %.1f indexed %.1f" % (np.median(tt[un]), np.median(tt[idx])))
inb = (best_lam >= 2) & (best_lam <= 10)
m = un & (best_ang < 0.35)
print("unindexed matched within 0.35 deg with band 1-20 A:", m.sum(), "of which outside 2-10 A:", (m & ~inb).sum())
print("  lambdas of those:", np.round(np.sort(best_lam[m & ~inb]), 2))
print("unindexed nearest-angle quantiles (any lambda 1-20):", np.round(np.percentile(best_ang[un], [10, 25, 50, 75, 90]), 2))
print("unindexed nearest-angle within band only:", end=" ")
# chance reference: shuffle directions randomly
rng = np.random.default_rng(0)
# per-camera: indexed residual vector (gamma,nu offsets) and fraction indexed
g = np.rad2deg(np.arctan2(u[:, 0], u[:, 2]))
qi = (hkl[idx] @ UB.T); qi /= np.linalg.norm(qi, axis=1, keepdims=True)
# predicted kf direction from predicted s direction: kf = ki + lam*s ; s_hat known, |s|=2sin(th) -> use reflection of beam
def kf_from_s(sh):
    ki = np.array([0, 0, 1.0]); return ki - 2 * (sh @ ki)[:, None] * sh   # mirror
print()
ko = kf_from_s(u); kp = kf_from_s(qi)
dg = np.rad2deg(np.arctan2(kp[:, 0], kp[:, 2]) - np.arctan2(ko[idx, 0], ko[idx, 2]))
dn = np.rad2deg(np.arcsin(kp[:, 1]) - np.arcsin(ko[idx, 1]))
print("bank  n  indexed  med2theta  mean d_gamma  mean d_nu  (pred-obs, deg; +-sem)")
for b in np.unique(bank):
    on = bank == b; oi = bank[idx] == b
    if on.sum() < 4: continue
    sg = dg[oi].std() / np.sqrt(max(oi.sum(), 1)); sn = dn[oi].std() / np.sqrt(max(oi.sum(), 1))
    print(f"{b:4d} {on.sum():3d} {idx[on].mean():6.2f}  {np.median(tt[on]):7.1f}   {dg[oi].mean():+6.3f}+-{sg:.3f}  {dn[oi].mean():+6.3f}+-{sn:.3f}")
