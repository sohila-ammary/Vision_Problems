
import cv2
import numpy as np
import argparse
import sys
from typing import Optional

# ── geometry helpers ─────────────────────────────────────────────────────────

def to_homogeneous(pt):
    """2-D pixel point → 3-vector."""
    return np.array([pt[0], pt[1], 1.0])


def line_through(p1, p2):
    """Homogeneous line l = p1 × p2."""
    return np.cross(to_homogeneous(p1), to_homogeneous(p2))


def intersect_lines(l1, l2):
    """Intersection of two homogeneous lines."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-10:
        return None            # lines are parallel (point at infinity)
    return (p[0] / p[2], p[1] / p[2])


def compute_vvp(vertical_segments):
    """
    Compute the Vertical Vanishing Point from a list of line segments
    each represented as ((x1,y1), (x2,y2)).
    Uses least-squares intersection of homogeneous lines.
    """
    lines = [line_through(s[0], s[1]) for s in vertical_segments]
    # Build system: each line l gives constraint l·v = 0
    A = np.array([[l[0], l[1], l[2]] for l in lines])
    _, _, Vt = np.linalg.svd(A)
    v = Vt[-1]
    if abs(v[2]) < 1e-10:
        return None
    return (v[0] / v[2], v[1] / v[2])


def signed_distance(a, b):
    """Signed 1-D distance along the y-axis (for collinear vertical points)."""
    return b[1] - a[1]


def cross_ratio_vertical(t_img, b_img, vvp, fvp=None):
    """
    Compute the cross-ratio (t, b; vvp, fvp) for a vertical segment.

    When fvp (floor vanishing point) is unavailable we use the simplified
    form from Criminisi 2000 §4 which only needs the VVP and the two
    endpoints of the segment, assuming the camera's principal point is
    on the horizon (a reasonable approximation for most photos):

        ratio = (vvp_y - b_y) / (vvp_y - t_y)

    This is the ratio in which VVP divides the segment [t, b] — it is
    projectively invariant for any pair of parallel verticals.
    """
    vvp_y = vvp[1]
    t_y   = t_img[1]
    b_y   = b_img[1]
    denom = vvp_y - t_y
    if abs(denom) < 1e-6:
        return None
    return (vvp_y - b_y) / denom


def estimate_height(ref_base, ref_top, tgt_base, tgt_top, vvp, ref_height_cm):
    """
    Estimate target height using the cross-ratio method.
    ref_top/ref_base/tgt_top/tgt_base are image (pixel) coordinates.
    vvp is the vertical vanishing point in image coordinates.
    """
    cr_ref = cross_ratio_vertical(ref_top, ref_base, vvp)
    cr_tgt = cross_ratio_vertical(tgt_top, tgt_base, vvp)
    if cr_ref is None or cr_tgt is None or abs(cr_ref) < 1e-6:
        return None
    return ref_height_cm * (cr_tgt / cr_ref)


# ── application state ────────────────────────────────────────────────────────

state = {
    "phase":       1,          # 1=verticals, 2=reference, 3=target
    "clicks":      [],         # pending clicks
    "verticals":   [],         # committed vertical segments [(p1,p2), ...]
    "vvp":         None,       # computed vanishing point
    "ref":         None,       # (base, top) of reference object
    "tgt":         None,       # (base, top) of target object
    "ref_h":       None,       # real-world height of reference (cm)
    "result_cm":   None,
}

COLORS = {
    "vert":   (50,  140, 220),   # blue  — vertical reference lines
    "vvp":    (30,  80,  200),   # dark blue — VVP dot
    "ref":    (40,  180,  60),   # green — reference object
    "tgt":    (200,  60, 200),   # purple — target object
    "result": (20,  200, 230),   # yellow — result text
    "pending":(0,   230, 230),   # cyan  — pending clicks
    "white":  (255, 255, 255),
    "black":  (0,     0,   0),
}

PHASE_LABELS = {
    1: "Phase 1: Click top+bottom of vertical lines, [V] to commit | [D] when done",
    2: "Phase 2: Click BASE then TOP of reference object, [R] to commit",
    3: "Phase 3: Click BASE then TOP of target object,   [E] to compute",
}

WIN = "Vanishing Point Height Estimator"


# ── drawing ──────────────────────────────────────────────────────────────────

def draw_segment(img, p1, p2, color, thickness=2, label=""):
    cv2.line(img, p1, p2, color, thickness)
    for p in (p1, p2):
        cv2.circle(img, p, 5, color, -1)
        cv2.circle(img, p, 5, COLORS["white"], 1)
    if label:
        mid = ((p1[0]+p2[0])//2 + 8, (p1[1]+p2[1])//2)
        cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2, cv2.LINE_AA)


def draw_vvp(img, vvp):
    if vvp is None:
        return
    cx, cy = int(vvp[0]), int(vvp[1])
    cv2.drawMarker(img, (cx, cy), COLORS["vvp"],
                   cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.putText(img, "VVP", (cx+8, cy-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["vvp"], 2, cv2.LINE_AA)


def draw_vvp_rays(img, vvp, segments):
    """Draw dashed lines from VVP to the endpoints of vertical segments."""
    if vvp is None:
        return
    vpt = (int(vvp[0]), int(vvp[1]))
    for seg in segments:
        for pt in seg:
            cv2.line(img, vpt, pt, COLORS["vvp"], 1, cv2.LINE_AA)


def draw_hud(img):
    h, w = img.shape[:2]
    phase = state["phase"]
    bar_h = 26
    cv2.rectangle(img, (0, h - bar_h - 4), (w, h), (30, 30, 30), -1)
    hint = PHASE_LABELS.get(phase, "Press [Q] to quit")
    cv2.putText(img, hint, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # result overlay
    if state["result_cm"] is not None:
        txt = f"Estimated height: {state['result_cm']:.1f} cm"
        sz, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = (w - sz[0]) // 2
        cv2.rectangle(img, (x-10, 8), (x + sz[0]+10, 14 + sz[1]), (0,0,0), -1)
        cv2.putText(img, txt, (x, 8 + sz[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["result"], 2, cv2.LINE_AA)


def render(base_img):
    img = base_img.copy()

    # vertical reference lines
    for i, seg in enumerate(state["verticals"]):
        draw_segment(img, seg[0], seg[1], COLORS["vert"], 2, f"V{i+1}")

    # VVP
    draw_vvp_rays(img, state["vvp"], state["verticals"])
    draw_vvp(img, state["vvp"])

    # reference object
    if state["ref"]:
        draw_segment(img, state["ref"][0], state["ref"][1],
                     COLORS["ref"], 3, f"Ref {state['ref_h']:.0f}cm")

    # target object
    if state["tgt"]:
        label = (f"Est {state['result_cm']:.1f}cm"
                 if state["result_cm"] else "Target")
        draw_segment(img, state["tgt"][0], state["tgt"][1],
                     COLORS["tgt"], 3, label)

    # pending clicks
    for pt in state["clicks"]:
        cv2.circle(img, pt, 7, COLORS["pending"], -1)

    draw_hud(img)
    return img


# ── interaction ──────────────────────────────────────────────────────────────

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["clicks"].append((x, y))
        n = len(state["clicks"])
        phase = state["phase"]
        if phase == 1:
            if n == 1:
                print(f"  [V{len(state['verticals'])+1}] top clicked — now click bottom, then [V]")
        elif phase == 2:
            if n == 1:
                print("  Reference BASE recorded — click TOP, then [R]")
        elif phase == 3:
            if n == 1:
                print("  Target BASE recorded — click TOP, then [E]")


def try_commit_vertical():
    if len(state["clicks"]) < 2:
        print("  Need 2 clicks (top + bottom) first."); return
    seg = (state["clicks"][0], state["clicks"][1])
    state["verticals"].append(seg)
    state["clicks"].clear()
    # recompute VVP
    if len(state["verticals"]) >= 2:
        vvp = compute_vvp(state["verticals"])
        state["vvp"] = vvp
        if vvp:
            print(f"  VVP updated: ({vvp[0]:.1f}, {vvp[1]:.1f})  "
                  f"[{len(state['verticals'])} lines]")
    else:
        print(f"  Vertical {len(state['verticals'])} committed. Add at least one more.")


def try_commit_reference():
    if len(state["clicks"]) < 2:
        print("  Need 2 clicks first."); return
    base, top = state["clicks"][0], state["clicks"][1]
    # ensure base is below top in image coords (larger y = lower in image)
    if top[1] > base[1]:
        base, top = top, base
    state["ref"] = (base, top)
    state["clicks"].clear()
    state["phase"] = 3
    print(f"  Reference set. Now mark the target object.")
    maybe_compute()


def try_commit_target():
    if len(state["clicks"]) < 2:
        print("  Need 2 clicks first."); return
    base, top = state["clicks"][0], state["clicks"][1]
    if top[1] > base[1]:
        base, top = top, base
    state["tgt"] = (base, top)
    state["clicks"].clear()
    maybe_compute()


def maybe_compute():
    if not (state["vvp"] and state["ref"] and state["tgt"] and state["ref_h"]):
        return
    h = estimate_height(
        state["ref"][0], state["ref"][1],
        state["tgt"][0], state["tgt"][1],
        state["vvp"],
        state["ref_h"],
    )
    state["result_cm"] = h
    if h:
        ref_px  = abs(state["ref"][1][1]  - state["ref"][0][1])
        tgt_px  = abs(state["tgt"][1][1]  - state["tgt"][0][1])
        cr_ref  = cross_ratio_vertical(state["ref"][1], state["ref"][0], state["vvp"])
        cr_tgt  = cross_ratio_vertical(state["tgt"][1], state["tgt"][0], state["vvp"])
        print(f"\n{'='*55}")
        print(f"  Reference  : {ref_px} px  =  {state['ref_h']:.1f} cm")
        print(f"  Target     : {tgt_px} px  (raw pixel ratio {tgt_px/ref_px:.3f})")
        print(f"  CR ref     : {cr_ref:.4f}")
        print(f"  CR target  : {cr_tgt:.4f}")
        print(f"  CR ratio   : {cr_tgt/cr_ref:.4f}")
        print(f"  >>> Estimated height: {h:.2f} cm <<<")
        print(f"{'='*55}\n")
    else:
        print("  Could not compute — check VVP and markings.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Height estimation via vanishing point & cross-ratio")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--ref-height", type=float, default=None,
                        help="Real height of reference object in cm")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: cannot open '{args.image}'"); sys.exit(1)

    # Resize large images
    max_dim = 960
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    state["ref_h"] = args.ref_height or float(
        input("Enter real height of reference object (cm): "))

    print(__doc__)
    print(f"Reference object height: {state['ref_h']} cm\n")
    print("Phase 1: Click TOP then BOTTOM of vertical lines in the scene.")
    print("         (Use door frames, wall edges, window sides, etc.)")
    print("         At least 2 lines needed. Press [V] after each pair.\n")

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        frame = render(img)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            state.update(phase=1, clicks=[], verticals=[], vvp=None,
                         ref=None, tgt=None, result_cm=None)
            print("Cleared — start over.")

        elif key == ord("v"):
            if state["phase"] == 1:
                try_commit_vertical()

        elif key == ord("d"):
            if state["phase"] == 1:
                if len(state["verticals"]) < 2:
                    print("  Need at least 2 vertical lines first.")
                elif state["vvp"] is None:
                    print("  Could not compute VVP — check lines are not parallel.")
                else:
                    state["phase"] = 2
                    state["clicks"].clear()
                    print("\nPhase 2: Click BASE then TOP of the reference object, [R] to commit.")

        elif key == ord("r"):
            if state["phase"] == 2:
                try_commit_reference()

        elif key == ord("e"):
            if state["phase"] == 3:
                try_commit_target()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()