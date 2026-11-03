"""Case 2 of the MIPINN paper: orthotropic conduction in a 3D slab.

Paper section 5.2. A physical problem, not a manufactured one: there is no
volumetric source, and the only heat input is a uniform flux through one face.

    domain     x in [0, 0.20], y in [0, 0.05], z in [0, 0.25]   (metres)
               y is the 50 mm through-plane axis
    PDE        k_xx T_xx + k_yy T_yy + k_zz T_zz = 0
    k          k_xx = k_zz = 41 W/mK (in-plane, tied), k_yy = 7 W/mK

    y = 0.05   heated (rear) face     k_yy dT/dy = q''                 (Neumann)
    others     convective             -k_nn dT/dn = h (T - T_inf)      (Robin)

    q'' = 2000 W/m^2, h = 15 W/m^2K, T_inf = 300 K
    sensors    default: a cell-centred 5 x 5 grid on each of the five non-flux
               faces (125). The paper's layout, 25 on the front face y = 0
               (Figure 9), is ``sensor_grid(n_side_dict={"y0": 5})``.

The paper does not fix the orientation of the 250 x 200 footprint or which
y-face is heated; the choices above are ours.

SCALED FORMULATION. The solver's field is theta = (T - T_inf)/DT, with

    DT = Q / (h A_conv) = q'' L_x L_z / (h A_conv) = 100 W / 1.425 W/K = 70.18 K

A_conv being the five convective faces. In the basis's own unit coordinates
(xi, eta, zeta) = (x/L_x, y/L_y, z/L_z) (``TensorBasis._to_unit``) the problem
reads

    PDE          eps_x^2 r theta_xixi + theta_etaeta + eps_z^2 r theta_zetazeta = 0
    heated face  theta_eta = Pi = q'' L_y / (k_yy DT)
    Robin faces  d theta / dn + Bi_a theta = 0,   Bi_a = h L_a / k_a

with r = k_in / k_yy, eps_x = L_y/L_x = 0.25 and eps_z = L_y/L_z = 0.2; at the
true k, eps_x^2 r = 0.37, eps_z^2 r = 0.23, Pi = 0.20 and Bi = 0.073 / 0.107 /
0.091 (x / y / z). The rows are ASSEMBLED in flux units -- every Robin row as
(s k_a Phi_a + h Phi) beta = 0, the heated face as k_yy Phi_y beta = q''/DT,
a the face's normal axis and s = +1 on the high face, -1 on the low one --
which are the rows above times a constant each; the solver's row normalisation
removes that constant, so the two forms give the same solve. What the scaling
changes: the Robin rows are homogeneous, theta is O(1) instead of a 300 K
offset carrying a ~70 K signal, and DT does not depend on k, so every trial k
converts back to kelvin (T = T_inf + DT theta) by the same factor. Energy
balance forces the area mean of theta over the five convective faces to 1.

DIFFERENCES FROM CASE 3, which uses the same theta = (T - T_inf)/DT:
  * k stays in the PDE. Case 2 is orthotropic, so the PDE rows depend on the
    anisotropy r and are rebuilt per trial k; Case 3's isotropic k drops out of
    them. The in-plane coefficients here are O(1) (0.37, 0.23), not ~1e-3: no
    thin-plate stiffness.
  * The heat input covers the whole y = L_y face as a pure flux, with no
    convection there: no footprint, no h_F, a single box rather than patches.
    DT uses one uniform h, so its mean of theta is an area mean.
  * Every other face convects; Case 3's bottom face is adiabatic.
  * The unsatisfiable rows are the four heated-face edges (below), not a
    heater step, and Case 2 keeps the solver's unit-row normalisation, with
    which its forward accuracy was established, instead of Case 3's area
    weighting.
  * Two unknowns, inferred as log(k_in, k_yy); Case 3 has one, kappa = k/K_REF.

WHY THE CONSTRAINTS DEPEND ON k. Unlike Case 1.x, every boundary row carries a
conductivity, and on the Robin faces k cannot be divided out (the row becomes
Phi_a + (h/k) Phi, whose direction moves with k). So H_c -- and therefore its
null space -- must be rebuilt for each trial k. ``face_blocks`` precomputes the
k-independent pieces so that rebuilding is a weighted sum, not a re-evaluation
of the basis.

REFERENCE SOLUTION. The box is separable, so the exact solution is an
eigenfunction series (Ozisik-type): Robin eigenfunctions X_m(x), Z_n(z) in the
plane and a through-plane function Y_mn(y) meeting the Robin condition at y = 0,
with coefficients fixed by the flux at y = 0.05. Measured truncation error at
200 modes per axis: <= 1e-6 K on the heated face, ~1e-9 K in the interior, and
machine precision on the sensor face (high modes decay as exp(-lambda L_y)
before reaching it). It is used only to MEASURE the PIELM error; the PIELM
solve never sees it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from .. import config  # noqa: F401  (enables x64 before any array is made)

import jax.numpy as jnp

from ..solver import Constraints

LX, LY, LZ = 0.20, 0.05, 0.25
BOUNDS = ((0.0, LX), (0.0, LY), (0.0, LZ))

# (k_xx, k_yy, k_zz), axis order of the basis. The inverse problem has two
# unknowns, k_in = k_xx = k_zz and k_yy; ``expand_k`` maps between the two.
K_TRUE = (41.0, 7.0, 41.0)
Q_FLUX = 2000.0
H_CONV = 15.0
T_INF = 300.0

A_HEATED = LX * LZ
A_CONV = LX * LZ + 2.0 * LY * (LX + LZ)  # the five convective faces, 0.095 m^2
Q_HEATER = Q_FLUX * A_HEATED  # 100 W
DT = Q_HEATER / (H_CONV * A_CONV)  # 70.18 K; theta = (T - T_INF) / DT


def expand_k(k_in, k_yy):
    """(k_in, k_yy) -> (k_xx, k_yy, k_zz) with the in-plane pair tied."""
    return jnp.stack([jnp.asarray(k_in), jnp.asarray(k_yy), jnp.asarray(k_in)])


class Face(NamedTuple):
    name: str
    axis: int  # normal axis
    side: int  # 0 = low face, 1 = high face
    kind: str  # "flux" or "robin"

    @property
    def sign(self) -> float:
        """Outward normal along ``axis``: d/dn = sign * d/dx_axis."""
        return 1.0 if self.side == 1 else -1.0

FACES = (
    Face("x0", 0, 0, "robin"),
    Face("x1", 0, 1, "robin"),
    Face("y0", 1, 0, "robin"),  # front face
    Face("y1", 1, 1, "flux"),  # rear, heated face
    Face("z0", 2, 0, "robin"),
    Face("z1", 2, 1, "robin"),
)
FACE_NAMES = tuple(f.name for f in FACES)


class Edge(NamedTuple):
    """An edge, named and located by the two faces that meet along it.

    ``fixed`` holds the (axis, side) of both faces, lower axis first; the edge
    runs along the remaining axis. The four edges of the heated face carry the
    flux condition alone; every other edge carries the Robin conditions of both
    its faces, which agree there (both give T_ab = h^2 (T - T_inf) / (k_a k_b)).
    """

    name: str
    axis: int  # axis the edge runs along
    fixed: tuple[tuple[int, int], tuple[int, int]]  # (axis, side) of each face

    @property
    def faces(self) -> tuple[Face, Face]:
        return tuple(f for f in FACES if (f.axis, f.side) in self.fixed)

    @property
    def kind(self) -> str:
        """"flux" on the rim of the heated face, "robin" elsewhere."""
        return "flux" if any(f.kind == "flux" for f in self.faces) else "robin"


EDGES = (
    Edge("y0z0", 0, ((1, 0), (2, 0))),
    Edge("y0z1", 0, ((1, 0), (2, 1))),
    Edge("y1z0", 0, ((1, 1), (2, 0))),  # heated
    Edge("y1z1", 0, ((1, 1), (2, 1))),  # heated
    Edge("x0z0", 1, ((0, 0), (2, 0))),
    Edge("x0z1", 1, ((0, 0), (2, 1))),
    Edge("x1z0", 1, ((0, 1), (2, 0))),
    Edge("x1z1", 1, ((0, 1), (2, 1))),
    Edge("x0y0", 2, ((0, 0), (1, 0))),
    Edge("x0y1", 2, ((0, 0), (1, 1))),  # heated
    Edge("x1y0", 2, ((0, 1), (1, 0))),
    Edge("x1y1", 2, ((0, 1), (1, 1))),  # heated
)

# face name -> the four edges on its rim
EDGES_OF_FACE = {f.name: [e.name for e in EDGES if (f.axis, f.side) in e.fixed]
                 for f in FACES}


# --------------------------------------------------------------------------
# Point sampling
# --------------------------------------------------------------------------


def sample_interior(n: int, seed: int):
    """(n, 3) uniform random interior points, boundary excluded."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-6, 1.0 - 1e-6, size=(n, 3))
    return jnp.asarray(u * np.array([LX, LY, LZ]))


def face_dofs(degrees, face: Face) -> int:
    """Tangential polynomial dimension of a face: prod over the other axes."""
    out = 1
    for a, n in enumerate(degrees):
        if a != face.axis:
            out *= n + 1
    return out


def sample_faces(n_per_face: dict, seed: int) -> dict:
    """Random points on each open face (edges excluded, where n is undefined).

    Args:
        n_per_face: face name -> point count.

    Returns:
        face name -> (n, 3) points.
    """
    rng = np.random.default_rng(seed)
    L = np.array([LX, LY, LZ])
    out = {}
    for f in FACES:
        n = n_per_face[f.name]
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=(n, 3))
        u[:, f.axis] = float(f.side)
        out[f.name] = jnp.asarray(u * L)
    return out


def edge_dofs(degrees, edge: Edge) -> int:
    """Polynomial dimension along an edge."""
    return degrees[edge.axis] + 1


def sample_edges(n_per_edge: dict, seed: int) -> dict:
    """Random points on each open edge (corners excluded, as for the faces).

    Args:
        n_per_edge: edge name -> point count.

    Returns:
        edge name -> (n, 3) points.
    """
    rng = np.random.default_rng(seed)
    L = np.array([LX, LY, LZ])
    out = {}
    for e in EDGES:
        n = n_per_edge[e.name]
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=(n, 3))
        for a, s in e.fixed:
            u[:, a] = float(s)
        out[e.name] = jnp.asarray(u * L)
    return out


def sensor_layout(n_side: int = 5, n_side_dict: dict | None = None) -> dict:
    """Points per side of each face's sensor grid, as face name -> n.

    ``n_side`` applies to every non-flux face. ``n_side_dict`` (face name ->
    points per side) replaces it when given; it may name the flux face, and a
    face it omits carries no sensors.
    """
    if n_side_dict is None:
        layout = {f.name: n_side for f in FACES if f.kind != "flux"}
    else:
        unknown = set(n_side_dict) - set(FACE_NAMES)
        if unknown:
            raise ValueError(f"unknown face(s) {sorted(unknown)}; faces are {FACE_NAMES}")
        layout = {name: n_side_dict[name] for name in FACE_NAMES if name in n_side_dict}
    for name, n in layout.items():
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"points per side on {name} must be an integer >= 0, got {n!r}")
    layout = {name: int(n) for name, n in layout.items() if n > 0}
    if not layout:
        raise ValueError("the sensor layout has no sensors")
    return layout


def describe_layout(layout: dict) -> str:
    return "cell-centred grids: " + ", ".join(f"{name} {n}x{n}" for name, n in layout.items())


def sensor_grid(n_side: int = 5, n_side_dict: dict | None = None):
    """Sensors on a cell-centred n x n grid per face (``sensor_layout``).

    Cell centres keep every sensor off the edges, where the heated-face flux
    and the Robin data conflict and the forward error is largest.

    Returns (x, faces): (n, 3) positions in metres and the (n,) face name of
    each, in FACES order and, within a face, row-major in its two tangential
    axes (lower axis slowest). {"y0": 5} gives the paper's 25 front-face points
    in the order earlier runs used.
    """
    xs, names = [], []
    for name, n in sensor_layout(n_side, n_side_dict).items():
        f = FACES[FACE_NAMES.index(name)]
        a, b = (ax for ax in range(3) if ax != f.axis)
        t = (np.arange(n) + 0.5) / n
        ua, ub = np.meshgrid(BOUNDS[a][0] + t * (BOUNDS[a][1] - BOUNDS[a][0]),
                             BOUNDS[b][0] + t * (BOUNDS[b][1] - BOUNDS[b][0]), indexing="ij")
        P = np.empty((n * n, 3))
        P[:, a], P[:, b], P[:, f.axis] = ua.ravel(), ub.ravel(), BOUNDS[f.axis][f.side]
        xs.append(P)
        names += [name] * (n * n)
    return jnp.asarray(np.concatenate(xs)), np.array(names)


def faces_at(X, tol: float = 1e-6) -> list[tuple[str, ...]]:
    """Names of the faces each point lies on, |x_axis - bound| <= tol (metres).

    One name for a point inside a face, two on an edge, three at a corner,
    none in the interior.
    """
    X = np.asarray(X, dtype=float)
    on = {f.name: np.abs(X[:, f.axis] - BOUNDS[f.axis][f.side]) <= tol for f in FACES}
    return [tuple(name for name in FACE_NAMES if on[name][i]) for i in range(len(X))]


FEM_COLUMNS = ("x", "y", "z", "Temperature")


def load_fem_csv(path, tol: float = 1e-6):
    """COMSOL temperatures at surface points: (x, faces, T).

    The file is a plain CSV with the header x,y,z,Temperature, coordinates in
    metres and T in kelvin. Every point must lie inside exactly one face: a
    point in the interior, outside the box, or on an edge or corner raises.
    Rows are returned in the ``sensor_grid`` order (FACES order, then row-major
    in the tangential axes).
    """
    from pathlib import Path

    path = Path(path)
    with open(path) as fh:
        header = tuple(c.strip() for c in fh.readline().split(","))
    if header != FEM_COLUMNS:
        raise ValueError(f"{path.name}: header {header}, expected {FEM_COLUMNS}")
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    X, T = d[:, :3], d[:, 3]
    lo, hi = np.array([b[0] for b in BOUNDS]), np.array([b[1] for b in BOUNDS])
    if np.any(X < lo - tol) or np.any(X > hi + tol):
        raise ValueError(f"{path.name}: points outside the box {BOUNDS} -- coordinates "
                         f"must be in metres (max {X.max(axis=0)})")
    on = faces_at(X, tol)
    n_on = np.array([len(f) for f in on])
    if np.any(n_on != 1):
        raise ValueError(f"{path.name}: {int((n_on == 0).sum())} interior, "
                         f"{int((n_on == 2).sum())} edge and {int((n_on == 3).sum())} corner "
                         f"points; every point must lie inside exactly one face")
    faces = np.array([f[0] for f in on])
    rank = np.array([FACE_NAMES.index(f) for f in faces])
    order = []
    for i in np.unique(rank):
        idx = np.flatnonzero(rank == i)
        a, b = (ax for ax in range(3) if ax != FACES[i].axis)
        order.append(idx[np.lexsort((X[idx, b], X[idx, a]))])
    order = np.concatenate(order)
    return jnp.asarray(X[order]), faces[order], T[order]


def test_grid(n=(21, 11, 26)):
    """Structured closed 3D grid (boundaries included) for error measurement."""
    axes = [np.linspace(lo, hi, m) for (lo, hi), m in zip(BOUNDS, n)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return jnp.asarray(g), axes


def midplane_grid(n_x: int = 81, n_z: int = 101, y: float = LY / 2):
    """The paper's Figure 11 slice: the xz-plane at y = 25 mm."""
    xs, zs = np.linspace(0, LX, n_x), np.linspace(0, LZ, n_z)
    XX, ZZ = np.meshgrid(xs, zs, indexing="xy")
    pts = np.stack([XX.ravel(), np.full(XX.size, y), ZZ.ravel()], axis=1)
    return jnp.asarray(pts), xs, zs


AXIS_NAMES = ("x", "y", "z")


def slice_positions(axis: int):
    """The four evaluation planes normal to ``axis``: both faces, 1/3, 2/3."""
    L = BOUNDS[axis][1]
    return (0.0, L / 3.0, 2.0 * L / 3.0, L)


def slice_grid(axis: int, value: float, spacing: float = 2e-3):
    """Uniform tensor grid on one plane, edges and corners included.

    Returns (axes, pts): ``axes`` holds one 1D coordinate array per axis, with
    the sliced axis a single value, so the grid is exactly the tensor product
    the series reference evaluates in factorised form; ``pts`` is its (N, 3)
    flattening in C order.
    """
    axes = []
    for a, (lo, hi) in enumerate(BOUNDS):
        if a == axis:
            axes.append(np.array([value]))
        else:
            axes.append(np.linspace(lo, hi, int(round((hi - lo) / spacing)) + 1))
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return axes, jnp.asarray(g)


# --------------------------------------------------------------------------
# Constraint assembly
# --------------------------------------------------------------------------


class FaceBlock(NamedTuple):
    face: Face
    Phi: jnp.ndarray  # (n, N*) values
    dPhi: jnp.ndarray  # (n, N*) derivative along the normal axis


def face_blocks(basis, faces_pts: dict) -> list[FaceBlock]:
    """k-independent feature blocks per face, computed once."""
    return [
        FaceBlock(f, basis.features(faces_pts[f.name]),
                  basis.features_d1(faces_pts[f.name], f.axis))
        for f in FACES
    ]


class EdgeBlock(NamedTuple):
    edge: Edge
    Phi: jnp.ndarray  # (n, N*) values
    dPhi: tuple  # (n, N*) derivative along each face's normal, order of edge.faces


def edge_blocks(basis, edges_pts: dict) -> list[EdgeBlock]:
    """k-independent feature blocks per edge, computed once."""
    out = []
    for e in EDGES:
        P = edges_pts[e.name]
        out.append(EdgeBlock(e, basis.features(P),
                             tuple(basis.features_d1(P, f.axis) for f in e.faces)))
    return out


def _bc_rows(f: Face, k, Phi, dPhi):
    """Face f's condition at n points, for the field theta: (H, K) rows."""
    n = Phi.shape[0]
    if f.kind == "flux":
        return f.sign * k[f.axis] * dPhi, jnp.full(n, Q_FLUX / DT)
    return f.sign * k[f.axis] * dPhi + H_CONV * Phi, jnp.zeros(n)


def build_constraints(blocks: list[FaceBlock], k, edges: list[EdgeBlock] = ()) -> Constraints:
    """Assemble H_c(k) beta = K_c for theta from precomputed face (and edge) blocks.

        heated:  k_yy Phi_y beta = q'' / DT
        Robin:   (s k_a Phi_a + h Phi) beta = 0

    On an edge, a heated-rim edge gets the flux row only; any other edge gets
    one Robin row per adjacent face. Labels: "<kind>-<face>" for face rows,
    "edge-<kind>-<edge>" for edge rows.
    """
    k = jnp.asarray(k, dtype=float)
    H, K, labels = [], [], []
    for b in blocks:
        h_, k_ = _bc_rows(b.face, k, b.Phi, b.dPhi)
        H.append(h_)
        K.append(k_)
        labels += [f"{b.face.kind}-{b.face.name}"] * len(k_)
    for b in edges:
        e = b.edge
        for f, dPhi in zip(e.faces, b.dPhi):
            if e.kind == "flux" and f.kind != "flux":
                continue
            h_, k_ = _bc_rows(f, k, b.Phi, dPhi)
            H.append(h_)
            K.append(k_)
            labels += [f"edge-{e.kind}-{e.name}"] * len(k_)
    return Constraints(H=jnp.concatenate(H), K=jnp.concatenate(K), labels=tuple(labels))


# --------------------------------------------------------------------------
# The inverse problem, assembled once for every sampler
# --------------------------------------------------------------------------


class InverseProblem(NamedTuple):
    basis: object
    x_f: jnp.ndarray
    faces: dict
    blocks: list
    obs: object  # inverse.Observations
    obs_faces: np.ndarray  # (n_obs,) face name of each sensor
    T_series: np.ndarray  # (n_obs,) exact T at the sensors; obs.T_clean for "exact"
    series: object  # SeriesReference at the true k
    beta_of_k: object  # (k_in, k_yy) -> beta of theta = (T - T_INF)/DT
    forward_fn: object  # theta = log(k_in, k_yy) -> whitened residual
    settings: dict


def make_inverse_problem(
    *,
    degrees=(10, 6),
    rho_f: float = 3.0,
    rho_b: float = 2.0,
    noise_frac: float = 0.01,
    sigma_abs: float | None = None,
    seed_train: int = 0,
    seed_bc: int = 1,
    seed_obs: int = 2,
    n_modes: int = 200,
    ls_method: str = "svd",
    n_side: int = 5,
    n_side_dict: dict | None = None,
    source: str = "exact",
    fem_csv=None,
    ns_method: str = "svd",
    sensors: tuple | None = None,
) -> InverseProblem:
    """Case 2 as posed to EKS, the emulator and NUTS -- one definition for all.

    Unknowns: theta = log(k_in, k_yy), with k_xx = k_zz = k_in tied (the paper
    assumes identical in-plane conductivities and reports only these two).

    DATA never come from the PIELM solver, so the inversion does not see its
    own discretisation as truth. ``source``:
      "exact" -- the eigenfunction series at the ``sensor_grid(n_side,
                 n_side_dict)`` points, or at ``sensors`` = (x, faces): (n, 3)
                 positions in metres and the face name of each, which must
                 be the only face ``faces_at`` finds there;
      "fem"   -- COMSOL temperatures at the points of ``fem_csv``
                 (``load_fem_csv``), which fix the layout. ``sigma_abs`` is
                 required and ``n_side_dict`` is refused.
    Gaussian noise is added in both. The series values at the sensors are kept
    as ``T_series``, so the COMSOL error is measured, never assumed.

    NOISE is ``noise_frac`` of max|T| over the domain, the paper's "1%" read as
    in Case 1.2 (sigma = 0.01 x 380.1 K = 3.80 K), unless ``sigma_abs`` gives
    it directly in kelvin. At 3.8 K the 25 front-face sensors, which span only
    1.8 K, left k_in essentially prior-dominated.

    The solver's field is theta; the forward map converts it to kelvin
    (T = T_INF + DT theta), so the data, sigma and residuals are all in K.
    ``ls_method`` is the reduced least-squares solver (``solver.LS_METHODS``)
    and ``ns_method`` the null-space factorisation (``solver.NS_METHODS``).
    For any but "svd" / "svd", rank(H_c) is fixed once, at K_TRUE: it is
    structural, N* - (n_in - 1)^2 (n_y - 1), the same at every k, and any
    particle where it is not comes back as NaN.
    """
    import math

    from ..basis import TensorBasis
    from .. import inverse

    n_in, n_y = degrees
    deg3 = (n_in, n_y, n_in)
    basis = TensorBasis(degrees=deg3, bounds=BOUNDS)
    n_f = math.ceil(rho_f * basis.n_star)
    n_face = {f.name: math.ceil(rho_b * face_dofs(deg3, f)) for f in FACES}
    x_f = sample_interior(n_f, seed_train)
    faces = sample_faces(n_face, seed_bc)
    blocks = face_blocks(basis, faces)

    series = SeriesReference(K_TRUE, n_modes)
    T_max = float(series.T_tensor(*[np.linspace(lo, hi, 41) for lo, hi in BOUNDS]).max())
    fem = {}
    if source == "exact":
        if fem_csv is not None:
            raise ValueError('fem_csv is only read with source="fem"')
        if sensors is None:
            x_obs, obs_faces = sensor_grid(n_side, n_side_dict)
            layout_desc = describe_layout(sensor_layout(n_side, n_side_dict))
        else:
            if n_side_dict is not None:
                raise ValueError("give sensors or n_side_dict, not both")
            x_obs, obs_faces = jnp.asarray(sensors[0], dtype=float), np.asarray(sensors[1])
            if x_obs.ndim != 2 or x_obs.shape[1] != 3 or len(obs_faces) != len(x_obs):
                raise ValueError("sensors must be ((n, 3) positions, (n,) face names)")
            bad = [i for i, (on, f) in enumerate(zip(faces_at(x_obs, tol=1e-9), obs_faces))
                   if on != (f,)]
            if bad:
                raise ValueError(f"{len(bad)} sensors are not on their named face only, "
                                 f"first index {bad[0]}")
            names, counts = np.unique(obs_faces, return_counts=True)
            layout_desc = "explicit points: " + ", ".join(
                f"{a} {b}" for a, b in sorted(zip(names, counts),
                                              key=lambda t: FACE_NAMES.index(t[0])))
        T_series = series.T(np.asarray(x_obs))
        T_data = T_series
    elif source == "fem":
        if fem_csv is None or sigma_abs is None or n_side_dict is not None or sensors is not None:
            raise ValueError('source="fem" needs fem_csv and sigma_abs, and takes its '
                             'sensor layout from the file, not from n_side_dict or sensors')
        import hashlib
        from pathlib import Path

        x_obs, obs_faces, T_data = load_fem_csv(fem_csv)
        T_series = series.T(np.asarray(x_obs))
        names, counts = np.unique(obs_faces, return_counts=True)
        layout_desc = f"COMSOL points of {Path(fem_csv).name}: " + ", ".join(
            f"{a} {b}" for a, b in sorted(zip(names, counts), key=lambda t: FACE_NAMES.index(t[0])))
        d = T_data - T_series
        fem = dict(fem_csv=str(Path(fem_csv).resolve().relative_to(config.PROJECT_ROOT)),
                   fem_sha256=hashlib.sha256(Path(fem_csv).read_bytes()).hexdigest(),
                   fem_minus_series_mean=float(d.mean()),
                   fem_minus_series_maxabs=float(np.abs(d).max()),
                   fem_minus_series_l2_over_sigma=float(np.linalg.norm(d) / sigma_abs))
    else:
        raise ValueError(f'source must be "exact" or "fem", got {source!r}')
    # make_observations evaluates this once, at x_obs.
    obs = inverse.make_observations(
        basis, x_obs, lambda X: jnp.asarray(T_data),
        noise_frac, seed_obs, sigma_mode="max_abs", T_scale=T_max, sigma_abs=sigma_abs,
    )

    beta3 = inverse.make_beta_of_k_varying(
        basis, x_f, jnp.zeros(n_f), lambda k3: build_constraints(blocks, k3),
        ls_method=ls_method, ns_method=ns_method,
        k_rank=None if ls_method == ns_method == "svd" else K_TRUE,
    )

    def beta_of_k(k):
        k = jnp.asarray(k).reshape(-1)
        return beta3(expand_k(k[0], k[1]))

    forward_fn = inverse.make_forward_fn(beta_of_k, obs, log_space=True,
                                         t_offset=T_INF, t_scale=DT)

    settings = dict(
        degrees=list(deg3), n_star=basis.n_star, rho_f=rho_f, rho_b=rho_b,
        n_colloc=n_f, n_bc=sum(n_face.values()), n_obs=int(obs.x.shape[0]),
        obs_layout=layout_desc, obs_source=source, **fem,
        noise_frac=None if sigma_abs is not None else noise_frac,
        sigma_mode="absolute" if sigma_abs is not None else "max_abs", sigma=obs.sigma,
        seed_train=seed_train, seed_bc=seed_bc, seed_obs=seed_obs,
        series_modes=n_modes, field="theta", dt=DT, ls_method=ls_method, ns_method=ns_method,
    )
    return InverseProblem(basis=basis, x_f=x_f, faces=faces, blocks=blocks, obs=obs,
                          obs_faces=obs_faces, T_series=np.asarray(T_series),
                          series=series, beta_of_k=beta_of_k, forward_fn=forward_fn,
                          settings=settings)


# --------------------------------------------------------------------------
# Energy balance: heat in through y = L_y against convective loss elsewhere
# --------------------------------------------------------------------------


def _face_quadrature(face: Face, n_q: int):
    """Gauss-Legendre nodes and weights on one face."""
    g, w = np.polynomial.legendre.leggauss(n_q)
    others = [a for a in range(3) if a != face.axis]
    nodes, weights = [], []
    for a in others:
        lo, hi = BOUNDS[a]
        nodes.append(lo + (g + 1) * (hi - lo) / 2)
        weights.append(w * (hi - lo) / 2)
    U, V = np.meshgrid(nodes[0], nodes[1], indexing="ij")
    W = np.outer(weights[0], weights[1]).ravel()
    P = np.zeros((U.size, 3))
    P[:, others[0]], P[:, others[1]] = U.ravel(), V.ravel()
    P[:, face.axis] = BOUNDS[face.axis][face.side]
    return P, W


def energy_balance(T_fn, dT_fn, k, n_q: int = 40):
    """(Q_in, Q_out) in watts for a field in KELVIN given as callables on (N, 3)
    points (for a theta solve: T = T_INF + DT theta, dT = DT dtheta).

    Q_in integrates k_yy dT/dy over the heated face (enforced only at points,
    so this tests it BETWEEN them); Q_out integrates h (T - T_inf) over the
    five convective faces. Exact heat input is q'' L_x L_z = 100 W.
    """
    q_in, q_out = 0.0, 0.0
    for f in FACES:
        P, W = _face_quadrature(f, n_q)
        if f.kind == "flux":
            q_in += float(np.sum(W * k[f.axis] * np.asarray(dT_fn(P, f.axis))))
        else:
            q_out += float(np.sum(W * H_CONV * (np.asarray(T_fn(P)) - T_INF)))
    return q_in, q_out


# --------------------------------------------------------------------------
# Reference: eigenfunction series
# --------------------------------------------------------------------------


def _robin_eigs(L: float, Hb: float, M: int):
    """First M roots of (mu^2 - H^2) sin(mu L) - 2 mu H cos(mu L) = 0.

    Equivalent to tan(mu L) = 2 mu H / (mu^2 - H^2) without the poles; exactly
    one sign change in each ((m-1) pi / L, m pi / L).
    """
    f = lambda m: (m * m - Hb * Hb) * np.sin(m * L) - 2.0 * m * Hb * np.cos(m * L)
    return np.array([
        brentq(f, i * np.pi / L + 1e-9, (i + 1) * np.pi / L - 1e-9,
               xtol=1e-15, rtol=1e-15)
        for i in range(M)
    ])


class SeriesReference:
    """Exact Case 2 temperature as a truncated eigenfunction series.

        theta = T - T_inf = sum_mn A_mn X_m(x) Z_n(z) Y_mn(y)

        X_m = mu_m cos(mu_m x) + H_x sin(mu_m x),         H_x = h / k_xx
        Z_n = nu_n cos(nu_n z) + H_z sin(nu_n z),         H_z = h / k_zz
        Y_mn = cosh(lam y) + (h / (k_yy lam)) sinh(lam y),
               lam^2 = (k_xx mu^2 + k_zz nu^2) / k_yy     (Robin at y = 0)
        A_mn = q'' <X_m, 1> <Z_n, 1> /
               (|X_m|^2 |Z_n|^2 [k_yy lam sinh(lam L_y) + h cosh(lam L_y)])

    Y_mn / [...] is evaluated as a ratio of exponentials: lam L_y passes the
    float64 range of cosh (~710) beyond ~300 modes per axis, while the ratio
    itself stays below 1.
    """

    def __init__(self, k=K_TRUE, n_modes: int = 200):
        self.k = tuple(float(v) for v in k)
        kx, ky, kz = self.k
        self.Hx, self.Hz = H_CONV / kx, H_CONV / kz
        self.mu, cx = self._axis(LX, self.Hx, n_modes)
        self.nu, cz = self._axis(LZ, self.Hz, n_modes)
        self.lam = np.sqrt((kx * self.mu[:, None] ** 2 + kz * self.nu[None, :] ** 2) / ky)
        self.A = Q_FLUX * cx[:, None] * cz[None, :]

    @staticmethod
    def _axis(L, Hb, M):
        mu = _robin_eigs(L, Hb, M)
        mean = np.sin(mu * L) + (Hb / mu) * (1.0 - np.cos(mu * L))
        norm = 0.5 * ((mu**2 + Hb**2) * (L + Hb / (mu**2 + Hb**2)) + Hb)
        return mu, mean / norm

    @staticmethod
    def _eig(mu, Hb, s, d):
        s = s[:, None]
        if d == 0:
            return mu * np.cos(mu * s) + Hb * np.sin(mu * s)
        return mu * (-mu * np.sin(mu * s) + Hb * np.cos(mu * s))

    def _through(self, y, d):
        """Y_mn^(d)(y) / [k_yy lam sinh(lam L_y) + h cosh(lam L_y)]."""
        ky, lam = self.k[1], self.lam[None]
        y = y[:, None, None]
        ep, em = np.exp(lam * (y - LY)), np.exp(-lam * (y + LY))
        e2 = np.exp(-2.0 * lam * LY)
        den = ky * lam * (1.0 - e2) + H_CONV * (1.0 + e2)
        if d == 0:
            num = (ep + em) + (H_CONV / (ky * lam)) * (ep - em)
        else:
            num = lam * (ep - em) + (H_CONV / ky) * (ep + em)
        return num / den

    def _eval(self, X, dax=None, chunk: int = 64):
        X = np.asarray(X, dtype=float)
        d = [0, 0, 0] if dax is None else [int(dax == a) for a in range(3)]
        out = []
        for i in range(0, len(X), chunk):
            p = X[i:i + chunk]
            out.append(np.einsum(
                "pm,pn,pmn,mn->p",
                self._eig(self.mu, self.Hx, p[:, 0], d[0]),
                self._eig(self.nu, self.Hz, p[:, 2], d[2]),
                self._through(p[:, 1], d[1]),
                self.A,
            ))
        return np.concatenate(out) if out else np.zeros(0)

    def T(self, X):
        return self._eval(X) + T_INF

    def T_tensor(self, xs, ys, zs):
        """T on the tensor grid xs x ys x zs, shape (nx, ny, nz).

        Factorised: sum over m of X_m(x) A_mn Y_mn(y) is formed once per y and
        contracted with Z_n(z), so a grid costs O(ny M N + nx ny nz N) rather
        than O(nx ny nz M N) -- seconds instead of minutes for a 2 mm slice.
        """
        Xm = self._eig(self.mu, self.Hx, np.asarray(xs, float), 0)  # (nx, M)
        Zn = self._eig(self.nu, self.Hz, np.asarray(zs, float), 0)  # (nz, N)
        out = np.empty((len(xs), len(ys), len(zs)))
        for j, y in enumerate(np.asarray(ys, float)):
            G = self._through(np.array([y]), 0)[0] * self.A  # (M, N)
            out[:, j, :] = Xm @ G @ Zn.T
        return out + T_INF

    def dT(self, X, axis: int):
        return self._eval(X, dax=axis)
