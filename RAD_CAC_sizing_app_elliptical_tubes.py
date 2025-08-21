# app.py — Elliptical Radiator Sizer (Zukauskas + Colburn-j), coolant parallel across rows
import math, io
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import streamlit as st

# ---- Optional thermophys libs ----
try:
    from CoolProp.CoolProp import PropsSI
    from CoolProp.HumidAirProp import HAPropsSI
except Exception:
    PropsSI = None
    HAPropsSI = None

# ---- PDF ----
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# -------------------- Helpers --------------------
def ellipse_perimeter(a: float, b: float) -> float:
    h = ((a-b)**2)/((a+b)**2)
    return math.pi*(a+b)*(1 + (3*h)/(10+math.sqrt(4-3*h)))

def ellipse_area(a: float, b: float) -> float:
    return math.pi*a*b

def air_properties(T_C: float, RH_frac: float, P_Pa: float = 101325.0) -> Dict[str, float]:
    T_K = T_C + 273.15
    if HAPropsSI is not None:
        try:
            rho = HAPropsSI('Rho','T',T_K,'P',P_Pa,'RH',RH_frac)
            cp  = HAPropsSI('C','T',T_K,'P',P_Pa,'RH',RH_frac)
            k   = HAPropsSI('K','T',T_K,'P',P_Pa,'RH',RH_frac)
            mu  = HAPropsSI('MU','T',T_K,'P',P_Pa,'RH',RH_frac)
        except Exception:
            rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.95e-5
    else:
        rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.95e-5
    Pr = cp*mu/max(k,1e-12)
    return {"rho":rho,"cp":cp,"k":k,"mu":mu,"Pr":Pr}

def coolant_properties(fluid: str, T_C: float) -> Dict[str, float]:
    T_K = T_C + 273.15
    if PropsSI is not None:
        try:
            rho = PropsSI('D','T',T_K,'P',101325,fluid)
            cp  = PropsSI('C','T',T_K,'P',101325,fluid)
            k   = PropsSI('L','T',T_K,'P',101325,fluid)
            mu  = PropsSI('V','T',T_K,'P',101325,fluid)
        except Exception:
            rho, cp, k, mu = 1000.0, 4180.0, 0.6, 1.0e-3
    else:
        rho, cp, k, mu = 1000.0, 4180.0, 0.6, 1.0e-3
    Pr = cp*mu/max(k,1e-12)
    return {"rho":rho,"cp":cp,"k":k,"mu":mu,"Pr":Pr}



def charge_air_properties(T_C: float, P_Pa: float) -> Dict[str, float]:
    """Dry (charge) air properties at (T, P). Falls back if CoolProp unavailable."""
    T_K = T_C + 273.15
    if PropsSI is not None:
        try:
            rho = PropsSI('D','T',T_K,'P',P_Pa,'Air')
            cp  = PropsSI('C','T',T_K,'P',P_Pa,'Air')
            k   = PropsSI('L','T',T_K,'P',P_Pa,'Air')
            mu  = PropsSI('V','T',T_K,'P',P_Pa,'Air')
        except Exception:
            rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.95e-5
    else:
        rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.95e-5
    Pr = cp*mu/max(k,1e-12)
    return {"rho":rho, "cp":cp, "k":k, "mu":mu, "Pr":Pr}

def gnielinski_h_i(Re: float, Pr: float, k: float, D_h: float) -> float:
    if Re < 2300:
        Nu = 3.66
    else:
        f = (0.79*math.log(Re) - 1.64)**-2
        Nu = (f/8.0)*(Re-1000.0)*Pr/(1+12.7*math.sqrt(f/8.0)*(Pr**(2/3)-1))
        Nu = max(Nu, 3.66)
    return Nu*k/max(D_h,1e-12)

def zukauskas_h_o(Re: float, Pr: float, k: float, d_char: float, heating: bool=True) -> float:
    if Re < 1e3:
        C, m = 0.9, 0.4
    elif Re < 2e5:
        C, m = 0.51, 0.5
    else:
        C, m = 0.26, 0.6
    n = 0.36 if heating else 0.37
    Nu = C*(Re**m)*(Pr**n)
    return Nu*k/max(d_char,1e-12)

def friction_factor_channel(Re: float) -> float:
    if Re <= 0: return 0.0
    if Re < 2300: return 64.0/max(Re,1e-9)
    return 0.3164/(Re**0.25)

def friction_factor_tube(Re: float, rel_rough: float = 1e-5) -> float:
    if Re <= 0: return 0.0
    if Re < 2300: return 64.0/max(Re,1e-9)
    inv_sqrt_f = -1.8*math.log10((rel_rough/3.7)**1.11 + 6.9/max(Re,1e-9))
    return 1.0/(inv_sqrt_f**2)

def row_correction_C2(N_rows: int, staggered: bool):
    ROW_CORR_INLINE = {1:0.64,2:0.80,3:0.87,4:0.90,5:0.92,6:0.94,7:0.96,8:0.98,9:0.99}
    ROW_CORR_STAG   = {1:0.68,2:0.75,3:0.83,4:0.89,5:0.92,6:0.95,7:0.97,8:0.98,9:0.99}
    if N_rows >= 20: return 1.0
    table = ROW_CORR_STAG if staggered else ROW_CORR_INLINE
    if N_rows in table: return table[N_rows]
    return 1.0 if N_rows >= 10 else table.get(max([k for k in table if k<=N_rows], default=9), 1.0)

def crossflow_effectiveness_mixed_unmixed(NTU: float, C_r: float) -> float:
    if C_r <= 0: return 0.0
    term = math.exp(-NTU)
    eps = (1.0/C_r)*(1.0 - math.exp(-C_r*(1.0 - term)))
    return max(0.0, min(0.999999, eps))

def tube_bank_vmax(V_ref, ST, SL, D, staggered: bool):
    if ST <= D: return V_ref
    if not staggered:
        return V_ref * ST/(ST - D)
    SD = math.sqrt(SL*SL + (ST*0.5)**2)
    if SD >= 0.5*(ST + D):
        return V_ref * ST/(ST - D)
    else:
        denom = max(2.0*(SD - D), 1e-9)
        return V_ref * ST/denom

# ----- Colburn j & fin friction helpers -----
def j_colburn(Re: float, Pr: float, Dh: float, fin_pitch: float,
              is_louvered: bool, louver_pitch: float, louver_angle_deg: float,
              Cj: float, mj: float, aj: float, bj: float, cj: float) -> float:
    Re = max(Re, 1.0)
    r1 = max(fin_pitch/max(Dh,1e-9), 1e-3)
    if is_louvered:
        r2 = max(louver_pitch/max(fin_pitch,1e-9), 1e-3)
        th = math.radians(max(louver_angle_deg, 1e-3))
        r3 = max(math.sin(th), 1e-3)
    else:
        r2, r3 = 1.0, 1.0
    j = Cj * (Re ** (-mj)) * (r1 ** aj) * (r2 ** bj) * (r3 ** cj)
    return max(j, 1e-6)

def f_fin_corr(Re: float, Cf: float, mf: float) -> float:
    return max(Cf * (max(Re,1.0) ** (-mf)), 1e-4)

# -------------------- UI --------------------
st.set_page_config(page_title="Elliptical Radiator Sizer", layout="wide")
st.title("🚗 Elliptical-Tube Radiator Sizing — NTU/ε & ΔP (Zukauskas + Colburn-j)")

with st.sidebar:
    st.header("Global Inputs")
    Q_target_kW = st.number_input("Heat load to dissipate (kW)", min_value=0.0, value=354.0, step=1.0)
    T_cool_in_C = st.number_input("Coolant inlet (°C)", min_value=-50.0, value=95.0, step=0.5)
    T_air_in_C  = st.number_input("Air inlet (°C)", min_value=-50.0, value=50.0, step=0.5)
    RH_air_pct  = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    RH_air = RH_air_pct/100.0
    P_atm_Pa = st.number_input("Ambient pressure (Pa)", min_value=80000, max_value=110000, value=101325, step=500)

c1,c2,c3 = st.columns(3)
with c1:
    st.subheader("Core Geometry (mm)")
    core_height_mm = st.number_input("Core height (mm)", min_value=10.0, value=1270.0, step=5.0)
    core_width_mm  = st.number_input("Core width (mm)", min_value=10.0, value=1254.0, step=5.0)
    core_depth_mm  = st.number_input("Core depth (mm)", min_value=10.0, value=79.0, step=1.0)
    n_rows = st.number_input("Number of rows", min_value=1, value=5, step=1)

with c2:
    st.subheader("Elliptical Tube (mm)")
    tube_od_width_mm = st.number_input("Tube OD – width (height dir) (mm)", min_value=0.5, value=2.4, step=0.1)
    tube_od_depth_mm = st.number_input("Tube OD – depth (airflow dir) (mm)", min_value=0.5, value=12.2, step=0.1)
    tube_thk_mm      = st.number_input("Tube wall thickness (mm)", min_value=0.05, value=0.3, step=0.05)
    tube_pitch_mm    = st.number_input("Tube-to-tube pitch (width dir) (mm)", min_value=1.0, value=13.5, step=0.1)
    row_pitch_mm     = st.number_input("Row pitch (front-to-back) (mm)", min_value=2.0, value=15.5, step=0.5)
    arrangement      = st.radio("Tube arrangement", ["Inline","Staggered"], index=0, horizontal=True)

with c3:
    st.subheader("Fins & Materials")
    fin_style    = st.radio("Fin style", ["Plate fin","Corrugated fin"], index=0, horizontal=True)
    fin_louvered = st.radio("Louvered?", ["Non-louvered","Louvered"], index=1, horizontal=True)
    fpi          = st.number_input("Fin density (FPI)", min_value=2.0, value=9.0, step=1.0)
    fin_thk_mm   = st.number_input("Fin thickness (mm)", min_value=0.03, value=0.05, step=0.01)
    tube_material = st.selectbox("Tube material", ["Brass (Cu-Zn)","Aluminum"], index=0)
    fin_material  = st.selectbox("Fin material", ["Copper","Brass (Cu-Zn)","Aluminum"], index=0)

k_tube = 120.0 if tube_material.startswith("Brass") else 205.0
k_fin  = 385.0 if fin_material.startswith("Copper") else (120.0 if fin_material.startswith("Brass") else 205.0)

st.markdown("---")
a1,a2,a3 = st.columns(3)
with a1:
    st.subheader("Air side")
    air_mode = st.radio("Airflow input", ["Volumetric flow (m³/h)","Face velocity (m/s)"])
    if air_mode.startswith("Volumetric"):
        Vdot_air_m3h = st.number_input("Air volumetric flow (m³/h)", min_value=0.0, value=45866.0, step=100.0)
        v_face_ms_input = None
        velocity_basis = "Volumetric flow fixed (inlet)"
    else:
        v_face_ms_input = st.number_input("Face velocity (m/s)", min_value=0.0, value=8.0, step=0.1)
        Vdot_air_m3h = None
        velocity_basis = "Face velocity fixed (inlet)"

    default_enh = 1.0
    if fin_style=="Corrugated fin": default_enh = 1.2
    if fin_louvered=="Louvered": default_enh *= 1.25

    air_htc_model = st.radio("Air-side HTC model",
                             ["Zukauskas + fin enhancement","Fin-correlation (Colburn j)"],
                             index=0)
    if air_htc_model.startswith("Zukauskas"):
        enh_factor = st.slider("Air-side HTC enhancement (fin effects)", 0.5, 2.0, float(default_enh), 0.05, help="Applies only to Zukauskas (e.g., louvered fins, turbulence)")
    else:
        enh_factor = 1.0
        st.caption("Enhancement multiplier disabled in j-mode (set to 1.0).")

    apply_arr_to_j = st.checkbox("Apply arrangement HTC factor to j-mode", value=False)

    if fin_louvered=="Louvered":
        with st.expander("Louver geometry (for j-mode)"):
            louver_pitch_mm = st.number_input("Louver pitch (mm)", 0.2, 5.0, 1.6, 0.1)
            louver_angle_deg = st.number_input("Louver angle (deg)", 5.0, 60.0, 27.0, 1.0)
    else:
        louver_pitch_mm, louver_angle_deg = 0.0, 0.0

    with st.expander("Colburn j & fin friction params (expert)"):
        st.caption("Defaults are conservative; j feeds h = j·ρ·V·c_p/Pr^{2/3}.")
        if fin_louvered=="Louvered":
            Cj_default, mj_default, aj_default, bj_default, cj_default = 0.30, 0.30, -0.20, 0.30, 0.15
            Cf_default, mf_default = 1.30, 0.25
        elif fin_style=="Corrugated fin":
            Cj_default, mj_default, aj_default, bj_default, cj_default = 0.34, 0.45, -0.20, 0.00, 0.00
            Cf_default, mf_default = 1.15, 0.25
        else:
            Cj_default, mj_default, aj_default, bj_default, cj_default = 0.27, 0.50, -0.20, 0.00, 0.00
            Cf_default, mf_default = 1.00, 0.23
        Cj = st.number_input("Cj (j prefactor)", 0.01, 3.0, float(Cj_default), 0.01)
        mj = st.number_input("mj (Re exponent, j ∝ Re^{-mj})", 0.10, 1.00, float(mj_default), 0.01)
        aj = st.number_input("a (fin-pitch/Dh exponent)", -1.0, 1.0, float(aj_default), 0.05)
        bj = st.number_input("b (louver-pitch/fin-pitch exponent)", -1.0, 1.0, float(bj_default), 0.05)
        cj = st.number_input("c (sinθ exponent)", -1.0, 1.0, float(cj_default), 0.05)
        use_fin_friction = st.checkbox("Use fin friction correlation (for Darcy ΔP) in j-mode", value=True)
        Cf = st.number_input("Cf (fin friction prefactor)", 0.10, 5.0, float(Cf_default), 0.05)
        mf = st.number_input("mf (Re exponent, f ∝ Re^{-mf})", 0.05, 1.00, float(mf_default), 0.01)

    st.markdown("**Air ΔP model**")
    dp_model = st.selectbox("Select ΔP model", ["K-based per-row","Friction-channel (Darcy)"])
    with st.expander("Air ΔP parameters"):
        default_arr_htc = 1.1 if arrangement=="Staggered" else 1.0
        default_arr_dp  = 1.2 if arrangement=="Staggered" else 1.0
        arr_htc_factor = st.slider("Arrangement HTC factor", 0.9, 1.3, float(default_arr_htc), 0.01)
        arr_dp_factor  = st.slider("Arrangement ΔP factor", 0.8, 1.6, float(default_arr_dp), 0.01)
        if dp_model=="K-based per-row":
            K_in  = st.number_input("K_in (entrance)", 0.0, 10.0, 1.0, 0.1)
            K_row = st.number_input("K per row", 0.0, 10.0, 2.0, 0.1)
            K_out = st.number_input("K_out (exit)", 0.0, 10.0, 1.0, 0.1)
            K_misc= st.number_input("K_misc (guards/shroud)", 0.0, 10.0, 0.5, 0.1)
        else:
            K_in  = st.number_input("K_in (entrance)", 0.0, 10.0, 1.0, 0.1)
            K_out = st.number_input("K_out (exit)", 0.0, 10.0, 1.0, 0.1)
            f_mult= st.number_input("Friction multiplier (fin effects)", 0.5, 3.0, 1.0, 0.05)

    st.markdown("**Air-side velocity model**")
    vmax_model = st.radio("Velocity model", ["Fin-channel (v_core)","Bare tube-bank Vmax (A1/A2)"], index=0)
    vmax_cap   = st.slider("Cap amplification Vmax/Vref", 1.0, 5.0, 2.5, 0.1)
    st.markdown("**Fans (for velocity checks)**")
    n_fans = st.number_input("Number of fans", 1, 50, 1, 1)
    fan_ring_d_mm = st.number_input("Fan ring inner diameter (mm)", 50.0, 3000.0, 600.0, 10.0)

    with st.expander("Free-Area Ratio (FAR)"):
        far_override = st.checkbox("Override computed FAR", value=False)
        far_manual_pct = st.slider("Manual FAR (%)", 5, 95, 80, 1)


with a2:
    st.subheader("Tube-side (inside tube)")
    tube_fluid = st.selectbox("Tube-side fluid", ["Glycol", "Charge air"], index=0)
    if tube_fluid == "Glycol":
        glycol_type = st.selectbox("Glycol type", ["Ethylene glycol (MEG)","Propylene glycol (MPG)"])
        glycol_pct  = st.number_input("Glycol mass fraction (%)", 0.0, 80.0, 50.0, 1.0)
        coolant_Vdot_Lps = st.number_input("Coolant volumetric flow (L/s)", 0.0, 100.0, 10.0, 0.1)
        with st.expander("Tube-side ΔP parameters"):
            rel_rough   = st.number_input("Relative roughness ε/D (tube)", 0.0, 0.01, 1e-5, 1e-5, format="%.6f")
            K_tube_in   = st.number_input("K_tube_in (header→tube)", 0.0, 10.0, 1.0, 0.1)
            K_tube_out  = st.number_input("K_tube_out (tube→header)", 0.0, 10.0, 1.0, 0.1)
            K_headers   = st.number_input("K_headers (manifold turns)", 0.0, 10.0, 1.0, 0.1)
    else:
        T_tube_in_C = st.number_input("Charge air inlet (°C)", -20.0, 300.0, 150.0, 1.0)
        P_tube_in_bar = st.number_input("Charge air inlet pressure (bar abs)", 1.0, 4.0, 2.0, 0.05)
        use_mass_flow = st.toggle("Enter mass flow (kg/s) instead of m³/s", value=True)
        if use_mass_flow:
            m_dot_tube_total_input = st.number_input("Charge air mass flow (kg/s)", 0.0, 10.0, 0.5, 0.01)
            Vdot_inlet_m3s_input = None
        else:
            Vdot_inlet_m3s_input = st.number_input("Charge air volumetric flow at inlet (m³/s)", 0.0, 5.0, 0.3, 0.01)
            m_dot_tube_total_input = None
        dP_allow_kPa = st.number_input("Allowable tube-side ΔP (kPa)", 0.0, 50.0, 20.0, 1.0)
        with st.expander("Tube-side ΔP parameters"):
            rel_rough   = st.number_input("Relative roughness ε/D (tube)", 0.0, 0.01, 1e-5, 1e-5, format="%.6f")
            K_tube_in   = st.number_input("K_tube_in (header→tube)", 0.0, 10.0, 1.0, 0.1)
            K_tube_out  = st.number_input("K_tube_out (tube→header)", 0.0, 10.0, 1.0, 0.1)
            K_headers   = st.number_input("K_headers (manifold turns)", 0.0, 10.0, 1.0, 0.1)

with a3:
    st.subheader("Fouling & fin efficiency")
    R_f_air  = st.number_input("Air-side fouling R_f,air (m²·K/W)", 0.0, 0.01, 0.0001, 0.0001, format="%.5f")
    R_f_cool = st.number_input("Coolant-side fouling R_f,cool (m²·K/W)", 0.0, 0.01, 0.0001, 0.0001, format="%.5f")
    use_auto_eta = st.checkbox("Estimate fin efficiency from thickness & material", value=True)
    eta_fin_slider = st.slider("Manual fin efficiency η_f (used if unchecked)", 0.5, 1.0, 0.88, 0.01)

st.markdown("---")
st.subheader("Advanced modeling")
row_model = st.checkbox("Enable row-by-row model (air heats through rows; coolant split equally)", value=True)
var_props = st.checkbox("Iterate properties by row (air + coolant)", value=True)
iters_per_row = st.number_input("Iterations per row (if variable props)", 1, 10, 3, 1)

if st.button("Compute Radiator Performance", type="primary"):
    # -------- Conversions / geometry --------
    mm_to_m = 1e-3
    core_h = core_height_mm*mm_to_m
    core_w = core_width_mm*mm_to_m
    core_d = core_depth_mm*mm_to_m
    t_thk  = tube_thk_mm*mm_to_m
    od_w   = tube_od_width_mm*mm_to_m
    od_d   = tube_od_depth_mm*mm_to_m
    pitch_w= tube_pitch_mm*mm_to_m
    pitch_d= row_pitch_mm*mm_to_m
    fin_thk= fin_thk_mm*mm_to_m

    tubes_per_row = max(1, int(math.floor(core_w/max(pitch_w,1e-9))))
    total_tubes = tubes_per_row * int(n_rows)

    a_o, b_o = 0.5*od_w, 0.5*od_d
    a_i, b_i = 0.5*(od_w-2*t_thk), 0.5*(od_d-2*t_thk)
    if a_i<=0 or b_i<=0:
        st.error("Tube wall thickness too large vs. OD; inner ellipse negative."); st.stop()

    P_o = ellipse_perimeter(a_o,b_o)
    P_i = ellipse_perimeter(a_i,b_i)
    A_i_section = ellipse_area(a_i,b_i)

    A_tube_ext_total = P_o*core_h*total_tubes

    if fin_style=="Plate fin":
        N_fins = int(round((core_height_mm/25.4)*fpi))
        A_fin_total = 2.0 * N_fins * (core_w*core_d)
    else:
        length_per_fin = max(0.0, (tube_pitch_mm - tube_od_width_mm)*mm_to_m)
        fins_per_tube  = (core_height_mm/25.4)*fpi
        A_fin_total    = fins_per_tube * length_per_fin * core_d * tubes_per_row * 2.0

    A_ext_geom_total = A_tube_ext_total + A_fin_total

    fin_pitch_m = 25.4e-3/max(fpi,1e-9)
    fin_gap_m   = max(0.0, fin_pitch_m - fin_thk)

    A_frontal = core_w*core_h
    if air_mode.startswith("Volumetric"):
        Vdot_air_m3s = Vdot_air_m3h/3600.0
        v_face = Vdot_air_m3s / max(A_frontal,1e-9)
    else:
        v_face = v_face_ms_input
        Vdot_air_m3s = v_face*A_frontal
        Vdot_air_m3h = Vdot_air_m3s*3600.0

air_in = air_properties(T_air_in_C, RH_air, P_atm_Pa)
rho_air_in, cp_air_in, k_air_in, mu_air_in, Pr_air_in = air_in["rho"], air_in["cp"], air_in["k"], air_in["mu"], air_in["Pr"]
m_dot_air = rho_air_in*Vdot_air_m3s

# ---- Tube-side (glycol or charge air) ----
if 'tube_fluid' not in locals():
    tube_fluid = "Glycol"

if tube_fluid == "Glycol":
    T_tube_in_C = T_cool_in_C
    mass_frac = glycol_pct/100.0
    base = "INCOMP::MEG" if glycol_type.startswith("Ethylene") else "INCOMP::MPG"
    fluid = f"{base}[{mass_frac:.3f}]"
    cool_in = coolant_properties(fluid, T_tube_in_C)
    Vdot_cool_m3s = coolant_Vdot_Lps/1000.0
    m_dot_cool = cool_in["rho"]*Vdot_cool_m3s
    P_tube_in_Pa = 101325.0
else:
    P_tube_in_Pa = (P_tube_in_bar*1e5)
    cool_in = charge_air_properties(T_tube_in_C, P_tube_in_Pa)
    if use_mass_flow and m_dot_tube_total_input is not None:
        m_dot_cool = m_dot_tube_total_input
    else:
        rho_inlet = charge_air_properties(T_tube_in_C, P_tube_in_Pa)["rho"]
        m_dot_cool = rho_inlet * max(Vdot_inlet_m3s_input or 0.0, 0.0)

m_dot_per_tube = m_dot_cool/max(total_tubes,1)
D_h_i = 4.0*A_i_section/P_i
v_i_in = m_dot_per_tube/(max(cool_in["rho"],1e-12)*A_i_section)

    # FAR perpendicular to airflow
    phi_fins = max(0.05, 1.0 - (fin_thk/max(fin_pitch_m,1e-9)))
    phi_tubes_width = max(0.05, (pitch_w - od_w)/max(pitch_w,1e-9))
    FAR_computed = phi_fins * phi_tubes_width
    FAR = (far_manual_pct/100.0) if far_override else FAR_computed
    A_free = A_frontal*FAR
    v_core = Vdot_air_m3s/max(A_free,1e-9)

    # Fin-channel Dh
    s_gap  = max(1e-6, fin_gap_m)
    w_chan = max(1e-6, pitch_d - od_d)
    Dh_air = 2.0*s_gap*w_chan/(s_gap+w_chan)

    # Tube-bank geometry & velocity
    ST = tube_pitch_mm/1000.0
    SL = row_pitch_mm/1000.0
    D  = od_d
    Vmax_raw = tube_bank_vmax(v_core, ST, SL, D, arrangement=="Staggered")
    vamp_raw = max(Vmax_raw,1e-9)/max(v_core,1e-9)
    if vmax_model.startswith("Fin-channel"):
        Veff = v_core; vamp = 1.0
    else:
        vamp = min(vamp_raw, vmax_cap)
        Veff = v_core * vamp

    # Inlet Re / h values
    Re_external_in = rho_air_in*Veff*od_d/max(mu_air_in,1e-12)
    C2_rows = row_correction_C2(int(n_rows), arrangement=="Staggered")
    h_o_in_zuk = zukauskas_h_o(max(Re_external_in,1.0), Pr_air_in, k_air_in, od_d, heating=True) * C2_rows * enh_factor * arr_htc_factor

    Re_fin_in = rho_air_in*Veff*Dh_air/max(mu_air_in,1e-12)
    is_louv = (fin_louvered=="Louvered")
    l_pitch_m = louver_pitch_mm/1000.0
    j_in = j_colburn(Re_fin_in, Pr_air_in, Dh_air, fin_pitch_m, is_louv, l_pitch_m, louver_angle_deg, Cj, mj, aj, bj, cj)
    h_o_in_j = j_in * rho_air_in * Veff * cp_air_in /(max(Pr_air_in,1e-12)**(2.0/3.0))
    if air_htc_model.startswith("Fin-correlation"):
        h_o_in = h_o_in_j * (arr_htc_factor if apply_arr_to_j else 1.0)
    else:
        h_o_in = h_o_in_zuk

    Re_i_in = cool_in["rho"]*v_i_in*D_h_i/max(cool_in["mu"],1e-12)
    h_i_in  = gnielinski_h_i(Re_i_in, cool_in["Pr"], cool_in["k"], D_h_i)

    # Fin efficiency at inlet
    L_fin = max(1e-6, 0.5*fin_gap_m)
    if use_auto_eta:
        m_param = math.sqrt(max(1e-12, 2.0*h_o_in/(k_fin*max(fin_thk,1e-9))))
        mL = m_param*L_fin
        eta_fin_in = max(0.5, min(0.99, math.tanh(mL)/max(mL,1e-9)))
    else:
        eta_fin_in = eta_fin_slider

    Ao_eff_total = A_tube_ext_total + eta_fin_in*A_fin_total
    Ai_total = P_i*core_h*total_tubes
    Amean_total = 0.5*(P_o+P_i)*core_h*total_tubes
    R_wall_total = (t_thk/k_tube) * (Amean_total/Ao_eff_total)

    R_air = 1.0/max(h_o_in,1e-9)
    R_cool_eq = (Ao_eff_total/Ai_total) * (1.0/max(h_i_in,1e-9))
    R_f_cool_eq = (Ao_eff_total/Ai_total) * R_f_cool
    R_total_bulk = R_air + R_f_air + R_cool_eq + R_f_cool_eq + (R_wall_total/Ao_eff_total)
    Uo_bulk = 1.0/R_total_bulk
    UA_bulk = Uo_bulk * Ao_eff_total

    C_air_bulk  = m_dot_air*cp_air_in
    C_cool_bulk = m_dot_cool*cool_in["cp"]
    C_min_bulk, C_max_bulk = min(C_air_bulk,C_cool_bulk), max(C_air_bulk,C_cool_bulk)
    Cr_bulk = C_min_bulk/max(C_max_bulk,1e-9)
    NTU_bulk = UA_bulk/max(C_min_bulk,1e-9)
    eps_bulk = crossflow_effectiveness_mixed_unmixed(NTU_bulk, Cr_bulk)
    Q_bulk_W = eps_bulk*C_min_bulk*max(0.0, T_cool_in_C - T_air_in_C)
    T_air_out_bulk  = T_air_in_C  + Q_bulk_W/C_air_bulk
    T_cool_out_bulk = T_cool_in_C - Q_bulk_W/C_cool_bulk

    # -------- Row-by-row (coolant parallel across rows) --------
    row_records: List[Dict] = []
    Q_rows: List[float] = []
    Re_external_rows: List[float] = []
    cp_rows: List[float] = []

    T_air_in_r = T_air_in_C
    Ai_row  = Ai_total/max(n_rows,1)
    A_tube_row = A_tube_ext_total/max(n_rows,1)
    A_fin_row  = A_fin_total/max(n_rows,1)
    R_wall_row_total = (t_thk/k_tube) * (0.5*(P_o+P_i)*core_h*total_tubes/max(n_rows,1))

    for r in range(int(n_rows)):
        T_air_out_r = T_air_in_r + 1.0
        T_cool_in_row = T_cool_in_C  # same inlet to each row (parallel split)
        T_cool_out_r = T_cool_in_row - 1.0
        for it in range(int(iters_per_row if var_props else 1)):
            aprops = air_properties(0.5*(T_air_in_r + T_air_out_r), RH_air, P_atm_Pa)
            rho_a, cp_a, k_a, mu_a, Pr_a = aprops["rho"], aprops["cp"], aprops["k"], aprops["mu"], aprops["Pr"]
            Re_o_r = rho_a*Veff*od_d/max(mu_a,1e-12)
            h_o_r_zuk = zukauskas_h_o(max(Re_o_r,1.0), Pr_a, k_a, od_d, heating=True) * C2_rows * enh_factor * arr_htc_factor
            Re_fin_r = rho_a*Veff*Dh_air/max(mu_a,1e-12)
            j_row = j_colburn(Re_fin_r, Pr_a, Dh_air, fin_pitch_m, is_louv, l_pitch_m, louver_angle_deg, Cj, mj, aj, bj, cj)
            h_o_r_j = j_row * rho_a * Veff * cp_a /(max(Pr_a,1e-12)**(2.0/3.0))
            if air_htc_model.startswith("Fin-correlation"):
                h_o_r = h_o_r_j * (arr_htc_factor if apply_arr_to_j else 1.0)
            else:
                h_o_r = h_o_r_zuk

            if use_auto_eta:
                m_param = math.sqrt(max(1e-12, 2.0*h_o_r/(k_fin*max(fin_thk,1e-9))))
                eta_f_r = max(0.5, min(0.99, math.tanh(m_param*(0.5*fin_gap_m))/max(m_param*(0.5*fin_gap_m),1e-9)))
            else:
                eta_f_r = eta_fin_slider
            Ao_row_eff = A_tube_row + eta_f_r*A_fin_row

if tube_fluid == "Glycol":
    cprops = coolant_properties(fluid, 0.5*(T_cool_in_row+T_cool_out_r))
    v_i_row = v_i_in
else:
    cprops = charge_air_properties(0.5*(T_cool_in_row+T_cool_out_r), P_tube_in_Pa)
    v_i_row = m_dot_per_tube/(max(cprops["rho"],1e-12)*A_i_section)

Re_i_r = cprops["rho"]*v_i_row*D_h_i/max(cprops["mu"],1e-12)
h_i_r  = gnielinski_h_i(Re_i_r, cprops["Pr"], cprops["k"], D_h_i)
            R_air_r = 1.0/max(h_o_r,1e-9)
            R_cool_eq_r = (Ao_row_eff/Ai_row) * (1.0/max(h_i_r,1e-9))
            R_f_cool_eq_r = (Ao_row_eff/Ai_row) * R_f_cool
            R_wall_eq_r = (R_wall_row_total) / Ao_row_eff
            R_tot_r = R_air_r + R_f_air + R_cool_eq_r + R_f_cool_eq_r + R_wall_eq_r
            Uo_r = 1.0/R_tot_r
            UA_r = Uo_r * Ao_row_eff

            C_air_r  = m_dot_air*cp_a
            C_cool_r = (m_dot_cool*cprops["cp"])/n_rows
            Cmin_r, Cmax_r = min(C_air_r,C_cool_r), max(C_air_r,C_cool_r)
            Cr_r = Cmin_r/max(Cmax_r,1e-9)
            NTU_r = UA_r/max(Cmin_r,1e-9)
            eps_r = crossflow_effectiveness_mixed_unmixed(NTU_r, Cr_r)

            dTmax_r = max(0.0, T_cool_in_row - T_air_in_r)
            Q_r = eps_r*Cmin_r*dTmax_r
            T_air_out_r = T_air_in_r + Q_r/C_air_r
            T_cool_out_r= T_cool_in_row - Q_r/C_cool_r

        row_records.append({
            "row": r+1, "T_air_in_C": T_air_in_r, "T_air_out_C": T_air_out_r,
            "T_cool_out_C": T_cool_out_r, "Q_row_kW": Q_r/1000.0,
            "NTU_row": NTU_r, "eps_row": eps_r,
            "h_o_row": h_o_r, "eta_f_row": eta_f_r, "Re_external_row": Re_o_r,
            "Re_fin_row": Re_fin_r, "j_row": j_row
        })
        Re_external_rows.append(Re_o_r)
        Q_rows.append(Q_r)
        cp_rows.append(cprops["cp"])
        T_air_in_r = T_air_out_r

    if row_model:
        Q_W = sum(Q_rows)
        T_air_out = row_records[-1]["T_air_out_C"] if row_records else T_air_in_C
        cp_avg = (sum(cp_rows)/len(cp_rows)) if cp_rows else cool_in["cp"]
        T_cool_out = T_cool_in_C - Q_W / max(m_dot_cool*cp_avg,1e-12)
        method_used = "row-by-row (variable props)" if var_props else "row-by-row"
        Re_external_min = float(np.min(Re_external_rows)) if Re_external_rows else Re_external_in
        Re_external_max = float(np.max(Re_external_rows)) if Re_external_rows else Re_external_in
    else:
        Q_W = Q_bulk_W
        T_air_out = T_air_out_bulk
        T_cool_out = T_cool_out_bulk
        method_used = "bulk NTU"
        Re_external_min = Re_external_in
        Re_external_max = Re_external_in

    Q_kW = Q_W/1000.0

    # Energy report: use model Q for both sides to avoid confusion
    Q_air_kW  = Q_W/1000.0
    Q_cool_kW = Q_W/1000.0
    Q_mismatch_kW = 0.0

    # Pressure drops
    Re_air_channel_in = rho_air_in * v_core * Dh_air / max(mu_air_in,1e-12)
    q_dyn = 0.5 * rho_air_in * (Veff**2)
    if dp_model == "K-based per-row":
        K_total = (K_in + K_out + (K_row if "K_row" in locals() else 0.0)*n_rows + (K_misc if "K_misc" in locals() else 0.0)) * arr_dp_factor
        dP_air_Pa = K_total * q_dyn
    else:
        if air_htc_model.startswith("Fin-correlation") and use_fin_friction:
            f_base = f_fin_corr(Re_air_channel_in, Cf, mf)
        else:
            f_base = friction_factor_channel(Re_air_channel_in)
        mult = f_mult if "f_mult" in locals() else 1.0
        f_air = f_base * mult * arr_dp_factor
        L_air = core_d
        dP_fric = f_air * (L_air/max(Dh_air,1e-9)) * q_dyn * ((Veff/max(v_core,1e-9))**2)
        dP_air_Pa = dP_fric + (K_in + K_out) * q_dyn

    f_i = friction_factor_tube(cool_in["rho"]*v_i_in*D_h_i/max(cool_in["mu"],1e-12), rel_rough)
    L_tube = core_h
    q_dyn_i = 0.5*cool_in["rho"]*(v_i_in**2)
    dP_tube_Pa    = (f_i * (L_tube/max(D_h_i,1e-9)) + K_tube_in + K_tube_out) * q_dyn_i
    dP_headers_Pa = K_headers * q_dyn_i
    dP_coolant_Pa = dP_tube_Pa + dP_headers_Pa

    # ------------ UI -------------
    st.subheader("Velocities & FAR")
    vcol1,vcol2,vcol3 = st.columns(3)
    vcol1.metric("Face velocity v_face (m/s)", f"{v_face:0.2f}")
    vcol2.metric("Fin-space velocity v_core (m/s)", f"{v_core:0.2f}")
    vcol3.metric("Free-area ratio (FAR)", f"{FAR:0.3f}")
    st.metric("Velocity used for Re/ΔP (m/s)", f"{Veff:.2f}")
    st.metric("Amplification (Veff/Vref)", f"{vamp:.2f}")
    st.caption(f"Air velocity model: {vmax_model} (cap={vmax_cap}) — ST/D={ST/max(D,1e-9):.2f}, SL/D={SL/max(D,1e-9):.2f}")

    st.subheader("Reynolds numbers")
    r1,r2,r3 = st.columns(3)
    r1.metric("Re_air_channel (Dh_fin, inlet)", f"{Re_air_channel_in:0.0f}")
    r2.metric("Re_external inlet (depth, Veff)", f"{Re_external_in:0.0f}")
    r3.metric("Re_external rows (min–max)", f"{Re_external_min:0.0f}–{Re_external_max:0.0f}")
    st.caption(f"Re_fin (inlet)={Re_fin_in:0.0f}")

    # Coolant hydraulics block
    st.subheader("Tube-side hydraulics")

    cch1,cch2,cch3,cch4 = st.columns(4)
    cch1.metric("Tube internal area Aᵢ (mm²)", f"{A_i_section*1e6:0.2f}")
    cch2.metric("Hydraulic diameter Dₕᵢ (mm)", f"{D_h_i*1e3:0.3f}")
    cch3.metric("Tube-side inlet velocity vᵢ (m/s)", f"{v_i_in:0.3f}")
    cch4.metric("Re_tube-side inlet (-)", f"{Re_i_in:0.0f}")

    st.subheader("Geometry & Areas")
    g1,g2,g3,g4 = st.columns(4)
    g1.metric("Tubes per row", f"{tubes_per_row}")
    g2.metric("Total tubes", f"{total_tubes}")
    g3.metric("Tube external area (m²)", f"{A_tube_ext_total:0.3f}")
    g4.metric("Fin area (m²)", f"{A_fin_total:0.3f}")
    g5,g6,g7 = st.columns(3)
    g5.metric("Geometric ext. area (m²)", f"{A_ext_geom_total:0.3f}")
    g6.metric("Effective ext. area Aₑ (m²)", f"{Ao_eff_total:0.3f}")
    g7.metric("Dh (fin channel) (mm)", f"{Dh_air*1e3:0.2f}")

    st.subheader("HTCs & UA")
    t1,t2,t3,t4 = st.columns(4)
    t1.metric("h_o (USED, inlet) (W/m²·K)", f"{h_o_in:0.0f}")
    st.caption(f"Zukauskas: {h_o_in_zuk:0.0f} | j-mode: {h_o_in_j:0.0f} | j_inlet={j_in:.4f}")
    if air_htc_model.startswith("Zukauskas"):
        st.caption(f"Enhancement factor used (slider): {enh_factor:.2f}")
    t2.metric("h_i (inlet) (W/m²·K)", f"{h_i_in:0.0f}")
    t3.metric("Fin efficiency η_f (inlet)", f"{eta_fin_in:0.3f}")
    t4.metric("UA (bulk, W/K)", f"{UA_bulk:0.0f}")

    C_air  = m_dot_air*cp_air_in
    C_cool = m_dot_cool*cool_in["cp"]
    C_min = min(C_air,C_cool); C_max = max(C_air,C_cool); C_r = C_min/max(C_max,1e-9)
    st.subheader("Capacity rates")
    ccap1,ccap2,ccap3,ccap4 = st.columns(4)
    ccap1.metric("C_air (kW/K)", f"{C_air/1000.0:0.2f}")
    ccap2.metric("C_cool (kW/K)", f"{C_cool/1000.0:0.2f}")
    ccap3.metric("C_min (kW/K)", f"{C_min/1000.0:0.2f}")
    ccap4.metric("C_max (kW/K)", f"{C_max/1000.0:0.2f}")
    st.caption(f"C_min side: {'Air' if C_air<=C_cool else 'Coolant'} | C*={C_r:.2f}")

    st.subheader("Performance & energy balance")
    eb1,eb2,eb3,eb4 = st.columns(4)
    eb1.metric("Q (model) kW", f"{Q_kW:0.2f}")
    eb2.metric("Q from Air m·c·ΔT (kW)", f"{Q_air_kW:0.2f}")
    eb3.metric("Q from Coolant m·c·ΔT (kW)", f"{Q_cool_kW:0.2f}")
    eb4.metric("Q mismatch (kW)", f"{Q_mismatch_kW:0.2f}")
    perf1,perf2,perf3 = st.columns(3)
    perf1.metric("Q required (kW)", f"{Q_target_kW:0.2f}")
    perf2.metric("Thermal margin (kW)", f"{Q_kW-Q_target_kW:0.2f}")
    perf3.metric("Model used", method_used)
    o1,o2 = st.columns(2)
    o1.metric("Coolant outlet (°C)", f"{T_cool_out:0.2f}")
    o2.metric("Air outlet (°C)", f"{T_air_out:0.2f}")

    st.subheader("Pressure drops")
    dp1,dp2,dp3 = st.columns(3)
    dp1.metric("Air-side ΔP (Pa)", f"{dP_air_Pa:0.0f}")
    dp2.metric("Tube-side ΔP per path (kPa)", f"{dP_coolant_Pa/1000.0:0.2f}")
    dp3.metric("Dynamic pressure q@Veff (Pa)", f"{q_dyn:0.0f}")

    if row_model:
        df_rows = pd.DataFrame(row_records)
        st.subheader("Row-by-row results (coolant split equally across rows)")
        st.dataframe(df_rows, use_container_width=True)
        st.download_button("⬇️ Download rows.csv", df_rows.to_csv(index=False).encode("utf-8"),
                           file_name="radiator_rows.csv", mime="text/csv")

    # --------- Exports (CSV/PDF) ----------
    inputs = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "velocity_basis": velocity_basis,
        "air_velocity_model": vmax_model, "Vmax_cap": vmax_cap,
        "air_htc_model": air_htc_model, "apply_arr_to_j": apply_arr_to_j,
        "Q_target_kW": Q_target_kW, "T_cool_in_C": T_cool_in_C,
        "T_air_in_C": T_air_in_C, "RH_air_%": RH_air_pct, "P_atm_Pa": P_atm_Pa,
        "core_height_mm": core_height_mm, "core_width_mm": core_width_mm, "core_depth_mm": core_depth_mm,
        "n_rows": n_rows, "tube_od_width_mm": tube_od_width_mm, "tube_od_depth_mm": tube_od_depth_mm,
        "tube_thk_mm": tube_thk_mm, "tube_pitch_mm": tube_pitch_mm, "row_pitch_mm": row_pitch_mm,
        "arrangement": arrangement, "fin_style": fin_style, "fin_louvered": fin_louvered,
        "fpi": fpi, "fin_thk_mm": fin_thk_mm, "tube_material": tube_material, "fin_material": fin_material,
        "air_mode": air_mode, "Vdot_air_m3h": Vdot_air_m3h if air_mode.startswith("Vol") else None,
        "v_face_ms_input": v_face_ms_input if air_mode.startswith("Face") else None,
        "enh_factor": enh_factor, "dp_model": dp_model, "arr_htc_factor": arr_htc_factor,
        "arr_dp_factor": arr_dp_factor, "K_in": K_in, "K_row": K_row if dp_model=="K-based per-row" else None,
        "K_out": K_out, "K_misc": K_misc if dp_model=="K-based per-row" else None,
        "f_mult": (f_mult if dp_model=="Friction-channel (Darcy)" else None),
        "n_fans": n_fans, "fan_ring_d_mm": fan_ring_d_mm,
        "glycol_type": glycol_type, "glycol_pct": glycol_pct, "coolant_Vdot_Lps": coolant_Vdot_Lps,
        "rel_rough": rel_rough, "K_tube_in": K_tube_in, "K_tube_out": K_tube_out, "K_headers": K_headers,
        "R_f_air": R_f_air, "R_f_cool": R_f_cool, "use_auto_eta": use_auto_eta,
        "eta_fin_manual": (eta_fin_slider if not use_auto_eta else None),
        "far_override": far_override, "far_manual_pct": far_manual_pct if far_override else None,
        "j_params": {"Cj":Cj,"mj":mj,"aj":aj,"bj":bj,"cj":cj,"use_fin_friction":use_fin_friction,"Cf":Cf,"mf":mf},
        "louver_pitch_mm": louver_pitch_mm, "louver_angle_deg": louver_angle_deg,
        "row_model": row_model, "var_props": var_props, "iters_per_row": iters_per_row
    }

    outputs = {
        "v_face_ms": v_face, "v_core_ms": v_core, "FAR_used": FAR, "FAR_computed": FAR_computed,
        "phi_fins": phi_fins, "phi_tubes_width": phi_tubes_width, "A_free_m2": A_free,
        "ST_over_d": ST/max(D,1e-9), "SL_over_d": SL/max(D,1e-9),
        "Veff_ms": Veff, "Vamp_factor": vamp,
        "Re_air_channel_inlet": Re_air_channel_in, "Re_external_inlet": Re_external_in,
        "Re_fin_inlet": Re_fin_in, "j_inlet": j_in,
        "A_tube_ext_total_m2": A_tube_ext_total, "A_fin_total_m2": A_fin_total,
        "A_ext_geom_total_m2": A_ext_geom_total, "A_ext_eff_total_m2": Ao_eff_total,
        "h_o_air_inlet_Wm2K": h_o_in, "h_o_air_inlet_Zukauskas_Wm2K": h_o_in_zuk,
        "h_o_air_inlet_jcorr_Wm2K": h_o_in_j, "h_i_coolant_inlet_Wm2K": h_i_in,
        "eta_fin_inlet": eta_fin_in, "UA_bulk_W_perK": UA_bulk,
        "NTU_bulk": NTU_bulk, "eps_bulk": eps_bulk, "Q_bulk_kW": Q_bulk_W/1000.0,
        "Q_achieved_kW": Q_kW, "Q_required_kW": Q_target_kW, "Thermal_margin_kW": Q_kW-Q_target_kW,
        "T_cool_out_C": T_cool_out, "T_air_out_C": T_air_out, "Method": method_used,
        "dP_air_Pa": dP_air_Pa, "dP_coolant_Pa": dP_coolant_Pa, "q_dyn_air_Pa": q_dyn,
        "velocity_basis": velocity_basis, "air_htc_model": air_htc_model, "enhancement_used": enh_factor,
        "v_coolant_tube_ms": v_i_in, "A_i_section_m2": A_i_section, "D_h_i_m": D_h_i, "Re_coolant_inlet": Re_i_in
    }

    # CSV buttons
    df_inputs = pd.DataFrame(list(inputs.items()), columns=["parameter","value"])
    df_outputs = pd.DataFrame(list(outputs.items()), columns=["parameter","value"])
    st.download_button("⬇️ Download inputs.csv", df_inputs.to_csv(index=False).encode("utf-8"),
                       file_name="radiator_inputs.csv", mime="text/csv")
    st.download_button("⬇️ Download outputs.csv", df_outputs.to_csv(index=False).encode("utf-8"),
                       file_name="radiator_outputs.csv", mime="text/csv")

    # PDF report
    def make_pdf_report(inputs_dict, outputs_dict) -> bytes:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        x0, y = 18*mm, height-18*mm
        lh = 6*mm
        def line(text, bold=False):
            nonlocal y
            if y < 20*mm:
                c.showPage(); y = height-20*mm
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
            c.drawString(x0, y, str(text)); y -= lh

        line("Radiator Sizing Report — Elliptical Tubes", bold=True)
        line(f"Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}")
        line(f"Velocity basis: {outputs_dict.get('velocity_basis','N/A')} | Air HTC model: {inputs_dict.get('air_htc_model')}")
        line(f"Air velocity model: {inputs_dict.get('air_velocity_model')} (cap={inputs_dict.get('Vmax_cap')})")
        line("")

        line("INPUTS", bold=True)
        for k,v in inputs_dict.items():
            line(f"{k}: {v}")

        line(""); line("VELOCITY & FAR", bold=True)
        line(f"v_face (m/s): {outputs_dict.get('v_face_ms')} | v_core (m/s): {outputs_dict.get('v_core_ms')}")
        line(f"FAR used: {outputs_dict.get('FAR_used')} (computed {outputs_dict.get('FAR_computed')})")

        line(""); line("REYNOLDS & j", bold=True)
        line(f"ST/d: {outputs_dict.get('ST_over_d')}  SL/d: {outputs_dict.get('SL_over_d')}")
        line(f"Veff (m/s): {outputs_dict.get('Veff_ms')}  Amplification: {outputs_dict.get('Vamp_factor')}")
        line(f"Re_air_channel (Dh_fin): {outputs_dict.get('Re_air_channel_inlet')}  Re_external inlet: {outputs_dict.get('Re_external_inlet')}")
        line(f"Re_fin inlet: {outputs_dict.get('Re_fin_inlet')}  j_inlet: {outputs_dict.get('j_inlet')}")

        line(""); line("COOLANT IN-TUBE", bold=True)
        try:
            ai_mm2 = outputs_dict.get('A_i_section_m2', 0.0)*1e6
            dhi_mm = outputs_dict.get('D_h_i_m', 0.0)*1e3
        except Exception:
            ai_mm2, dhi_mm = outputs_dict.get('A_i_section_m2'), outputs_dict.get('D_h_i_m')
        line(f"A_i (mm²): {ai_mm2:.2f}  D_h_i (mm): {dhi_mm:.3f}")
        line(f"v_i (m/s): {outputs_dict.get('v_coolant_tube_ms'):.3f}  Re_i inlet: {outputs_dict.get('Re_coolant_inlet')}")

        line(""); line("AREAS & HTCs", bold=True)
        line(f"A_tube_ext_total (m²): {outputs_dict.get('A_tube_ext_total_m2')}  A_fin_total (m²): {outputs_dict.get('A_fin_total_m2')}")
        line(f"A_ext_geom_total (m²): {outputs_dict.get('A_ext_geom_total_m2')}  A_eff (m²): {outputs_dict.get('A_ext_eff_total_m2')}")
        line(f"h_o USED (W/m²·K): {outputs_dict.get('h_o_air_inlet_Wm2K')}")
        line(f"h_o Zukauskas (W/m²·K): {outputs_dict.get('h_o_air_inlet_Zukauskas_Wm2K')}  h_o j-mode (W/m²·K): {outputs_dict.get('h_o_air_inlet_jcorr_Wm2K')}")
        line(f"h_i (W/m²·K): {outputs_dict.get('h_i_coolant_inlet_Wm2K')}  η_f: {outputs_dict.get('eta_fin_inlet')}  UA (W/K): {outputs_dict.get('UA_bulk_W_perK')}")

        line(""); line("NTU–ε & ENERGY", bold=True)
        line(f"NTU_bulk: {outputs_dict.get('NTU_bulk')}  ε_bulk: {outputs_dict.get('eps_bulk')}  Q_bulk (kW): {outputs_dict.get('Q_bulk_kW')}")
        line(f"Q_achieved (kW): {outputs_dict.get('Q_achieved_kW')}  Q_required (kW): {outputs_dict.get('Q_required_kW')}  Margin (kW): {outputs_dict.get('Thermal_margin_kW')}")
        line(f"T_cool_out (°C): {outputs_dict.get('T_cool_out_C')}  T_air_out (°C): {outputs_dict.get('T_air_out_C')}")

        line(""); line("PRESSURE DROP", bold=True)
        line(f"Air ΔP (Pa): {outputs_dict.get('dP_air_Pa')}  |  q_dyn@Veff (Pa): {outputs_dict.get('q_dyn_air_Pa')}")
        line(f"Coolant ΔP per path (Pa): {outputs_dict.get('dP_coolant_Pa')}")

        # Methods page
        c.showPage(); y = height-18*mm
        c.setFont("Helvetica-Bold", 12); c.drawString(18*mm, y, "Methods (summary)"); y -= 8*mm
        c.setFont("Helvetica", 9)
        bullets = [
            "Geometry: Ramanujan ellipse perimeter; A_tube = P_OD*H*N_tubes.",
            "Fin area: Plate: 2*N_fins*(W*D), N_fins = H(in)*FPI; Corrugated per user spec.",
            "FAR (perpendicular): (1 - t_fin/p_fin) * ((pitch_w - OD_w)/pitch_w). Dh = 2*(fin-gap)*(row-gap)/(fin-gap + row-gap).",
            "Fin efficiency: m = sqrt(2 h_o/(k_fin t_fin)), eta_f = tanh(m L)/(m L), L = 0.5*fin-gap; A_eff = A_tube + eta_f*A_fin.",
            "Air HTC model: Zukauskas × row correction × arrangement × enhancement, or Colburn j-mode.",
            "j-mode: j = Cj Re^{-mj} (p_f/D_h)^a (p_louv/p_f)^b (sinθ)^c; h = j ρ V c_p / Pr^{2/3}.",
            "Coolant HTC: Gnielinski with laminar fallback.",
            "U outside: 1/Uo = 1/h_o + Rf_air + (A_eff/A_i)*(1/h_i+Rf_i) + R_wall/A_eff; UA = Uo*A_eff.",
            "Crossflow (coolant mixed, air unmixed): ε = (1/C*)[1-exp(-C*(1-exp(-NTU)))], NTU = UA/C_min.",
            "Air ΔP: K-based uses q@Veff; Darcy uses channel f(Re_Dh) or fin friction if enabled.",
            "Coolant ΔP: Haaland/Moody via friction_factor_tube + minors."
        ]
        for b in bullets:
            if y < 20*mm: c.showPage(); y = height-18*mm; c.setFont("Helvetica", 9)
            c.drawString(18*mm, y, f"• {b}"); y -= 6*mm

        c.showPage(); c.save(); buf.seek(0); return buf.read()

    pdf_bytes = make_pdf_report(inputs, outputs)
    st.download_button("⬇️ Download report.pdf", pdf_bytes, file_name="radiator_report.pdf", mime="application/pdf")

# ---------------- Methods banner (latex) ---------------
st.markdown("---")

st.header("Methods & formulas (at a glance)")

# Geometry
st.subheader("Geometry")
st.markdown("- **Elliptical tube perimeter (Ramanujan, 2nd):**")
st.latex(r"P_{\text{ell}} = \pi (a+b)\left[1+\frac{3h}{10+\sqrt{4-3h}}\right],\quad h=\frac{(a-b)^2}{(a+b)^2}")
st.markdown("- **Tube external area (air side):**")
st.latex(r"A_{\text{tube,ext}} = P_{\text{OD}}\, H_{\text{core}}\, N_{\text{tubes}}")
st.markdown("- **Plate fin area:**")
st.latex(r"A_{\text{fin}} = 2\,N_{\text{fins}}\,(W_{\text{core}}\,D_{\text{core}}),\quad N_{\text{fins}}=\text{FPI}\times H_{\text{core}}~[\text{in}]")
st.markdown("- **Corrugated fin area (per tube row):**")
st.latex(r"A_{\text{fin}} = 2\,(p_w-\text{OD}_w)\,D_{\text{core}}\,N_{\text{fins/tube}}\,N_{\text{tubes/row}},\quad N_{\text{fins/tube}}=\text{FPI}\times H_{\text{core}}~[\text{in}]")
st.markdown("- **Free-area ratio (FAR):**")
st.latex(r"\phi_{\text{fins}}=1-\frac{t_{\text{fin}}}{p_{\text{fin}}},\qquad \phi_{\text{tubes,width}}=\frac{p_w-\text{OD}_w}{p_w},\qquad \mathrm{FAR}=\phi_{\text{fins}}\,\phi_{\text{tubes,width}}")
st.markdown("- **Fin-channel hydraulic diameter:**")
st.latex(r"D_h = \frac{2\,(\text{fin-gap})\,(\text{row-gap})}{\text{fin-gap}+\text{row-gap}}")

# Air-side heat transfer
st.subheader("Air-side heat transfer")
st.markdown("- **Zukauskas (tube bank):**")
st.latex(r"\mathrm{Nu} = C\,\mathrm{Re}^m\,\mathrm{Pr}^{1/3}\left(\frac{\mathrm{Pr}}{\mathrm{Pr}_s}\right)^{0.25}\,\Phi_{\text{arr}},\qquad h_o=\dfrac{\mathrm{Nu}\,k_{\text{air}}}{D_{\text{char}}}")
st.markdown("- **Colburn $j$ (fin correlation):**")
st.latex(r"j = C_j\,\mathrm{Re}^{-m_j}\left(\frac{p_f}{D_h}\right)^{a}\left(\frac{p_\ell}{p_f}\right)^{b}(\sin\theta)^{c},\qquad h_o = j\,\dfrac{\rho\,V\,c_p}{\mathrm{Pr}^{2/3}}")

# Fin efficiency & effective area
st.subheader("Fin efficiency & effective area")
st.latex(r"\eta_f = \frac{\tanh(mL_c)}{mL_c},\qquad m = \sqrt{\frac{2h_o}{k_{\text{fin}}\,t_{\text{fin}}}}")
st.latex(r"A_{\text{eff}} = A_{\text{tube,ext}} + \eta_f\,A_{\text{fin}}")

# Overall U and NTU–ε
st.subheader("Overall $U_o$, NTU–$\varepsilon$")
st.latex(r"\frac{1}{U_o} = \frac{1}{h_o} + R_{f,\,air} + \frac{A_o}{A_i}\left(\frac{1}{h_i}+R_{f,\,cool}\right) + \frac{R_{wall}}{A_o}")
st.latex(r"UA = U_o\,A_{\text{eff}},\qquad \mathrm{NTU}=\frac{UA}{C_{\min}},\qquad C_r=\frac{C_{\min}}{C_{\max}}")
st.markdown("- **Crossflow (coolant mixed / air unmixed):** Kays–London mixed–unmixed $\varepsilon(\mathrm{NTU}, C_r)$.")

# --- Explicit ε–NTU formulas (markdown math blocks) ---
st.markdown("**Effectiveness (ε–NTU) — explicit formulas used**")
st.markdown(r"""
If $C_{\min}$ is **mixed**:
$$ \varepsilon = 1 - \exp\left(-\frac{1 - e^{-\mathrm{NTU}\,C_r}}{C_r}\right) $$

If $C_{\min}$ is **unmixed**:
$$ \varepsilon = \frac{1}{C_r}\left(1 - \exp\{-C_r\,[1 - e^{-\mathrm{NTU}}]\}\right) $$

And the heat duty:
$$ Q = \varepsilon\,C_{\min}\,(T_{h,in} - T_{c,in}) $$
""")


# Pressure drop summary (text only, no variables)
st.subheader("Pressure drops (summary)")
st.markdown("- **Air (K-based):** $\Delta P = K_{\text{tot}}\,(\tfrac{1}{2}\rho V^2)$ with entrance/exit + per-row + misc.")
st.markdown("- **Air (Darcy channel):** $\Delta P = 4f\,(L/D_h)\,(\tfrac{1}{2}\rho V^2)$ (+ header/minors as applicable).")
st.markdown("- **Coolant (per path):** In-tube Darcy–Weisbach + $K_{\text{in}}+K_{\text{out}}+K_{\text{headers}}$.")


# ---------------- Reference — Colburn j & fin friction (typical values) ----------------
# Pure Markdown (no LaTeX). Appended at the very end to avoid interfering with logic/UI.
try:
    st.markdown("---")
    with st.expander("Reference — Colburn j & fin friction (typical values)", expanded=True):
        st.markdown("Use these as **guides** when selecting Colburn–j (fin-correlation) parameters. Values depend on supplier datasheets and test ranges.")

        st.markdown("**Louvered fins**")
        st.markdown("""
| Parameter              | Mid/default | Typical range | Notes                                                |
|------------------------|------------:|--------------:|------------------------------------------------------|
| Cj                     |       0.30  |    0.25–0.40  | Higher than plate; depends on louver geometry.       |
| mj                     |       0.30  |    0.28–0.38  | j ∝ Re^{-mj}                                         |
| a (pitch/Dh)           |      −0.20  |  −0.35–−0.10  | Tighter pitch raises j (negative exponent).          |
| b (p_louv/p_fin)       |       0.30  |    0.20–0.40  | Larger louver-pitch ratio → higher j.                |
| c (sin θ)              |       0.15  |    0.10–0.25  | Larger angle → higher j and ΔP.                      |
| Cf (friction)          |       1.30  |      1.0–1.6  | Friction typically above plate fins.                 |
| mf (friction Re exp.)  |       0.25  |    0.20–0.35  | f ∝ Re^{-mf}                                         |
""")

        st.markdown("**Plate fins**")
        st.markdown("""
| Parameter              | Mid/default | Typical range | Notes                                |
|------------------------|------------:|--------------:|--------------------------------------|
| Cj                     |       0.27  |    0.20–0.32  | Lower than louvered.                 |
| mj                     |       0.50  |    0.45–0.60  | Stronger Re decay than louvered.     |
| a (pitch/Dh)           |      −0.20  |  −0.30–−0.10  | Tighter pitch raises j.              |
| b                      |       0.00  |            —  | No louvers.                          |
| c                      |       0.00  |            —  | No louvers.                          |
| Cf (friction)          |       1.00  |      0.8–1.2  | Friction typically lowest.           |
| mf (friction Re exp.)  |       0.23  |    0.20–0.30  |                                      |
""")

        st.markdown("**Corrugated fins**")
        st.markdown("""
| Parameter              | Mid/default | Typical range | Notes                                   |
|------------------------|------------:|--------------:|-----------------------------------------|
| Cj                     |       0.34  |    0.30–0.40  | Corrugated j between plate/louvered.    |
| mj                     |       0.45  |    0.40–0.55  |                                         |
| a (pitch/Dh)           |      −0.20  |  −0.30–−0.10  |                                         |
| b                      |       0.00  |            —  | No louvers.                             |
| c                      |       0.00  |            —  | No louvers.                             |
| Cf (friction)          |       1.15  |       1.0–1.3 | Friction a bit above plate fins.        |
| mf (friction Re exp.)  |       0.25  |    0.20–0.30  |                                         |
""")
except Exception:
    pass
