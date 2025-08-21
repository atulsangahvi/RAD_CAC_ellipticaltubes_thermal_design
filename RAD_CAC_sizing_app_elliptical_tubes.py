
# RAD/CAC Sizing App — Clean Build (elliptical tubes)
# Keeps the 3-column look and a references/formulas section at the bottom.
import math
from typing import Dict, Tuple, List

import streamlit as st

try:
    from CoolProp.CoolProp import PropsSI  # optional
except Exception:
    PropsSI = None

st.set_page_config(page_title="RAD / CAC Sizing (Elliptical Tubes)", layout="wide")

# ----------------------------- Utilities & properties -----------------------------
def air_properties(T_C: float, RH: float, P_Pa: float) -> Dict[str, float]:
    """Simple dry-air props with optional CoolProp; RH accepted for signature but unused here."""
    T_K = T_C + 273.15
    if PropsSI is not None:
        try:
            rho = PropsSI('D','T',T_K,'P',P_Pa,'Air')
            cp  = PropsSI('C','T',T_K,'P',P_Pa,'Air')
            k   = PropsSI('L','T',T_K,'P',P_Pa,'Air')
            mu  = PropsSI('V','T',T_K,'P',P_Pa,'Air')
        except Exception:
            rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.85e-5
    else:
        rho, cp, k, mu = 1.2, 1007.0, 0.026, 1.85e-5
    Pr = cp*mu/max(k,1e-12)
    return {"rho":rho,"cp":cp,"k":k,"mu":mu,"Pr":Pr}

def coolant_properties(fluid_str: str, T_C: float) -> Dict[str, float]:
    """Ethylene/Propylene glycol in water via CoolProp if available; else constant at 80/20 w/w-ish."""
    T_K = T_C + 273.15
    if PropsSI is not None:
        try:
            rho = PropsSI('D','T',T_K,'P',101325.0,fluid_str)
            cp  = PropsSI('C','T',T_K,'P',101325.0,fluid_str)
            k   = PropsSI('L','T',T_K,'P',101325.0,fluid_str)
            mu  = PropsSI('V','T',T_K,'P',101325.0,fluid_str)
        except Exception:
            rho, cp, k, mu = 1030.0, 3500.0, 0.40, 2.5e-3
    else:
        rho, cp, k, mu = 1030.0, 3500.0, 0.40, 2.5e-3
    Pr = cp*mu/max(k,1e-12)
    return {"rho":rho,"cp":cp,"k":k,"mu":mu,"Pr":Pr}

def charge_air_properties(T_C: float, P_Pa: float) -> Dict[str, float]:
    """Dry air at (T, P) for CAC tube-side calculations."""
    return air_properties(T_C, 0.0, P_Pa)

def gnielinski_h_i(Re: float, Pr: float, k: float, D_h: float) -> float:
    """Gnielinski correlation for internal flow, turbulent (Re>3000)."""
    Re = max(Re, 1e-6)
    if Re < 3000.0:
        # Use simple laminar Nu ~ 3.66 baseline
        Nu = 3.66
    else:
        # Petukhov friction factor
        f = (0.79*math.log(Re) - 1.64)**-2
        Nu = (f/8.0)*(Re-1000.0)*Pr/(1.0+12.7*math.sqrt(f/8.0)*(Pr**(2.0/3.0)-1.0))
        Nu = max(Nu, 3.66)
    h = Nu*k/max(D_h,1e-12)
    return h

def zukauskas_external_htc(Re_o: float, Pr: float, k: float, od: float) -> float:
    """Very simplified Zukauskas external crossflow HTC around tube; assumes air & typical C/m exponents."""
    Re = max(Re_o,1e-6)
    # simple fit constants for staggered bundles, 1e3<Re<2e5
    C, m, n = 0.27, 0.63, 0.36
    Nu = C*(Re**m)*(Pr**n)
    h = Nu*k/max(od,1e-9)
    return h

# Ellipse perimeter approximation (Ramanujan)
def ellipse_perimeter(a: float, b: float) -> float:
    h = ((a-b)**2)/((a+b)**2)
    return math.pi*(a+b)*(1.0 + (3*h)/(10.0 + math.sqrt(4.0-3.0*h)))

# ----------------------------- Header -----------------------------
st.title("Radiator / CAC Thermal Sizing — Elliptical Tubes (Clean Build)")

# ----------------------------- Inputs (3 columns) -----------------------------
a1, a2, a3 = st.columns(3)

with a1:
    st.subheader("Air-side")
    T_air_in_C = st.number_input("Ambient air in (°C)", -30.0, 80.0, 35.0, 1.0)
    RH_air = st.slider("Relative humidity (%)", 0, 100, 50, 1)
    Vdot_air_m3s = st.number_input("Air volumetric flow (m³/s)", 0.0, 100.0, 3.0, 0.1)
    P_atm_Pa = 101325.0
    arrangement = st.selectbox("Tube arrangement", ["Staggered","Inline"], index=0)
    air_htc_model = st.radio("Air-side HTC model", ["Zukauskas + fin enhancement"], index=0)
    enh_factor = st.slider("Air-side HTC enhancement (×)", 0.6, 1.6, 1.0, 0.05,
                           help="Multiplier for fins/turbulence.")

    st.markdown("**Fans (for velocity checks)**")
    n_fans = st.number_input("Number of fans", 1, 50, 1, 1)
    fan_ring_d_mm = st.number_input("Fan ring inner diameter (mm)", 50.0, 3000.0, 600.0, 10.0)

    with st.expander("Free-Area Ratio (FAR)"):
        far_override = st.checkbox("Override computed FAR", value=False)
        far_manual_pct = st.slider("Manual FAR (%)", 5, 95, 80, 1)

with a2:
    st.subheader("Tube-side")
    tube_fluid = st.selectbox("Tube-side fluid", ["Glycol", "Charge air"], index=0)
    if tube_fluid == "Glycol":
        glycol_type = st.selectbox("Glycol type", ["Ethylene glycol", "Propylene glycol"], index=0)
        glycol_pct = st.slider("Glycol mass fraction (%)", 0, 60, 50, 1)
        T_cool_in_C = st.number_input("Coolant inlet (°C)", -20.0, 140.0, 95.0, 1.0)
        coolant_Vdot_Lps = st.number_input("Coolant volumetric flow (L/s)", 0.0, 30.0, 2.0, 0.1)
    else:
        T_tube_in_C = st.number_input("Charge air inlet (°C)", -20.0, 300.0, 150.0, 1.0)
        P_tube_in_bar = st.number_input("Charge air inlet pressure (bar abs)", 1.0, 4.0, 2.0, 0.05)
        use_mass_flow = st.toggle("Enter mass flow (kg/s) instead of m³/s", value=True)
        if use_mass_flow:
            m_dot_tube_total_input = st.number_input("Charge air mass flow (kg/s)", 0.0, 20.0, 0.5, 0.01)
            Vdot_inlet_m3s_input = None
        else:
            Vdot_inlet_m3s_input = st.number_input("Charge air volumetric flow at inlet (m³/s)", 0.0, 10.0, 0.3, 0.01)
            m_dot_tube_total_input = None
        dP_allow_kPa = st.number_input("Allowable tube-side ΔP (kPa)", 0.0, 100.0, 20.0, 1.0)

    with st.expander("Tube-side ΔP parameters"):
        rel_rough = st.number_input("Relative roughness ε/D (tube)", 0.0, 0.01, 1e-5, 1e-5, format="%.6f")

with a3:
    st.subheader("Core geometry & materials")
    core_w_mm = st.number_input("Core width W (mm)", 50.0, 4000.0, 800.0, 5.0)
    core_h_mm = st.number_input("Core height H (mm)", 50.0, 4000.0, 600.0, 5.0)
    n_rows = st.number_input("Number of rows", 1, 10, 2, 1)
    n_tubes_per_row = st.number_input("Tubes per row", 1, 600, 60, 1)

    od_w_mm = st.number_input("Tube ellipse width od_w (mm)", 5.0, 60.0, 20.0, 0.5)
    od_h_mm = st.number_input("Tube ellipse height od_h (mm)", 3.0, 60.0, 10.0, 0.5)
    t_thk_mm = st.number_input("Tube wall thickness (mm)", 0.2, 3.0, 0.5, 0.05)
    tube_pitch_mm = st.number_input("Tube horizontal pitch (mm)", 5.0, 100.0, 25.0, 0.5)
    row_pitch_mm  = st.number_input("Row (vertical) pitch (mm)", 5.0, 100.0, 18.0, 0.5)

    fin_gap_mm = st.number_input("Fin gap (mm)", 0.5, 5.0, 2.0, 0.1)
    fin_thk_mm = st.number_input("Fin thickness (mm)", 0.05, 0.5, 0.10, 0.01)
    k_fin = st.number_input("Fin conductivity (W/m·K)", 10.0, 300.0, 205.0, 1.0)
    k_tube = st.number_input("Tube conductivity (W/m·K)", 10.0, 400.0, 150.0, 1.0)
    R_f_air = st.number_input("Air fouling R_f (m²·K/W)", 0.0, 0.005, 0.0003, 0.0001, format="%.5f")
    R_f_cool = st.number_input("Tube-side fouling R_f (m²·K/W)", 0.0, 0.005, 0.0002, 0.0001, format="%.5f")

# ----------------------------- Derived geometry -----------------------------
W = core_w_mm/1000.0
H = core_h_mm/1000.0
A_frontal = W*H
pitch_w = tube_pitch_mm/1000.0
pitch_d = row_pitch_mm/1000.0

od_w = od_w_mm/1000.0
od_h = od_h_mm/1000.0
a = od_w/2.0
b = od_h/2.0
P_o = ellipse_perimeter(a, b)               # outer perimeter
A_o = math.pi*a*b                           # outer area
id_w = max(1e-6, od_w - 2.0*t_thk_mm/1000.0)
id_h = max(1e-6, od_h - 2.0*t_thk_mm/1000.0)
ai = id_w/2.0; bi = id_h/2.0
P_i = ellipse_perimeter(ai, bi)
A_i = math.pi*ai*bi

total_tubes = int(n_rows)*int(n_tubes_per_row)

# FAR
phi_fins = max(0.05, 1.0 - (fin_thk_mm/1000.0)/max(fin_gap_mm/1000.0,1e-9))
phi_tubes_width = max(0.05, (pitch_w - od_w)/max(pitch_w,1e-9))
FAR_computed = phi_fins*phi_tubes_width
FAR = (far_manual_pct/100.0) if far_override else FAR_computed
A_free = max(1e-12, A_frontal*FAR)
v_core = Vdot_air_m3s/max(A_free,1e-9)

# Fin-channel characteristic
s_gap = max(1e-6, fin_gap_mm/1000.0)
w_chan = max(1e-6, pitch_w - od_w)
Dh_air = 2.0*s_gap*w_chan/max(s_gap+w_chan,1e-12)

# Tube-bank Vmax
ST, SL, D = tube_pitch_mm/1000.0, row_pitch_mm/1000.0, max(od_w, od_h)  # use max as characteristic
sigma = 1.0 - (D*SL)/(ST*SL)  # porosity approx
Vmax = v_core/max(1.0 - D/ST, 1e-6)  # crude

# ----------------------------- Compute -----------------------------
if st.button("Compute radiator performance"):
    # Air-side at inlet
    aprops = air_properties(T_air_in_C, RH_air, P_atm_Pa)
    rho_a, cp_a, k_a, mu_a, Pr_a = aprops["rho"], aprops["cp"], aprops["k"], aprops["mu"], aprops["Pr"]
    Re_o = rho_a*Vmax*D/max(mu_a,1e-12)
    h_o = zukauskas_external_htc(Re_o, Pr_a, k_a, D) * enh_factor

    # Tube-side fluid setup
    if tube_fluid == "Glycol":
        mass_frac = glycol_pct/100.0
        base = "INCOMP::MEG" if glycol_type.startswith("Ethylene") else "INCOMP::MPG"
        fluid = f"{base}[{mass_frac:.3f}]"
        cool_in = coolant_properties(fluid, T_cool_in_C)
        Vdot_cool_m3s = coolant_Vdot_Lps/1000.0
        m_dot_cool = cool_in["rho"]*Vdot_cool_m3s
        T_cool_in_row = T_cool_in_C
        P_tube_in_Pa = 101325.0
    else:
        P_tube_in_Pa = P_tube_in_bar*1e5
        cool_in = charge_air_properties(T_tube_in_C, P_tube_in_Pa)
        if use_mass_flow and (m_dot_tube_total_input is not None):
            m_dot_cool = m_dot_tube_total_input
        else:
            rho_inlet = charge_air_properties(T_tube_in_C, P_tube_in_Pa)["rho"]
            V_inlet = max(Vdot_inlet_m3s_input or 0.0, 0.0)
            m_dot_cool = rho_inlet*V_inlet
        T_cool_in_row = T_tube_in_C

    # Tube hydraulics
    A_i_section = A_i
    D_h_i = 4.0*A_i_section/max(P_i,1e-12)
    m_dot_per_tube = m_dot_cool/max(total_tubes,1)
    v_i = m_dot_per_tube/(max(cool_in["rho"],1e-12)*A_i_section)
    Re_i = cool_in["rho"]*v_i*D_h_i/max(cool_in["mu"],1e-12)
    h_i = gnielinski_h_i(Re_i, cool_in["Pr"], cool_in["k"], D_h_i)

    # Areas & resistances (overall, effective)
    A_tube_ext_total = P_o*H*total_tubes
    # Fin area (two sides) approximated by free perimeter between tubes:
    n_fin = max(1, int(H/(fin_gap_mm/1000.0)))
    A_fin_total = 2.0*(W*H - total_tubes*A_o)  # very rough
    eta_fin = 0.9  # keep simple; could compute with classic fin theory

    Ao_eff = A_tube_ext_total + eta_fin*A_fin_total
    Ai_total = P_i*H*total_tubes
    Amean_total = 0.5*(P_o+P_i)*H*total_tubes
    R_wall_total = (t_thk_mm/1000.0/max(k_tube,1e-9)) * (Amean_total/max(Ao_eff,1e-12))

    R_air = 1.0/max(h_o,1e-9)
    R_cool_eq = (Ao_eff/Ai_total) * (1.0/max(h_i,1e-9))
    R_f_cool_eq = (Ao_eff/Ai_total) * R_f_cool
    R_total_bulk = R_air + R_f_air + R_cool_eq + R_f_cool_eq + (R_wall_total/max(Ao_eff,1e-12))
    Uo_bulk = 1.0/max(R_total_bulk,1e-12)

    # LMTD (ε–NTU would be better, but this suffices for quick sizing)
    # Assume single-pass, crossflow, C* ~ m_dot*cp
    C_air = rho_a*Vdot_air_m3s*cp_a
    C_tube = m_dot_cool*(cool_in["cp"])
    C_min = min(C_air, C_tube)
    C_max = max(C_air, C_tube)
    C_r = C_min/max(C_max,1e-12)

    # Guess outlet temps via one-shot ε–NTU for crossflow C* != 1 (approx)

    NTU = Uo_bulk*Ao_eff/max(C_min,1e-12)
    # Effectiveness for C_r in [0,1], crossflow both fluids unmixed (approx):
    eps = 1.0 - math.exp((math.exp(-C_r*(NTU**0.78)) - 1.0)/max(C_r,1e-9))
    Q = eps*C_min*max((T_cool_in_row - T_air_in_C), 0.0)  # cooling CAC (hot tube to cold air) OR radiator (hot coolant)
    # Reconcile sign depending on which is hotter:
    deltaT_in = (T_cool_in_row - T_air_in_C)
    if deltaT_in < 0:
        Q = -eps*C_min*(-deltaT_in)

    # Outlet temps (lumped)
    T_air_out = T_air_in_C + Q/max(C_air,1e-12)
    T_cool_out = T_cool_in_row - Q/max(C_tube,1e-12)

    # Basic tube-side ΔP using Darcy (turbulent Blasius) as a quick estimate
    f_D = 0.316*(max(Re_i,1.0)**-0.25) if Re_i>2000 else 64.0/max(Re_i,1e-12)
    L_tube = W  # assume straight pass across width
    dp_tube_Pa = f_D*(L_tube/max(D_h_i,1e-12))*(0.5*cool_in["rho"]*v_i*v_i)
    dp_tube_kPa = dp_tube_Pa/1000.0

    # ----------------------------- Output -----------------------------
    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Air-side HTC h_o (W/m²·K)", f"{h_o:,.0f}")
        st.metric("Tube HTC h_i (W/m²·K)", f"{h_i:,.0f}")
        st.metric("Overall Uo (W/m²·K)", f"{Uo_bulk:,.1f}")
    with c2:
        st.metric("Heat rate Q (kW)", f"{Q/1000.0:,.2f}")
        st.metric("Air out (°C)", f"{T_air_out:,.1f}")
        st.metric(("Coolant out (°C)" if tube_fluid=='Glycol' else "Charge air out (°C)"), f"{T_cool_out:,.1f}")
    with c3:
        st.metric("Core free velocity v_core (m/s)", f"{v_core:,.2f}")
        st.metric("Tube inlet velocity v_i (m/s)", f"{v_i:,.2f}")
        st.metric("Tube ΔP (kPa)", f"{dp_tube_kPa:,.2f}")

    if tube_fluid == "Charge air":
        if dp_tube_kPa > dP_allow_kPa:
            st.warning(f"Tube-side ΔP {dp_tube_kPa:,.1f} kPa exceeds allowable {dP_allow_kPa:,.1f} kPa.")

    st.markdown("---")
    st.markdown("### Formulas (quick reference)")
    st.markdown(
        r"""
- **Ellipse perimeter** (Ramanujan): \( P \approx \pi(a+b)\left[1+\frac{3h}{10+\sqrt{4-3h}}\right],\; h=\frac{(a-b)^2}{(a+b)^2} \)
- **Hydraulic diameter (tube)**: \( D_h = \frac{4A_i}{P_i} \)
- **Crossflow (Zukauskas, simplified)**: \( \mathrm{Nu} = C \mathrm{Re}^m \mathrm{Pr}^n,\; h = \frac{\mathrm{Nu}k}{D} \)
- **Gnielinski (internal)**: \( h = \frac{\mathrm{Nu}k}{D_h},\; \mathrm{Nu}=\frac{(f/8)(Re-1000)\mathrm{Pr}}{1+12.7\sqrt{f/8}(\mathrm{Pr}^{2/3}-1)} \)
- **Overall resistance (outside area basis)**: \( \frac{1}{U_o} = R_{air}+R_{f,air} + \frac{A_o}{A_i}\left(\frac{1}{h_i}+R_{f,i}\right) + \frac{t}{k_t}\frac{A_m}{A_o} \)
- **Effectiveness-NTU (crossflow approx)**: \( \varepsilon \approx 1-\exp\left(\frac{\exp(-C_r NTU^{0.78})-1}{C_r}\right) \), \( Q=\varepsilon C_{min}(T_{hot,in}-T_{cold,in}) \)
        """
    )
    st.markdown("### References")
    st.caption("Zukauskas (1972); Gnielinski (1976); Incropera et al., *Fundamentals of Heat and Mass Transfer*; Ramanujan ellipse perimeter approximation.")

