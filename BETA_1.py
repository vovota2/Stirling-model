import matplotlib
matplotlib.use('Agg') # Oprava stability renderování na webových serverech

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from scipy.interpolate import make_interp_spline
import tempfile
import os
import time

# --- FIX PRO NUMPY VERZE ---
if hasattr(np, 'trapezoid'):
    integrate = np.trapezoid
else:
    integrate = np.trapz

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Stirling Beta Model", layout="wide")

# CSS úpravy
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem; 
        padding-bottom: 2rem;
    }
    div[data-testid="stExpander"] div[role="button"] p {font-size: 1.05rem; font-weight: 600;}
    
    /* Zamezení blikání při real-time reloadu */
    .element-container {
        transition: none !important;
    }
    
    /* OKNA PRO VÝSLEDKY */
    .result-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .box-title {
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 5px;
    }

    /* STICKY TABS FIX */
    div[data-baseweb="tab-list"] {
        position: sticky;
        top: 3rem;
        z-index: 9999;
        background-color: white;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        box-shadow: 0 4px 6px -6px #222;
    }
    div[data-testid="stTabs"] {
        background-color: transparent;
    }

    /* CSS PRO PLOVOUCÍ TLAČÍTKO "PŘEPOČÍTAT" integrované do jednoho bloku */
    div.element-container:has(.recalc-anchor) {
        display: none;
    }
    div.element-container:has(.recalc-anchor) + div.element-container {
        position: fixed !important;
        bottom: 30px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 99999 !important;
        width: 320px !important;
    }
    div.element-container:has(.recalc-anchor) + div.element-container button {
        width: 100% !important;
        height: 60px !important; /* Větší výška pro oba řádky */
        border-radius: 15px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3) !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 0 !important;
        line-height: 1.2;
    }
    /* Pseudo-element tvořící druhý řádek uvnitř tlačítka */
    div.element-container:has(.recalc-anchor) + div.element-container button::after {
        content: 'pro nově zvolené parametry';
        font-size: 0.75rem;
        font-weight: normal;
        opacity: 0.8;
        display: block;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# POMOCNÁ FUNKCE PRO VSTUPY
# =============================================================================
def smart_input(label, min_val, slider_max, default_val, step, key_id):
    if f"{key_id}_num" not in st.session_state:
        st.session_state[f"{key_id}_num"] = default_val
    if f"{key_id}_slide" not in st.session_state:
        st.session_state[f"{key_id}_slide"] = min(default_val, slider_max)

    def update_from_slider():
        st.session_state[f"{key_id}_num"] = st.session_state[f"{key_id}_slide"]
        
    def update_from_num():
        val = st.session_state[f"{key_id}_num"]
        st.session_state[f"{key_id}_slide"] = min(val, slider_max)

    st.markdown(f"**{label}**")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.slider("S", float(min_val), float(slider_max), step=float(step), key=f"{key_id}_slide", on_change=update_from_slider, label_visibility="collapsed")
    with c2:
        st.number_input("I", min_value=float(min_val), max_value=None, step=float(step), key=f"{key_id}_num", on_change=update_from_num, label_visibility="collapsed")
    
    return st.session_state[f"{key_id}_num"]

# =============================================================================
# FUNKCE PRO VYKRESLENÍ ANIMOVANÉHO SCHÉMATU MOTORU
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_engine_animation(alpha_deg):
    fig, ax = plt.subplots(figsize=(3.5, 4.0))

    c_disp = '#4ded30' 
    c_wp = '#4a4a4a'   
    c_line = 'black'
    lw = 1.5
    scale = 0.85
    pipe_w = 10 * scale 
    pad_val = 3 

    cyl_w = 50 * scale 
    cyl_h = 175 * scale 
    cyl_x0 = 10
    cyl_x1 = cyl_x0 + cyl_w
    cyl_y0 = 10 
    cyl_y1 = cyl_y0 + cyl_h
    cyl_xc = cyl_x0 + cyl_w / 2

    reg_w = 22 * scale 
    reg_h = 45 * scale
    reg_xc = cyl_x1 + 50 * scale

    pipe_top_peak_y = cyl_y1 + 15 * scale 
    pipe_low_cyl_y = cyl_y0 + 55 * scale  
    
    reg_y_center = (pipe_top_peak_y + pipe_low_cyl_y) / 2
    reg_y0 = reg_y_center - reg_h / 2
    reg_y1 = reg_y_center + reg_h / 2

    disp_amp = 18 * scale
    disp_h = 55 * scale
    disp_base_y = 100 * scale
    wp_amp = 12 * scale
    wp_base_y = 40 * scale
    
    ax.plot([cyl_x0, cyl_x1], [wp_base_y + wp_amp, wp_base_y + wp_amp], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.plot([cyl_x0, cyl_x1], [wp_base_y - wp_amp, wp_base_y - wp_amp], color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.plot([cyl_x0, cyl_x1], [disp_base_y + disp_h + disp_amp, disp_base_y + disp_h + disp_amp], color='#4ded30', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.plot([cyl_x0, cyl_x1], [disp_base_y - disp_amp, disp_base_y - disp_amp], color='#4ded30', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.plot([cyl_x0, cyl_x0], [cyl_y0, cyl_y1], color=c_line, lw=lw)
    ax.plot([cyl_x1, cyl_x1], [cyl_y0, pipe_low_cyl_y - pipe_w/2], color=c_line, lw=lw)
    ax.plot([cyl_x1, cyl_x1], [pipe_low_cyl_y + pipe_w/2, cyl_y1], color=c_line, lw=lw)
    ax.plot([cyl_x0, cyl_xc - pipe_w/2], [cyl_y1, cyl_y1], color=c_line, lw=lw)
    ax.plot([cyl_xc + pipe_w/2, cyl_x1], [cyl_y1, cyl_y1], color=c_line, lw=lw)

    ax.plot([cyl_xc - pipe_w/2, cyl_xc - pipe_w/2, reg_xc + pipe_w/2, reg_xc + pipe_w/2],
            [cyl_y1, pipe_top_peak_y + pipe_w/2, pipe_top_peak_y + pipe_w/2, reg_y1 + pad_val], color=c_line, lw=lw)
    ax.plot([cyl_xc + pipe_w/2, cyl_xc + pipe_w/2, reg_xc - pipe_w/2, reg_xc - pipe_w/2],
            [cyl_y1, pipe_top_peak_y - pipe_w/2, pipe_top_peak_y - pipe_w/2, reg_y1 + pad_val], color=c_line, lw=lw)

    ax.plot([cyl_x1, reg_xc - pipe_w/2, reg_xc - pipe_w/2],
            [pipe_low_cyl_y + pipe_w/2, pipe_low_cyl_y + pipe_w/2, reg_y0 - pad_val], color=c_line, lw=lw)
    ax.plot([cyl_x1, reg_xc + pipe_w/2, reg_xc + pipe_w/2],
            [pipe_low_cyl_y - pipe_w/2, pipe_low_cyl_y - pipe_w/2, reg_y0 - pad_val], color=c_line, lw=lw)

    regen = patches.FancyBboxPatch((reg_xc - reg_w/2, reg_y0), reg_w, reg_h,
                                   boxstyle=f"round,pad={pad_val}", edgecolor=c_line, facecolor='white',
                                   hatch='xxxx', lw=lw)
    ax.add_patch(regen)

    c_heat = '#ff3333'
    hy0 = reg_y1 + 8 * scale             
    hy1 = hy0 + 26.5 * scale             
    hx_inner_l = reg_xc - pipe_w/2
    hx_outer_l = hx_inner_l - 10 * scale
    hx_inner_r = reg_xc + pipe_w/2
    hx_outer_r = hx_inner_r + 10 * scale
    h_dx = 2.5 * scale
    h_dy = 4 * scale

    h_left_pts = [
        (hx_inner_l, hy0), (hx_outer_l + h_dx, hy0), (hx_outer_l, hy0 + h_dy),
        (hx_outer_l, hy1 - h_dy), (hx_outer_l + h_dx, hy1), (hx_inner_l, hy1)
    ]
    ax.add_patch(patches.Polygon(h_left_pts, closed=True, facecolor=c_heat, edgecolor=c_line, lw=lw))

    h_right_pts = [
        (hx_inner_r, hy0), (hx_outer_r - h_dx, hy0), (hx_outer_r, hy0 + h_dy),
        (hx_outer_r, hy1 - h_dy), (hx_outer_r - h_dx, hy1), (hx_inner_r, hy1)
    ]
    ax.add_patch(patches.Polygon(h_right_pts, closed=True, facecolor=c_heat, edgecolor=c_line, lw=lw))

    c_cool = '#0033cc'
    cy_top = reg_y0 - 8 * scale  
    fin_h = 2.0 * scale          
    gap_h = 5.5 * scale          
    cx_inner_l = reg_xc - pipe_w/2
    cx_outer_l = cx_inner_l - 14.5 * scale 
    cx_gap_l = cx_inner_l - 5 * scale
    cx_inner_r = reg_xc + pipe_w/2
    cx_outer_r = cx_inner_r + 14.5 * scale 
    cx_gap_r = cx_inner_r + 5 * scale

    c_left_pts = [(cx_inner_l, cy_top)]
    cy = cy_top
    for i in range(4):
        c_left_pts.extend([(cx_outer_l, cy), (cx_outer_l, cy - fin_h)])
        cy -= fin_h
        if i < 3:
            c_left_pts.extend([(cx_gap_l, cy), (cx_gap_l, cy - gap_h)])
            cy -= gap_h
    c_left_pts.append((cx_inner_l, cy))
    ax.add_patch(patches.Polygon(c_left_pts, closed=True, facecolor=c_cool, edgecolor=c_line, lw=lw))

    c_right_pts = [(cx_inner_r, cy_top)]
    cy = cy_top
    for i in range(4):
        c_right_pts.extend([(cx_outer_r, cy), (cx_outer_r, cy - fin_h)])
        cy -= fin_h
        if i < 3:
            c_right_pts.extend([(cx_gap_r, cy), (cx_gap_r, cy - gap_h)])
            cy -= gap_h
    c_right_pts.append((cx_inner_r, cy))
    ax.add_patch(patches.Polygon(c_right_pts, closed=True, facecolor=c_cool, edgecolor=c_line, lw=lw))

    wp_poly = patches.Polygon(np.zeros((9, 2)), closed=True, edgecolor=c_line, facecolor=c_wp, lw=lw)
    ax.add_patch(wp_poly)
    
    disp = patches.Rectangle((cyl_x0 + 1, 0), cyl_w - 2, disp_h, edgecolor=c_line, facecolor=c_disp, lw=lw)
    ax.add_patch(disp)

    ax.set_xlim(0, 190)
    ax.set_ylim(0, 210)
    ax.axis('off')
    fig.tight_layout(pad=0.1)

    def animate(frame):
        phi = np.deg2rad(frame)
        alpha = np.deg2rad(alpha_deg)

        wp_top = wp_base_y - wp_amp * np.cos(phi - alpha) 
        
        wp_h = 15 * scale
        wp_body_bottom = wp_top - wp_h
        rod_side = 12 * scale
        rod_bottom = wp_body_bottom - rod_side

        wp_points = np.array([
            [cyl_x0 + 1, wp_top], [cyl_x1 - 1, wp_top],
            [cyl_x1 - 1, wp_body_bottom], 
            [cyl_xc + rod_side/2, wp_body_bottom], [cyl_xc + rod_side/2, rod_bottom], 
            [cyl_xc - rod_side/2, rod_bottom], [cyl_xc - rod_side/2, wp_body_bottom], 
            [cyl_x0 + 1, wp_body_bottom], [cyl_x0 + 1, wp_top]
        ])
        wp_poly.set_xy(wp_points)

        disp_bottom = disp_base_y - disp_amp * np.cos(phi)
        disp.set_y(disp_bottom)

        return wp_poly, disp

    ani = animation.FuncAnimation(fig, animate, frames=np.linspace(0, 360, 100, endpoint=False), blit=False)
    
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
        tmp_path = tmpfile.name

    ani.save(tmp_path, writer='pillow', fps=50) 
    plt.close(fig)

    with open(tmp_path, "rb") as f:
        gif_bytes = f.read()

    os.remove(tmp_path)
    return gif_bytes

# =============================================================================
# 1. BOČNÍ PANEL - VSTUPY
# =============================================================================
st.sidebar.header("🎛️ Nastavení simulace")

with st.sidebar.expander("1. Provozní parametry", expanded=True):
    f = smart_input(r"Frekvence $f$ (Hz)", 1, 200, 50, 1, "freq")
    p_st_MPa = smart_input(r"Střední tlak $p_{stř}$ (MPa)", 0.1, 50.0, 15.0, 0.1, "pres")
    TT = smart_input(r"Teplota $T_T$ (K)", 300, 1500, 973, 10, "temp_hot")
    TS = smart_input(r"Teplota $T_S$ (K)", 100, 800, 420, 10, "temp_cold")
    alpha_deg = smart_input(r"Fázový posun $\alpha$ (°)", 0, 180, 90, 1, "alpha")
    n_poly = smart_input(r"Polytropický exponent $n$ (-)", 1.0, 1.67, 1.4, 0.01, "n_poly")

with st.sidebar.expander("2. Geometrie", expanded=False):
    VTZ_ccm = smart_input(r"Zdvihový objem $V_{TZ}$ (cm$^3$)", 10.0, 1000.0, 118.58, 0.01, "vol_main")
    VTZ = VTZ_ccm * 1e-6 
    
    st.markdown("---")
    geom_mode = st.radio("Způsob zadání objemů:", ["Poměry (X)", "Objemy (cm³)"], horizontal=True)
    
    if geom_mode == "Poměry (X)":
        XSZ = smart_input(r"Poměr $X_{SZ} (= V_{SZ} / V_{TZ})$", 0.1, 5.0, 1.5, 0.1, "xsz")
        XR  = smart_input(r"Poměr $X_R (= V_R / V_{TZ})$", 0.1, 10.0, 2.0, 0.1, "xr")
        XTM = smart_input(r"Poměr $X_{TM}$ (Mrtvý teplý)", 0.1, 5.0, 1.2, 0.1, "xtm")
        XSM = smart_input(r"Poměr $X_{SM}$ (Mrtvý studený)", 0.1, 5.0, 2.5, 0.1, "xsm")
    else:
        VSZ_ccm = smart_input(r"Objem $V_{SZ}$ (cm³)", 1.0, 1000.0, 177.87, 1.0, "vsz_ccm")
        VR_ccm  = smart_input(r"Objem $V_R$ (cm³)", 1.0, 1000.0, 237.16, 1.0, "vr_ccm")
        VTM_ccm = smart_input(r"Objem $V_{TM}$ (cm³)", 1.0, 1000.0, 142.30, 1.0, "vtm_ccm")
        VSM_ccm = smart_input(r"Objem $V_{SM}$ (cm³)", 1.0, 1000.0, 296.45, 1.0, "vsm_ccm")
        XSZ = VSZ_ccm / VTZ_ccm
        XR = VR_ccm / VTZ_ccm
        XTM = VTM_ccm / VTZ_ccm
        XSM = VSM_ccm / VTZ_ccm
        
    st.markdown("---")
    vp_percent = smart_input(r"Objem překryvu zdvihů $V_P$ (% ideálu)", 0, 100, 0, 1, "vp_perc")

with st.sidebar.expander("3. Pracovní látka", expanded=False):
    plyn = st.radio("Zvolte médium", ["Helium", "Vodík", "Vzduch"])
    
    if plyn == "Helium":
        r_val = 2078.5
        kappa_val = 1.667
    elif plyn == "Vodík":
        r_val = 4124.0
        kappa_val = 1.405
    else: # Vzduch
        r_val = 287.0
        kappa_val = 1.400
    
    st.info(f"Parametry pro **{plyn}**: $r={r_val}$, $\kappa={kappa_val}$")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Restartovat nastavení", type="secondary"):
    st.session_state.clear()
    st.rerun()

# =============================================================================
# ŘÍZENÍ STAVU A VÝPOČET
# =============================================================================
calc_params = {
    'f': f, 'p_st_MPa': p_st_MPa, 'TT': TT, 'TS': TS, 'alpha_deg': alpha_deg,
    'n_poly': n_poly, 'VTZ_ccm': VTZ_ccm, 'XSZ': XSZ, 'XR': XR, 'XTM': XTM,
    'XSM': XSM, 'vp_percent': vp_percent, 'plyn': plyn, 'r': r_val, 'kappa': kappa_val
}

if 'last_params' not in st.session_state:
    st.session_state.last_params = calc_params.copy()
    st.session_state.show_loader = True

params_changed = calc_params != st.session_state.last_params

loader_placeholder = st.empty()

if st.session_state.get('show_loader', False) and not params_changed:
    with loader_placeholder.container():
        st.markdown("""<div class="loader-container"><div class="loader-ring"><div></div><div></div><div></div><div></div></div><p class="loader-text">Provádím termodynamický výpočet...</p></div>""", unsafe_allow_html=True)
    time.sleep(0.6) 
    loader_placeholder.empty()
    st.session_state.show_loader = False

# =============================================================================
# FUNKCE VÝPOČTU JÁDRA
# =============================================================================
def vypocet_modelu(params):
    alpha = np.deg2rad(params['alpha_deg'])
    TT = params['TT']
    TS = params['TS']
    tau = TS / TT
    p_st_pa = params['p_st_MPa'] * 1e6
    f = params['f']
    n_poly = params['n_poly']
    r = params['r']
    kappa = params['kappa']

    VTZ = params['VTZ_ccm'] * 1e-6
    XSZ = params['XSZ']
    XTM = params['XTM']
    XSM = params['XSM']
    XR = params['XR']
    vp_percent = params['vp_percent']

    VSZ = VTZ * XSZ; VTM = VTZ * XTM; VSM = VTZ * XSM; VR  = VTZ * XR

    term_sq = (VTZ**2 + VSZ**2)/4 - (VTZ * VSZ / 2) * np.cos(alpha)
    if term_sq < 0: term_sq = 0
    VP_ideal = (VTZ + VSZ)/2 - np.sqrt(term_sq)
    VP = VP_ideal * (vp_percent / 100.0)
    XP = VP / VTZ

    phi = np.linspace(0, 2*np.pi, 360)
    phi_deg = np.rad2deg(phi)

    VT = (VTZ / 2) * (1 - np.cos(phi)) + VTM
    term_disp = (VTZ / 2) * (1 + np.cos(phi))
    term_work = (VSZ / 2) * (1 - np.cos(phi - alpha))
    VS = term_disp + term_work + VSM - VP
    V = VR + VT + VS

    dVT_dphi = np.gradient(VT, phi)
    dVS_dphi = np.gradient(VS, phi)

    num_beta = XSZ * np.sin(alpha)
    den_beta = tau + XSZ * np.cos(alpha) - 1
    beta_angle = np.arctan2(num_beta, den_beta)

    term_cold = (1/tau) * (1 + XSZ + 2*XSM - 2*XP)
    term_reg  = (2 * XR * n_poly * np.log(tau)) / (tau - 1)
    A = 1 + 2*XTM + term_cold + term_reg
    B = np.sqrt((1 + (1/tau)*(XSZ * np.cos(alpha) - 1))**2 + ((1/tau) * XSZ * np.sin(alpha))**2)

    P_shape = (A - B * np.cos(phi - beta_angle))**(-n_poly)
    p_real = (p_st_pa / np.mean(P_shape)) * P_shape 
    p_mean_real = np.mean(p_real)

    W_cyklu = abs(integrate(p_real, V)) 
    Power_ind = W_cyklu * f
    Q_in = integrate(p_real, VT) 
    Q_out = integrate(p_real, VS)
    eta = (W_cyklu / Q_in) * 100
    pressure_ratio = np.max(p_real) / np.min(p_real)

    p0 = p_real[0]
    const_I = kappa / (kappa - 1)
    dI_total = const_I * (p_real * V - p0 * V[0])
    dQ_R_curve = dI_total - (VR * (p_real - p0))
    Q_reg_val = np.max(dQ_R_curve) - np.min(dQ_R_curve)
    ratio_Qreg = Q_reg_val / Q_in

    exp_term = (n_poly - 1) / n_poly
    T_gas_T = TT * (p_real / p_mean_real)**exp_term
    T_gas_S = TS * (p_real / p_mean_real)**exp_term
    T_reg_mean = (TT - TS) / np.log(TT/TS)

    m_inst = (p_real / r) * ( (VT / T_gas_T) + (VS / T_gas_S) + (VR / T_reg_mean) )
    mass_total_g = np.mean(m_inst) * 1000
    mass_deviation = (np.max(m_inst) - np.min(m_inst)) / np.mean(m_inst) * 100

    m_T_g = (p_real * VT / (r * T_gas_T)) * 1000
    m_S_g = (p_real * VS / (r * T_gas_S)) * 1000
    m_R_g = (p_real * VR / (r * T_reg_mean)) * 1000
    m_total_no_reg = m_T_g + m_S_g

    x_reg_vals = np.linspace(1.01, 2.99, 40) 
    xi = (x_reg_vals - 1.01) / (2.99 - 1.01)
    shape_reg = 3*xi**2 - 2*xi**3
    T_reg_profile = TT - (TT - TS) * shape_reg
    x_hot_vals = np.linspace(0, 0.99, 10)
    x_cold_vals = np.linspace(3.01, 4, 10)
    x_total = np.concatenate([x_hot_vals, x_reg_vals, x_cold_vals])
    phi_grid, x_grid = np.meshgrid(phi_deg, x_total)
    T_surface = np.zeros_like(x_grid)
    for i in range(len(phi)):
        row_hot = T_gas_T[i] * np.ones_like(x_hot_vals)
        row_reg = T_reg_profile
        row_cold = T_gas_S[i] * np.ones_like(x_cold_vals)
        T_surface[:, i] = np.concatenate([row_hot, row_reg, row_cold])

    return locals()

def solve_cycle_sweep(params):
    alpha = np.deg2rad(params['alpha_deg'])
    TT = params['TT']
    TS = params['TS']
    tau = TS / TT
    p_st_pa = params['p_st_MPa'] * 1e6
    f = params['f']
    n_poly = params['n_poly']
    r = params['r']
    kappa = params['kappa']

    VTZ = params['VTZ_ccm'] * 1e-6
    XSZ = params['XSZ']
    XTM = params['XTM']
    XSM = params['XSM']
    XR = params['XR']
    vp_percent = params['vp_percent']

    VSZ = VTZ * XSZ; VTM = VTZ * XTM; VSM = VTZ * XSM; VR  = VTZ * XR

    term_sq = (VTZ**2 + VSZ**2)/4 - (VTZ * VSZ / 2) * np.cos(alpha)
    if term_sq < 0: term_sq = 0
    VP_ideal = (VTZ + VSZ)/2 - np.sqrt(term_sq)
    VP = VP_ideal * (vp_percent / 100.0)
    XP = VP / VTZ

    phi = np.linspace(0, 2*np.pi, 360)

    VT = (VTZ / 2) * (1 - np.cos(phi)) + VTM
    term_disp = (VTZ / 2) * (1 + np.cos(phi))
    term_work = (VSZ / 2) * (1 - np.cos(phi - alpha))
    VS = term_disp + term_work + VSM - VP
    V = VR + VT + VS

    num_beta = XSZ * np.sin(alpha)
    den_beta = tau + XSZ * np.cos(alpha) - 1
    beta_angle = np.arctan2(num_beta, den_beta)

    term_cold = (1/tau) * (1 + XSZ + 2*XSM - 2*XP)
    term_reg  = (2 * XR * n_poly * np.log(tau)) / (tau - 1)
    A = 1 + 2*XTM + term_cold + term_reg
    B = np.sqrt((1 + (1/tau)*(XSZ * np.cos(alpha) - 1))**2 + ((1/tau) * XSZ * np.sin(alpha))**2)

    P_shape = (A - B * np.cos(phi - beta_angle))**(-n_poly)
    p_real = (p_st_pa / np.mean(P_shape)) * P_shape 

    W_cyklu = abs(integrate(p_real, V)) 
    Power_ind = W_cyklu * f
    Q_in = integrate(p_real, VT) 
    Q_out = integrate(p_real, VS)
    eta = (W_cyklu / Q_in) * 100
    
    p_max = np.max(p_real)
    p_min = np.min(p_real)
    pressure_ratio = p_max / p_min

    p0 = p_real[0]
    const_I = kappa / (kappa - 1)
    dI_total = const_I * (p_real * V - p0 * V[0])
    dQ_R_curve = dI_total - (VR * (p_real - p0))
    Q_reg_val = np.max(dQ_R_curve) - np.min(dQ_R_curve)
    ratio_Qreg = Q_reg_val / Q_in

    p_mean_real = np.mean(p_real)
    exp_term = (n_poly - 1) / n_poly
    T_gas_T = TT * (p_real / p_mean_real)**exp_term
    T_gas_S = TS * (p_real / p_mean_real)**exp_term
    T_reg_mean = (TT - TS) / np.log(TT/TS)

    m_inst = (p_real / r) * ( (VT / T_gas_T) + (VS / T_gas_S) + (VR / T_reg_mean) )
    mass_total_g = np.mean(m_inst) * 1000

    return {
        'P': Power_ind,
        'eta': eta,
        'psi': pressure_ratio,
        'p_max': p_max / 1e6, 
        'p_min': p_min / 1e6, 
        'Q_reg': Q_reg_val,
        'Q_ratio': ratio_Qreg,
        'W': W_cyklu,
        'Q_in': Q_in,
        'Q_out': abs(Q_out),
        'm_celk': mass_total_g
    }

# =============================================================================
# DATA A FUNKCE PRO ZÁLOŽKU BEALEOVA ČÍSLA
# =============================================================================
def get_smooth_curve(x, y, x_new):
    spline = make_interp_spline(x, y, k=2) 
    return spline(x_new)

# Funkce pro výpočet Bn s asymptotickou extrapolací (křivka se zplošťuje)
def get_bn_val(T_act, curve_y, curve_x):
    if T_act > 1200:
        slope = (curve_y[-1] - curve_y[-2]) / (curve_x[-1] - curve_x[-2])
        L = 300.0
        return curve_y[-1] + slope * L * (1.0 - np.exp(-(T_act - 1200.0) / L))
    elif T_act < 600:
        slope = (curve_y[1] - curve_y[0]) / (curve_x[1] - curve_x[0])
        L = 300.0
        return max(0.001, curve_y[0] - slope * L * (1.0 - np.exp(-(600.0 - T_act) / L)))
    else:
        return np.interp(T_act, curve_x, curve_y)

x_top = np.array([600, 800, 1000, 1200])
y_top = np.array([0.1, 0.17, 0.22, 0.26])
x_mid = np.array([600, 800, 1000, 1200])
y_mid = np.array([0.05, 0.113, 0.165, 0.2])
x_bot = np.array([600, 800, 1000, 1200])
y_bot = np.array([0.025, 0.057, 0.082, 0.1])
x_fsps = np.array([600, 700, 800, 900, 1000, 1100, 1200])
y_fsps = np.array([0.015, 0.026, 0.036, 0.045, 0.053, 0.059, 0.064])

T_range_full = np.linspace(600, 1200, 100)

curve_top = get_smooth_curve(x_top, y_top, T_range_full)
curve_mid = get_smooth_curve(x_mid, y_mid, T_range_full)
curve_bot = get_smooth_curve(x_bot, y_bot, T_range_full)
curve_fsps = get_smooth_curve(x_fsps, y_fsps, T_range_full)


# Načtení výsledků na základě POTVRZENÝCH parametrů
lp = st.session_state.last_params
res = vypocet_modelu(lp)
animated_gif = generate_engine_animation(lp['alpha_deg'])

# =============================================================================
# 4. ZOBRAZENÍ VÝSLEDKŮ 
# =============================================================================
layout_style = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(color='black', family="Arial"),
    xaxis=dict(showgrid=True, gridcolor='#d9d9d9', gridwidth=1, zeroline=False, showline=False, mirror=False),
    yaxis=dict(showgrid=True, gridcolor='#d9d9d9', gridwidth=1, zeroline=False, showline=False, mirror=False),
    margin=dict(l=60, r=40, t=70, b=60),
    legend=dict(bgcolor='rgba(255,255,255,0.8)', bordercolor='lightgray', borderwidth=1),
)

col_left, col_right = st.columns([3.5, 1.5])

with col_left:
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-bottom: 30px;">
            <h2 style="color: #2c3e50; font-size: 2.4rem; font-weight: 700; margin-bottom: 0px; text-align: center;">
                Model oběhu Stirlingova motoru
            </h2>
            <h4 style="color: #7f8c8d; font-size: 1.1rem; font-weight: 400; margin-top: 5px; text-align: center;">
                s polytropickými změnami na teplé a studené straně
            </h4>
            <div style="height: 3px; width: 60px; background-color: #FF4B4B; margin: 10px auto 0 auto; border-radius: 2px;"></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin: 0px; margin-bottom: 15px; color: #2c3e50; text-align: left;'>📊 Hlavní parametry cyklu</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Výkon P", f"{res['Power_ind']/1000:.2f} kW")
    c2.metric("Účinnost \u03b7", f"{res['eta']:.1f} %")
    c3.metric(r"Hmotnost $m_{celk}$", f"{res['mass_total_g']:.3f} g")
    c4.metric("Tlakový poměr ψ", f"{res['pressure_ratio']:.2f}")

with col_right:
    st.image(animated_gif, use_container_width=True)

# Plovoucí tlačítko Přepočítat model integrované dohromady s dodatečným textem
warn_container = st.container()
with warn_container:
    if params_changed:
        st.markdown('<div class="recalc-anchor"></div>', unsafe_allow_html=True)
        if st.button("⚙️ Přepočítat model", type="primary", use_container_width=True):
            st.session_state.last_params = calc_params.copy()
            st.session_state.show_loader = True
            st.session_state.force_auto_curve = True
            
            if 'selected_curve_idx' in st.session_state:
                del st.session_state['selected_curve_idx']
            if 'curve_choice_radio' in st.session_state:
                del st.session_state['curve_choice_radio']
                
            st.rerun()

st.markdown("<hr style='margin: 5px 0 15px 0;'>", unsafe_allow_html=True)

df_export = pd.DataFrame({"Uhel otoceni [deg]": np.round(res['phi_deg'], 1),"Tlak [MPa]": np.round(res['p_real'] / 1e6, 4),"Objem celkovy [cm3]": np.round(res['V'] * 1e6, 3),"Objem teply [cm3]": np.round(res['VT'] * 1e6, 3),"Objem studeny [cm3]": np.round(res['VS'] * 1e6, 3),"Teplota plyn T [K]": np.round(res['T_gas_T'], 2),"Teplota plyn S [K]": np.round(res['T_gas_S'], 2)})
csv_data = df_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')

# ZÁLOŽKY 
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋 Detailní výsledky", "📊 Tlak a objem", "🌡️ Teplotní průběhy", "⚡ Energetická bilance", "⚖️ Hmotnost média", "📈 Citlivostní analýza", "🎯 Odhad výkonu (Bn)"])

with tab1:
    c_head, c_down = st.columns([3, 1])
    with c_head:
        st.markdown("<h3 style='margin: 0px; padding-top: 0.2rem;'>📋 Detailní výsledky simulace</h3>", unsafe_allow_html=True)
    with c_down:
        st.download_button(label="📥 Stáhnout data výsledků (CSV)", data=csv_data, file_name='stirling_simulation_data.csv', mime='text/csv', type="secondary", use_container_width=True)
    
    st.markdown("<div style='height: 0.3rem;'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""<div class="result-box" style="height: 320px;"><div class="box-title">Energie a Výkon</div><ul><li>Indikovaná práce W: <b>{res['W_cyklu']:.2f} J</b></li><li>Teplo přivedené Q<sub>in</sub>: <b>{res['Q_in']:.2f} J</b></li><li>Teplo odvedené Q<sub>out</sub>: <b>{abs(res['Q_out']):.2f} J</b></li><li>Regenerované teplo Q<sub>R</sub>: <b>{res['Q_reg_val']:.2f} J</b></li><li>Poměr Q<sub>R</sub> / Q<sub>in</sub>: <b>{res['ratio_Qreg']:.2f} [-]</b></li><li>Indikovaný výkon P: <b>{res['Power_ind']/1000:.2f} kW</b></li><li>Účinnost cyklu η: <b>{res['eta']:.2f} %</b></li></ul></div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""<div class="result-box" style="height: 320px;"><div class="box-title">Teploty plynu</div><ul><li>Teplá strana (T<sub>Ts</sub>):<ul><li>Max: <b>{np.max(res['T_gas_T']):.1f} K</b></li><li>Min: <b>{np.min(res['T_gas_T']):.1f} K</b></li><li>Průměr: <b>{np.mean(res['T_gas_T']):.1f} K</b></li></ul></li><li>Studená strana (T<sub>Ss</sub>):<ul><li>Max: <b>{np.max(res['T_gas_S']):.1f} K</b></li><li>Min: <b>{np.min(res['T_gas_S']):.1f} K</b></li><li>Průměr: <b>{np.mean(res['T_gas_S']):.1f} K</b></li></ul></li></ul></div>""", unsafe_allow_html=True)
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(f"""<div class="result-box" style="height: 190px;"><div class="box-title">Tlakové poměry</div><ul><li>Tlakový poměr ψ: <b>{res['pressure_ratio']:.2f} [-]</b></li><li>Max. tlak p<sub>max</sub>: <b>{np.max(res['p_real'])/1e6:.2f} MPa</b></li><li>Min. tlak p<sub>min</sub>: <b>{np.min(res['p_real'])/1e6:.2f} MPa</b></li><li>Střední tlak p<sub>stř</sub>: <b>{lp['p_st_MPa']:.2f} MPa</b></li></ul></div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""<div class="result-box" style="height: 190px;"><div class="box-title">Hmotnost náplně</div><ul><li>Celková hmotnost média (m<sub>celk</sub>): <b>{res['mass_total_g']:.4f} g</b></li><li>Relativní odchylka hmotnosti: <b>{res['mass_deviation']:.3f} %</b></li></ul></div>""", unsafe_allow_html=True)

with tab2:
    col1a, col1b = st.columns(2)
    with col1a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['V']*1e6, y=res['p_real']/1e6, mode='lines', line=dict(color='black', width=2), name='p-V cyklus'))
        fig.update_layout(title=dict(text="p-V diagram", x=0.5, xanchor='center', yanchor='top'), xaxis_title="V (cm³)", yaxis_title="p (MPa)", height=400, **layout_style)
        st.plotly_chart(fig, use_container_width=True)
    with col1b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['p_real']/1e6, mode='lines', line=dict(color='black', width=2), name='Tlak p'))
        fig.add_hline(y=lp['p_st_MPa'], line_dash="dash", line_color="red", annotation_text="p<sub>stř</sub>")
        fig.update_layout(title=dict(text="Průběh tlaku v závislosti na φ", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="p (MPa)", height=400, **layout_style)
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
        st.plotly_chart(fig, use_container_width=True)
    col2a, col2b = st.columns(2)
    with col2a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['VT']*1e6, y=res['p_real']/1e6, mode='lines', line=dict(color='red', width=2), name='Teplý válec'))
        fig.update_layout(title=dict(text="p-V<sub>T</sub> diagram (Teplý válec)", x=0.5, xanchor='center', yanchor='top'), xaxis_title="V<sub>T</sub> (cm³)", yaxis_title="p (MPa)", height=400, **layout_style)
        st.plotly_chart(fig, use_container_width=True)
    with col2b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['VS']*1e6, y=res['p_real']/1e6, mode='lines', line=dict(color='blue', width=2), name='Studený válec'))
        fig.update_layout(title=dict(text="p-V<sub>S</sub> diagram (Studený válec)", x=0.5, xanchor='center', yanchor='top'), xaxis_title="V<sub>S</sub> (cm³)", yaxis_title="p (MPa)", height=400, **layout_style)
        st.plotly_chart(fig, use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['VT']*1e6, mode='lines', line=dict(color='red'), name='V<sub>T</sub>'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['VS']*1e6, mode='lines', line=dict(color='blue'), name='V<sub>S</sub>'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['V']*1e6, mode='lines', line=dict(color='black', width=3), name='V<sub>celk</sub>'))
    fig.add_hline(y=res['VTM']*1e6, line_dash="dot", line_color="red", annotation_text="V<sub>TM</sub>")
    fig.add_hline(y=res['VSM']*1e6, line_dash="dot", line_color="blue", annotation_text="V<sub>SM</sub>")
    fig.add_hline(y=res['VR']*1e6, line_dash="dash", line_color="magenta", annotation_text="V<sub>R</sub>")
    fig.update_layout(title=dict(text="Průběh objemů v závislosti na φ", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="V (cm³)", height=500, **layout_style)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Teplota média v motoru v průběhu cyklu")
    x_TR_intersect = np.interp(res['T_reg_mean'], res['T_reg_profile'][::-1], res['x_reg_vals'][::-1])
    fig_3d = go.Figure(data=[go.Surface(z=res['T_surface'], x=res['x_grid'], y=res['phi_grid'], colorscale='Jet', colorbar=dict(title='T (K)'))])
    fig_3d.add_trace(go.Scatter3d(x=[x_TR_intersect]*2, y=[0, 360], z=[res['T_reg_mean']]*2, mode='lines', line=dict(color='magenta', width=8), name='T_R'))
    fig_3d.update_layout(scene=dict(xaxis_title='x (-)', yaxis_title='φ (°)', zaxis_title='T (K)'), margin=dict(l=0, r=0, b=0, t=10), height=700)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['T_gas_T'], mode='lines', line=dict(color='red', width=3), name='T<sub>Ts</sub>'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['T_gas_S'], mode='lines', line=dict(color='blue', width=3), name='T<sub>Ss</sub>'))
    max_idx_T = np.argmax(res['T_gas_T']); min_idx_T = np.argmin(res['T_gas_T'])
    max_idx_S = np.argmax(res['T_gas_S']); min_idx_S = np.argmin(res['T_gas_S'])
    fig.add_trace(go.Scatter(x=[res['phi_deg'][max_idx_T]], y=[res['T_gas_T'][max_idx_T]], mode='markers', marker=dict(color='red', size=8), showlegend=False))
    fig.add_trace(go.Scatter(x=[res['phi_deg'][min_idx_T]], y=[res['T_gas_T'][min_idx_T]], mode='markers', marker=dict(color='red', size=8), showlegend=False))
    fig.add_trace(go.Scatter(x=[res['phi_deg'][max_idx_S]], y=[res['T_gas_S'][max_idx_S]], mode='markers', marker=dict(color='blue', size=8), showlegend=False))
    fig.add_trace(go.Scatter(x=[res['phi_deg'][min_idx_S]], y=[res['T_gas_S'][min_idx_S]], mode='markers', marker=dict(color='blue', size=8), showlegend=False))
    fig.add_hline(y=lp['TT'], line_dash="dash", line_color="red", annotation_text="T<sub>T</sub>")
    fig.add_hline(y=lp['TS'], line_dash="dash", line_color="blue", annotation_text="T<sub>S</sub>")
    fig.add_hline(y=res['T_reg_mean'], line_dash="dot", line_color="magenta", annotation_text="T<sub>R</sub>")
    fig.update_layout(title=dict(text="Průběh teplot v závislosti na φ", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="T (K)", height=500, **layout_style)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['T_gas_S']/res['T_gas_T'], mode='lines', line=dict(color='blue'), name='T<sub>Ss</sub> / T<sub>Ts</sub>'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['T_reg_mean']/res['T_gas_T'], mode='lines', line=dict(color='magenta'), name='T<sub>R</sub> / T<sub>Ts</sub>'))
    fig.add_hline(y=np.mean(res['T_gas_S']/res['T_gas_T']), line_dash="dash", line_color="blue")
    fig.add_hline(y=np.mean(res['T_reg_mean']/res['T_gas_T']), line_dash="dash", line_color="magenta")
    fig.update_layout(title=dict(text="Průběh teplotních poměrů", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="Poměr (-)", height=400, **layout_style)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['dQ_R_curve'], mode='lines', line=dict(color='black', width=2), showlegend=False))
    valMax, valMin = np.max(res['dQ_R_curve']), np.min(res['dQ_R_curve'])
    fig.add_hline(y=valMax, line_dash="dot", line_color="black")
    fig.add_hline(y=valMin, line_dash="dot", line_color="black")
    fig.add_annotation(x=225, y=valMin, ax=225, ay=valMax, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowside="end+start", arrowcolor="red", arrowwidth=2)
    fig.add_annotation(x=245, y=(valMax+valMin)/2, text=f"Q<sub>R</sub> = {res['Q_reg_val']:.1f} J", showarrow=False, font=dict(color="red", size=14, weight="bold"), xanchor="left")
    fig.update_layout(title=dict(text="Průběh regenerovaného tepla Q<sub>R</sub>", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="Q<sub>R</sub> (J)", height=500, **layout_style)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['p_real'] * res['dVT_dphi'], fill='tozeroy', mode='lines', line=dict(color='red', width=1), name='Teplá (p·dV<sub>T</sub>/dφ)'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['p_real'] * res['dVS_dphi'], fill='tozeroy', mode='lines', line=dict(color='blue', width=1), name='Studená (p·dV<sub>S</sub>/dφ)'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=(res['p_real'] * res['dVT_dphi']) + (res['p_real'] * res['dVS_dphi']), mode='lines', line=dict(color='black', width=3, dash='dash'), name='Celkem (p·dV/dφ)'))
    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.update_layout(title=dict(text="Okamžitá práce (p · dV)", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="dW/dφ (J/rad)", height=500, **layout_style)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['m_T_g'], mode='lines', line=dict(color='red', width=2), name='m<sub>T</sub>'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['m_S_g'], mode='lines', line=dict(color='blue', width=2), name='m<sub>S</sub>'))
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['m_R_g'], mode='lines', line=dict(color='magenta', width=2, dash='dash'), name='m<sub>R</sub>'))
    fig.update_layout(title=dict(text="Hmotnost média v jednotlivých prostorech v průběhu cyklu", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="Hmotnost (g)", height=500, **layout_style)
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['phi_deg'], y=res['m_inst']*1000, mode='lines', line=dict(color='black', width=2), name='m<sub>celk</sub>'))
    fig.add_hline(y=res['mass_total_g'], line_dash="dash", line_color="red", annotation_text="Průměr")
    y_mid = res['mass_total_g']
    fig.update_layout(title=dict(text=f"Celková hmotnost média m<sub>celk</sub>", x=0.5, xanchor='center', yanchor='top'), xaxis_title="φ (°)", yaxis_title="m<sub>celk</sub> (g)", height=400, **layout_style)
    fig.update_yaxes(range=[y_mid*0.999, y_mid*1.001])
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=45)
    st.plotly_chart(fig, use_container_width=True)

def add_extrema(fig, x, y, color, secondary_y=None, y_fmt=".2f", x_fmt=".2f", ay_max=-35, ay_min=35):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n < 3: return
    if np.max(y) - np.min(y) < 1e-6: return
    
    idx_max = np.argmax(y)
    idx_min = np.argmin(y)
    y_ref = 'y2' if secondary_y else 'y'
    
    if 0 < idx_max < n - 1:
        trace = go.Scatter(
            x=[x[idx_max]], y=[y[idx_max]], mode='markers', 
            marker=dict(color=color, size=9, line=dict(color='white', width=1.5)), 
            showlegend=False
        )
        if secondary_y is not None:
            fig.add_trace(trace, secondary_y=secondary_y)
        else:
            fig.add_trace(trace)

        fig.add_annotation(
            x=x[idx_max], y=y[idx_max], 
            text=f"Max: {y[idx_max]:{y_fmt}}<br>x = {x[idx_max]:{x_fmt}}", 
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=color, ax=0, ay=ay_max, 
            font=dict(color=color, size=11), bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, borderpad=2,
            yref=y_ref, xref='x'
        )
        
    if 0 < idx_min < n - 1:
        trace = go.Scatter(
            x=[x[idx_min]], y=[y[idx_min]], mode='markers', 
            marker=dict(color=color, size=9, line=dict(color='white', width=1.5)), 
            showlegend=False
        )
        if secondary_y is not None:
            fig.add_trace(trace, secondary_y=secondary_y)
        else:
            fig.add_trace(trace)

        fig.add_annotation(
            x=x[idx_min], y=y[idx_min], 
            text=f"Min: {y[idx_min]:{y_fmt}}<br>x = {x[idx_min]:{x_fmt}}", 
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=color, ax=0, ay=ay_min, 
            font=dict(color=color, size=11), bgcolor="rgba(255,255,255,0.85)", bordercolor=color, borderwidth=1, borderpad=2,
            yref=y_ref, xref='x'
        )

with tab6:
    col_sweep1, col_sweep2 = st.columns([1, 2.5])
    
    with col_sweep1:
        param_options = [
            "Střední tlak p_stř (MPa)", 
            "Frekvence f (Hz)", 
            "Teplota ohřívače T_T (K)", 
            "Teplota chladiče T_S (K)", 
            "Fázový posun α (°)",
            "Poměr zdvihů X_SZ (-)",
            "Mrtvý objem teplý X_TM (-)",
            "Mrtvý objem studený X_SM (-)",
            "Objem regenerátoru X_R (-)",
            "Objem překryvu zdvihů V_P (% ideálu)",
            "Polytropický exponent n (-)"
        ]
        param_type = st.selectbox("Měněný parametr (osa X):", param_options, key='param_x_sel')

        y_options = [
            "Výkon P (kW) a Účinnost η",
            "Tlaky (p_max, p_min) a poměr ψ", 
            "Regenerace (Q_R, poměr Q_R/Q_in)",
            "Energie (W, Q_in, |Q_out|)",
            "Celková hmotnost média m_celk"
        ]
        y_choice = st.selectbox("Zkoumaná veličina (osa Y):", y_options, key='param_y_sel')
        
        st.markdown("<hr style='margin: 5px 0 10px 0;'>", unsafe_allow_html=True)

        if "tlak" in param_type: min_v, max_v, step_v, curr_val = 1.0, 30.0, 1.0, lp['p_st_MPa']
        elif "Frekvence" in param_type: min_v, max_v, step_v, curr_val = 10.0, 200.0, 10.0, lp['f']
        elif "T_T" in param_type: min_v, max_v, step_v, curr_val = 500.0, 1500.0, 50.0, lp['TT']
        elif "T_S" in param_type: min_v, max_v, step_v, curr_val = 200.0, 600.0, 20.0, lp['TS']
        elif "Fázový" in param_type: min_v, max_v, step_v, curr_val = 0.0, 180.0, 5.0, lp['alpha_deg']
        elif "X_SZ" in param_type: min_v, max_v, step_v, curr_val = 0.5, 3.0, 0.1, lp['XSZ']
        elif "X_TM" in param_type: min_v, max_v, step_v, curr_val = 0.0, 5.0, 0.1, lp['XTM']
        elif "X_SM" in param_type: min_v, max_v, step_v, curr_val = 0.0, 5.0, 0.1, lp['XSM']
        elif "X_R" in param_type: min_v, max_v, step_v, curr_val = 0.0, 5.0, 0.1, lp['XR']
        elif "V_P" in param_type: min_v, max_v, step_v, curr_val = 0.0, 100.0, 5.0, lp['vp_percent']
        elif "exponent" in param_type: min_v, max_v, step_v, curr_val = 1.0, 1.67, 0.05, lp['n_poly']
        
        st.markdown(f"<p style='margin-top:-5px; margin-bottom:5px; font-size:0.95rem;'><b>Výchozí stav:</b> {curr_val}</p>", unsafe_allow_html=True)
        
        c_min, c_max = st.columns(2)
        with c_min:
            sweep_min = st.number_input("Min hodnota osy X", value=float(min_v), step=step_v, key=f's_min_{param_type}')
        with c_max:
            sweep_max = st.number_input("Max hodnota osy X", value=float(max_v), step=step_v, key=f's_max_{param_type}')
            
        steps_count = st.slider("Počet kroků výpočtu", 5, 200, 100, key=f's_steps_{param_type}')
        st.markdown("<br>", unsafe_allow_html=True)
        run_sweep = st.button("🚀 Spustit analýzu", type="primary", use_container_width=True)

    with col_sweep2:
        if run_sweep:
            x_vals = np.linspace(sweep_min, sweep_max, steps_count)
            results = []
            
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            
            for idx, x_val in enumerate(x_vals):
                sweep_params = lp.copy() 
                if "tlak" in param_type: sweep_params['p_st_MPa'] = x_val
                elif "Frekvence" in param_type: sweep_params['f'] = x_val
                elif "T_T" in param_type: sweep_params['TT'] = x_val
                elif "T_S" in param_type: sweep_params['TS'] = x_val
                elif "Fázový" in param_type: sweep_params['alpha_deg'] = x_val
                elif "X_SZ" in param_type: sweep_params['XSZ'] = x_val
                elif "X_TM" in param_type: sweep_params['XTM'] = x_val
                elif "X_SM" in param_type: sweep_params['XSM'] = x_val
                elif "X_R" in param_type: sweep_params['XR'] = x_val
                elif "V_P" in param_type: sweep_params['vp_percent'] = x_val
                elif "exponent" in param_type: sweep_params['n_poly'] = x_val
                
                sweep_res = solve_cycle_sweep(sweep_params)
                results.append(sweep_res)
                progress_bar.progress((idx + 1) / steps_count)
            
            progress_container.empty() 

            if "tlak" in param_type: x_label = "Střední tlak p<sub>stř</sub> (MPa)"
            elif "Frekvence" in param_type: x_label = "Frekvence f (Hz)"
            elif "T_T" in param_type: x_label = "Teplota ohřívače T<sub>T</sub> (K)"
            elif "T_S" in param_type: x_label = "Teplota chladiče T<sub>S</sub> (K)"
            elif "Fázový" in param_type: x_label = "Úhel α (°)"
            elif "X_SZ" in param_type: x_label = "Poměr objemů X<sub>SZ</sub> (-)"
            elif "X_TM" in param_type: x_label = "Mrtvý objem X<sub>TM</sub> (-)"
            elif "X_SM" in param_type: x_label = "Mrtvý objem X<sub>SM</sub> (-)"
            elif "X_R" in param_type: x_label = "Mrtvý objem X<sub>R</sub> (-)"
            elif "V_P" in param_type: x_label = "Objem překryvu V<sub>P</sub> (% ideálu)"
            elif "exponent" in param_type: x_label = "Exponent n (-)"
            
            title_declension = {
                "Střední tlak p_stř (MPa)": "středním tlaku p<sub>stř</sub>",
                "Frekvence f (Hz)": "frekvenci f",
                "Teplota ohřívače T_T (K)": "teplotě ohřívače T<sub>T</sub>",
                "Teplota chladiče T_S (K)": "teplotě chladiče T<sub>S</sub>",
                "Fázový posun α (°)": "fázovém posunu α",
                "Poměr zdvihů X_SZ (-)": "poměru zdvihů X<sub>SZ</sub>",
                "Mrtvý objem teplý X_TM (-)": "mrtvém objemu X<sub>TM</sub>",
                "Mrtvý objem studený X_SM (-)": "mrtvém objemu X<sub>SM</sub>",
                "Objem regenerátoru X_R (-)": "objemu regenerátoru X<sub>R</sub>",
                "Objem překryvu zdvihů V_P (% ideálu)": "objemu překryvu V<sub>P</sub>",
                "Polytropický exponent n (-)": "polytropickém exponentu n"
            }
            declined_param = title_declension.get(param_type, param_type)

            y_title_main = y_choice.split('(')[0].strip()
            
            if "Výkon" in y_choice:
                arr_P = [r['P']/1000 for r in results]
                arr_eta = [r['eta'] for r in results]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=x_vals, y=arr_P, name="Výkon (kW)", line=dict(color='blue', width=3)), secondary_y=False)
                fig.add_trace(go.Scatter(x=x_vals, y=arr_eta, name="Účinnost (%)", line=dict(color='red', width=3, dash='dot')), secondary_y=True)
                
                add_extrema(fig, x_vals, arr_P, 'blue', secondary_y=False, y_fmt=".2f")
                add_extrema(fig, x_vals, arr_eta, 'red', secondary_y=True, y_fmt=".1f")

                eta_min, eta_max = np.min(arr_eta), np.max(arr_eta)
                if eta_max - eta_min < 0.1:
                    eta_mean = np.mean(arr_eta)
                    fig.update_yaxes(title_text="Účinnost η (%)", range=[eta_mean - 1, eta_mean + 1], secondary_y=True, showgrid=False, title_font=dict(color="red"), showline=False, mirror=False)
                else:
                    fig.update_yaxes(title_text="Účinnost η (%)", secondary_y=True, showgrid=False, title_font=dict(color="red"), showline=False, mirror=False)
                
                fig.update_yaxes(title_text="Výkon P (kW)", secondary_y=False, showgrid=True, gridcolor='#d9d9d9', title_font=dict(color="blue"), showline=False, mirror=False)

            elif "Tlaky" in y_choice:
                arr_pmax = [r['p_max'] for r in results]
                arr_pmin = [r['p_min'] for r in results]
                arr_psi = [r['psi'] for r in results]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=x_vals, y=arr_pmax, name="Max. tlak p<sub>max</sub>", line=dict(color='red', width=3)), secondary_y=False)
                fig.add_trace(go.Scatter(x=x_vals, y=arr_pmin, name="Min. tlak p<sub>min</sub>", line=dict(color='blue', width=3)), secondary_y=False)
                fig.add_trace(go.Scatter(x=x_vals, y=arr_psi, name="Tlakový poměr ψ", line=dict(color='purple', width=3, dash='dot')), secondary_y=True)
                
                add_extrema(fig, x_vals, arr_pmax, 'red', secondary_y=False, y_fmt=".2f")
                add_extrema(fig, x_vals, arr_pmin, 'blue', secondary_y=False, y_fmt=".2f")
                add_extrema(fig, x_vals, arr_psi, 'purple', secondary_y=True, y_fmt=".2f")

                fig.update_yaxes(title_text="Tlak p (MPa)", secondary_y=False, showgrid=True, gridcolor='#d9d9d9', showline=False, mirror=False)
                fig.update_yaxes(title_text="Tlakový poměr ψ (-)", secondary_y=True, showgrid=False, title_font=dict(color="purple"), showline=False, mirror=False)

            elif "Regenerace" in y_choice:
                arr_qreg = [r['Q_reg'] for r in results]
                arr_qratio = [r['Q_ratio'] for r in results]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=x_vals, y=arr_qreg, name="Regenerované teplo Q<sub>R</sub>", line=dict(color='darkgreen', width=3)), secondary_y=False)
                fig.add_trace(go.Scatter(x=x_vals, y=arr_qratio, name="Poměr Q<sub>R</sub> / Q<sub>in</sub>", line=dict(color='olive', width=3, dash='dot')), secondary_y=True)
                
                add_extrema(fig, x_vals, arr_qreg, 'darkgreen', secondary_y=False, y_fmt=".1f")
                add_extrema(fig, x_vals, arr_qratio, 'olive', secondary_y=True, y_fmt=".2f")

                fig.update_yaxes(title_text="Regenerované teplo Q<sub>R</sub> (J)", secondary_y=False, showgrid=True, gridcolor='#d9d9d9', title_font=dict(color="darkgreen"), showline=False, mirror=False)
                fig.update_yaxes(title_text="Poměr Q<sub>R</sub> / Q<sub>in</sub> (-)", secondary_y=True, showgrid=False, title_font=dict(color="olive"), showline=False, mirror=False)

            elif "Energie" in y_choice:
                arr_w = [r['W'] for r in results]
                arr_qin = [r['Q_in'] for r in results]
                arr_qout = [r['Q_out'] for r in results]
                
                max_vals = {'qin': np.max(arr_qin), 'w': np.max(arr_w), 'qout': np.max(arr_qout)}
                lowest_key = sorted(max_vals, key=max_vals.get)[0]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=arr_qin, name="Přivedené Q<sub>in</sub>", line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=x_vals, y=arr_w, name="Práce W", line=dict(color='black', width=3, dash='dash')))
                fig.add_trace(go.Scatter(x=x_vals, y=arr_qout, name="Odvedené |Q<sub>out</sub>|", line=dict(color='blue', width=3)))
                
                add_extrema(fig, x_vals, arr_qin, 'red', secondary_y=None, y_fmt=".1f", ay_max=45 if lowest_key == 'qin' else -35)
                add_extrema(fig, x_vals, arr_w, 'black', secondary_y=None, y_fmt=".1f", ay_max=45 if lowest_key == 'w' else -35)
                add_extrema(fig, x_vals, arr_qout, 'blue', secondary_y=None, y_fmt=".1f", ay_max=45 if lowest_key == 'qout' else -35)

                fig.update_yaxes(title_text="Energie (J)", showgrid=True, gridcolor='#d9d9d9', showline=False, mirror=False)

            elif "hmotnost" in y_choice:
                arr_m = [r['m_celk'] for r in results]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_vals, y=arr_m, name="m<sub>celk</sub>", line=dict(color='saddlebrown', width=3)))
                
                add_extrema(fig, x_vals, arr_m, 'saddlebrown', secondary_y=None, y_fmt=".3f")

                fig.update_yaxes(title_text="Hmotnost média m<sub>celk</sub> (g)", showgrid=True, gridcolor='#d9d9d9', showline=False, mirror=False)

            fig.update_layout(title=dict(text=f"Závislost: {y_title_main} na {declined_param}", x=0.5, xanchor='center', yanchor='top'), xaxis_title=x_label, height=500, **layout_style)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("")
            
        st.markdown("---")
        st.caption(
            "Tento nástroj provádí sérii výpočtů (simulací), kde postupně mění **jeden zvolený parametr** (osa X) v zadaném rozsahu. "
            "Zároveň sleduje vliv na vámi **vybranou veličinu** (osa Y). Všechny ostatní parametry zůstávají zafixované na hodnotách, "
            "které máte aktuálně potvrzené z levého panelu. To umožňuje zkoumat chování motoru v extrémních stavech bez přepisování celého modelu."
        )

with tab7:
    st.markdown("### 🎯 Odhad reálného výkonu pomocí Bealeova čísla")
    st.write(
        "Tento nástroj umožňuje odhadnout skutečný výkon vašeho stroje na základě empirických dat sestavených G. Walkerem. "
        "Na rozdíl od ideálního indikovaného výkonu ($P_{ind}$), tento výpočet v sobě zahrnuje reálné aerodynamické i mechanické ztráty v závislosti na typu motoru."
    )
    
    # Funkce pro výpočet Bn s asymptotickou extrapolací (křivka se zplošťuje)
    def get_bn_val(T_act, curve_y, curve_x):
        if T_act > 1200:
            slope = (curve_y[-1] - curve_y[-2]) / (curve_x[-1] - curve_x[-2])
            L = 300.0
            return curve_y[-1] + slope * L * (1.0 - np.exp(-(T_act - 1200.0) / L))
        elif T_act < 600:
            slope = (curve_y[1] - curve_y[0]) / (curve_x[1] - curve_x[0])
            L = 300.0
            return max(0.001, curve_y[0] - slope * L * (1.0 - np.exp(-(600.0 - T_act) / L)))
        else:
            return np.interp(T_act, curve_x, curve_y)

    T_actual = lp['TT']
    bn_top = get_bn_val(T_actual, curve_top, T_range_full)
    bn_mid = get_bn_val(T_actual, curve_mid, T_range_full)
    bn_bot = get_bn_val(T_actual, curve_bot, T_range_full)
    bn_fsps = get_bn_val(T_actual, curve_fsps, T_range_full)
    
    p_mean_pa = lp['p_st_MPa'] * 1e6
    vsz_m3 = lp['VTZ_ccm'] * lp['XSZ'] * 1e-6
    freq_hz = lp['f']
    P_ind = res['Power_ind']
    
    bns = [bn_top, bn_mid, bn_bot, bn_fsps]
    
    base_options = [
        "Velké, dobře navržené a vysoce účinné motory s dobrým chlazením",
        "Běžné, průměrně navržené motory",
        "Menší, ekonomické motory se střední účinností navržené pro dlouhou životnost nebo omezené chlazení",
        "Motory s volnými písty a velkými mrtvými objemy"
    ]
    
    best_idx = 0
    min_diff = float('inf')
    for i, bn in enumerate(bns):
        P_b = bn * p_mean_pa * vsz_m3 * freq_hz
        ratio = P_b / P_ind if P_ind > 0 else 0
        diff = abs(ratio - 0.5)
        if diff < min_diff:
            min_diff = diff
            best_idx = i

    st.markdown(f"""
    <div style="margin-top: 15px; margin-bottom: 5px;">
        <span style="color: black; font-weight: bold; font-size: 1.1rem;">✅ Doporučená volba: {base_options[best_idx]}</span><br>
        <span style="font-size: 0.85rem; color: #666; margin-left: 28px;">(doporučeno na základě zvoleného mrtvého objemu)</span>
    </div>
    """, unsafe_allow_html=True)

    # Vytvoření dynamických textů pro přepínač (tučné písmo a ikona pro vítěze)
    display_options = list(base_options)
    display_options[best_idx] = f"✅ **{base_options[best_idx]}** (← Doporučeno pro zvolené mrtvé objemy)"

    if 'selected_curve_idx' not in st.session_state:
        st.session_state.selected_curve_idx = best_idx

    # Vynucení doporučení při přepočtu modelu (spolehlivě přepne index v session_state)
    if st.session_state.get('force_auto_curve', False):
        st.session_state.selected_curve_idx = best_idx
        st.session_state.force_auto_curve = False

    # Natvrdo propíšeme hodnotu do session state před inicializací radio buttonu
    st.session_state['curve_choice_radio'] = display_options[st.session_state.selected_curve_idx]

    def update_curve_idx():
        selected_str = st.session_state.curve_choice_radio
        for i, opt in enumerate(display_options):
            if opt == selected_str:
                st.session_state.selected_curve_idx = i
                break

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Renderování radio buttonu
    curve_choice = st.radio("Vyberte referenční kategorii vašeho motoru pro odečet Bn:", 
                            display_options, 
                            key="curve_choice_radio",
                            on_change=update_curve_idx)
        
    chosen_idx = st.session_state.selected_curve_idx
    bn_val = bns[chosen_idx]
        
    if T_actual < 600 or T_actual > 1200:
        st.warning(f"Zadaná teplota ohřívače ($T_T={T_actual}$ K) je mimo běžný rozsah empirických dat (600 - 1200 K). Hodnota Bealeova čísla byla odhadnuta extrapolací.")
        
    P_beale = bn_val * p_mean_pa * vsz_m3 * freq_hz
    
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
    
    cb1, cb2, cb3 = st.columns(3)
    cb1.metric("Odečtené Bealeovo číslo (B_n)", f"{bn_val:.4f} [-]")
    cb2.metric("Odhadovaný skutečný výkon", f"{P_beale/1000:.2f} kW")
    cb3.metric("Poměr vůči ideálnímu výkonu P", f"{(P_beale / P_ind) * 100:.1f} %")

    st.markdown("<br>", unsafe_allow_html=True)

    fig_beale, ax_b = plt.subplots(figsize=(10, 5.5), dpi=120)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})
    
    limit_ocel = 950
    limit_super = 1050
    
    ax_b.axvspan(limit_ocel, limit_super, color='#fff9c4', alpha=0.5, lw=0)
    ax_b.axvspan(limit_super, 1200, color='#ffcdd2', alpha=0.5, lw=0)
    ax_b.axvline(x=limit_ocel, color='#333', linestyle='-', linewidth=1)
    ax_b.axvline(x=limit_super, color='#333', linestyle='-', linewidth=1)

    ax_b.grid(which='major', color='#555', linestyle='-', linewidth=0.6, alpha=0.4)
    ax_b.minorticks_on()
    ax_b.grid(which='minor', color='#999', linestyle='--', linewidth=0.5, alpha=0.3)
    ax_b.xaxis.set_major_locator(ticker.MultipleLocator(100)) 
    ax_b.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax_b.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    c_read = '#4ded30'

    ax_b.plot(T_range_full, curve_top, 'k--', linewidth=1.5) 
    ax_b.plot(T_range_full, curve_mid, 'k-', linewidth=1.5, alpha=0.7) 
    ax_b.plot(T_range_full, curve_bot, 'k-.', linewidth=1.5)
    ax_b.plot(T_range_full, curve_fsps, color='#d32f2f', linestyle='-', linewidth=2, label='Trend')

    bbox_style = dict(boxstyle="round,pad=0.4", fc="white", ec="#aaa", alpha=0.9)
    bbox_style_red = dict(boxstyle="round,pad=0.4", fc="white", ec="#d32f2f", alpha=0.9)
    bbox_style_mag = dict(boxstyle="round,pad=0.4", fc="white", ec=c_read, alpha=0.9)

    ax_b.text(615, 0.229, r"$P_{skutečný} = B_n \cdot p_{stř} \cdot V_{SZ} \cdot f$", 
              fontsize=13, va='bottom', bbox=dict(facecolor='white', alpha=0.9, edgecolor='#aaa', boxstyle='round,pad=0.5'))

    T_plot_star = np.clip(T_actual, 600, 1200)
    ax_b.plot([T_plot_star, T_plot_star], [0, bn_val], color=c_read, linestyle=':', lw=1.5)
    ax_b.plot([600, T_plot_star], [bn_val, bn_val], color=c_read, linestyle=':', lw=1.5)
    ax_b.scatter(T_plot_star, bn_val, color=c_read, s=250, marker='*', zorder=15, edgecolors='black')

    box_x = 1075 
    ax_b.text(box_x, 0.13, f"       Aktuální model\n       Bn={bn_val:.3f}", 
              fontsize=11, fontweight='bold', color=c_read, ha='left', va='center', 
              bbox=bbox_style_mag, zorder=14)
    ax_b.scatter(1090, 0.13, color=c_read, s=250, marker='*', zorder=15, edgecolors='black')

    ax_b.annotate("Velké, dobře navržené\nvysoce účinné motory\ns dobrým chlazením", 
                  xy=(800, 0.170), xytext=(780, 0.190),
                  arrowprops=dict(arrowstyle="->", color='black', lw=0.8), 
                  fontsize=8, fontweight='bold', ha='center', bbox=bbox_style, multialignment='center')
                  
    ax_b.annotate("Běžné, průměrně\nnavržené motory", 
                  xy=(850, 0.126), xytext=(850, 0.150), 
                  arrowprops=dict(arrowstyle="->", color='black', lw=0.8), 
                  fontsize=8, fontweight='bold', ha='center', va='bottom', bbox=bbox_style, multialignment='center')
                  
    ax_b.annotate("Menší, ekonomické motory\nse střední účinností navržené\npro dlouhou životnost nebo\nomezené chlazení", 
                  xy=(800, 0.060), xytext=(800, 0.108), 
                  arrowprops=dict(arrowstyle="->", color='black', lw=0.8), 
                  fontsize=8, fontweight='bold', ha='left', va='top', bbox=bbox_style, multialignment='center')

    ax_b.annotate("Motory s volnými písty\na velkými mrtvými objemy", 
                  xy=(800, 0.036), xytext=(800, 0.010), 
                  arrowprops=dict(arrowstyle="->", color='#d32f2f', lw=1.2), 
                  fontsize=8, fontweight='bold', color='#d32f2f', ha='left', va='bottom', bbox=bbox_style_red, multialignment='center')

    ax_b.text(limit_ocel - 8, 0.085, "Limit běžné oceli", rotation=90, va='bottom', ha='right', fontsize=8, color='#444')
    ax_b.text((limit_ocel + limit_super)/2, 0.01, "VYSOCE LEGOVANÉ\nOCELI", ha='center', va='center', fontsize=9, fontweight='bold', color='#f57f17')
    ax_b.text(1125, 0.01, "KERAMIKA", ha='center', va='center', fontsize=9, fontweight='bold', color='#c62828')

    ax_b.set_xlim(600, 1200)
    ax_b.set_ylim(0, 0.26)
    ax_b.set_xlabel('Teplota ohřívače $T_T$ [K]', fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Bealeovo číslo $B_n$ [-]', fontsize=11, fontweight='bold')
    ax_b.set_title(r'Odečtení Bealeova čísla na základě teploty a referenční křivky', fontsize=14, pad=15)

    fig_beale.tight_layout()
    st.pyplot(fig_beale)
    
    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
    st.caption("Graf zrekonstruován z původní předlohy. **Zdroj:** MARTINI, William. *Stirling engine design manual*, 2004. Přetisk vydání z roku 1983. Honolulu: University press of the Pacific, ISBN: 1-4102-1604-7.")