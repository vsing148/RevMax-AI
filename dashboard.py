import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000/optimize"
st.set_page_config(page_title="RevMax - AI Pricing Strategy", layout="wide", page_icon="⚡")

# --- RevMax UI - Premium Dark Glassmorphism Theme ---
st.markdown("""
<style>
    /* -------------------------------------------------------------------------- */
    /*                                 VARIABLES                                  */
    /* -------------------------------------------------------------------------- */
    :root {
        --bg-dark: #0a0a0a;
        --bg-panel: #111111;
        --glass-panel: rgba(20, 20, 20, 0.6);
        --glass-border: 1px solid rgba(255, 255, 255, 0.08);
        
        --primary-gradient: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
        --primary-glow: 0 0 20px rgba(139, 92, 246, 0.4);
        
        --text-primary: #ffffff;
        --text-secondary: #9ca3af;
        --text-accent: #06b6d4;
        
        --font-main: 'Inter', system-ui, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* -------------------------------------------------------------------------- */
    /*                                GLOBAL RESET                                */
    /* -------------------------------------------------------------------------- */
    html, body, [class*="css"] {
        font-family: var(--font-main);
        color: var(--text-primary);
    }
    
    .stApp {
        background-color: var(--bg-dark);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 20%),
            radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.1) 0%, transparent 20%);
    }

    /* -------------------------------------------------------------------------- */
    /*                                  SIDEBAR                                   */
    /* -------------------------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-panel);
        border-right: 1px solid rgba(255,255,255,0.05);
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }
    
    section[data-testid="stSidebar"] h1 {
        font-family: var(--font-main);
        font-weight: 800;
        font-size: 2rem;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* -------------------------------------------------------------------------- */
    /*                                 COMPONENTS                                 */
    /* -------------------------------------------------------------------------- */
    
    /* Metrics / Cards - Glassmorphism */
    div[data-testid="stMetric"] {
        background: var(--glass-panel);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: var(--glass-border);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.2);
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 2rem;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
    }

    /* Buttons - Neon Gradient */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: var(--primary-glow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.6);
        filter: brightness(1.2);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: var(--primary-gradient);
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 12px;
        border-radius: 99px;
        font-family: var(--font-mono);
        font-size: 0.8rem;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Remove Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit Toolbar and Deploy Button */
    [data-testid="stToolbar"], [data-testid="stHeader"], .stDeployButton {
        visibility: hidden;
        height: 0%;
        display: none;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        display: none;
    }

</style>
""", unsafe_allow_html=True)

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("<h1>RevMax</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-secondary); margin-bottom: 30px;'>Revenue Intelligence Engine v3.0</p>", unsafe_allow_html=True)
    
    st.markdown("### MARKET PARAMETERS")
    competitor_price = st.slider("Competitor Rate ($)", 40.0, 60.0, 50.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### TEMPORAL CONTEXT")
    day_type = st.radio("Active Period", ["Weekday", "Weekend"])
    is_weekend = 1 if day_type == "Weekend" else 0
    
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    generate_btn = st.button("INITIATE OPTIMIZATION")

# --- Main Dashboard UI ---

# Top Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("Pricing Strategy Dashboard")

st.markdown("<br>", unsafe_allow_html=True)

if generate_btn:
    # 1. Call the API
    payload = {"competitor_price": competitor_price, "is_weekend": is_weekend}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # 2. Bento Grid Layout for Metrics
                # Row 1: Key Metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                
                with m_col1:
                    st.metric("Optimal Price Point", f"${data['suggested_price']}")
                with m_col2:
                    st.metric("Projected Volume", f"{data['predicted_sales']} units")
                with m_col3:
                    st.metric("Predicted Revenue", f"${data['predicted_revenue']}")
                    
                st.markdown("<br>", unsafe_allow_html=True)

                # 3. Visualization Card
                st.markdown("### Revenue Optimization Curve")
                
                chart_data = pd.DataFrame(data['graph_data'])
                
                # Premium Chart Styling
                fig = px.line(chart_data, x="price", y="revenue", 
                              labels={"price": "Price Point ($)", "revenue": "Revenue ($)"})
                
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    font=dict(color="#9ca3af", family="Inter"),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    hovermode="x unified"
                )
                
                # Gradient Line Effect (simulated with solid color for now)
                fig.update_traces(line_color="#06b6d4", line_width=4)
                
                # Optimal Point Marker
                fig.add_scatter(x=[data['suggested_price']], y=[data['predicted_revenue']], 
                                mode='markers', 
                                marker=dict(size=20, color='#8b5cf6', line=dict(width=3, color='white')), 
                                name='Optimal Strategy')
                
                # Graph Container
                st.markdown("""
                <div style="background: rgba(20,20,20,0.6); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); padding: 20px;">
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 4. Strategy Insight
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, rgba(8,8,8,0.8) 0%, rgba(20,20,20,0.8) 100%); 
                            border-left: 4px solid #06b6d4; padding: 20px; border-radius: 8px;">
                    <strong style="color: #06b6d4;">STRATEGY INSIGHT //</strong> 
                    The algorithm has identified <strong>${data['suggested_price']}</strong> as the optimal price floor. 
                    This position effectively undercuts the market median while retaining sufficient margin depth to maximize aggregate revenue.
                </div>
                """, unsafe_allow_html=True)
            except ValueError:
                st.error("Error decoding response from API. Detailed error logged.")
        else:
             st.error(f"API returned status code {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("❌ CRITICAL ERROR: Neural Core Disconnected. Please ensure `api.py` is running.")
    except requests.exceptions.Timeout:
        st.error("⚠️ TIMEOUT: The optimization took too long to respond.")
else:
    # Initial State / Landing View
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 50vh; opacity: 0.5;">
        <h3 style="color: var(--text-secondary);">AWAITING INPUT</h3>
        <p>Configure parameters on the left to initiate simulation.</p>
    </div>
    """, unsafe_allow_html=True)