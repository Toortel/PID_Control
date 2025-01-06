from dash import Dash, html, dcc, Input, Output
import numpy as np
import plotly.graph_objs as go
from scipy.integrate import solve_ivp

# Parametry obiektu
MASS = 10      # Masa [kg]
DAMPING = 20   # Współczynnik tłumienia [Ns/m]
STIFFNESS = 100  # Sprężystość [N/m]

# Model obiektu
def msd_model(t, state, F, m, b, k):
    x, v = state  # x - położenie, v - prędkość
    dxdt = v
    dvdt = (F - b * v - k * x) / m
    return [dxdt, dvdt]

# Regulator PID
def pid_controller(error, integral, derivative, Kp, Ki, Kd):
    return Kp * error + Ki * integral + Kd * derivative

# Funkcja symulacji
def simulate_msd(Kp, Ki, Kd, setpoint=1.0, duration=1000, dt=0.01):
    times = np.arange(0, duration, dt)
    x_vals = []
    v_vals = []
    force_vals = []
    error_vals = []
    
    x, v = 0, 0  # Początkowe warunki
    integral = 0
    prev_error = setpoint - x

    for t in times:
        error = setpoint - x
        integral += error * dt
        derivative = (error - prev_error) / dt
        force = pid_controller(error, integral, derivative, Kp, Ki, Kd)
        prev_error = error
        
        sol = solve_ivp(msd_model, [t, t + dt], [x, v], args=(force, MASS, DAMPING, STIFFNESS), t_eval=[t + dt])
        x, v = sol.y[:, 0]
        
        x_vals.append(x)
        v_vals.append(v)
        force_vals.append(force)
        error_vals.append(error)
    
    return times, x_vals, force_vals, error_vals

# Tworzenie aplikacji Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Regulator PID dla układu masy, sprężyny i tłumika"),
    html.Div([
        html.Label("Kp:"),
        dcc.Input(id="kp-input", type="number", value=10, step=0.1),
        html.Label("Ki:"),
        dcc.Input(id="ki-input", type="number", value=1, step=0.1),
        html.Label("Kd:"),
        dcc.Input(id="kd-input", type="number", value=0.1, step=0.1),
        html.Label("Setpoint:"),
        dcc.Input(id="setpoint-input", type="number", value=1.0, step=0.1),
        html.Label("Czas trwania [s]:"),
        dcc.Input(id="duration-input", type="number", value=10, step=1),
    ], style={"margin-bottom": "20px"}),
    dcc.Graph(id="msd-graph"),
])

@app.callback(
    Output("msd-graph", "figure"),
    [
        Input("kp-input", "value"),
        Input("ki-input", "value"),
        Input("kd-input", "value"),
        Input("setpoint-input", "value"),
        Input("duration-input", "value"),
    ],
)
def update_graph(kp, ki, kd, setpoint, duration):
    times, x_vals, force_vals, error_vals = simulate_msd(kp, ki, kd, setpoint, duration)
    
    # Tworzenie wykresów
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=x_vals, mode="lines", name="Położenie (x)"))
    fig.add_trace(go.Scatter(x=times, y=force_vals, mode="lines", name="Siła sterująca (F)"))
    fig.add_trace(go.Scatter(x=times, y=error_vals, mode="lines", name="Błąd regulacji (e)"))
    
    fig.update_layout(
        title="Symulacja układu MSD z regulatorem PID",
        xaxis_title="Czas [s]",
        yaxis_title="Wartości",
        legend_title="Parametry",
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
