from dash import Dash, html, dcc, Input, Output, callback_context
import numpy as np
import plotly.graph_objs as go
from scipy.integrate import solve_ivp
import simpful as sf

# System parameters
MASS = 10  # Mass [kg]
DAMPING = 20  # Damping coefficient [Ns/m]
STIFFNESS = 100  # Stiffness [N/m]


# Model of the mass-spring-damper system
def msd_model(t, state, F, m, b, k):
    x, v = state  # x - displacement, v - velocity
    dxdt = v
    dvdt = (F - b * v - k * x) / m
    return [dxdt, dvdt]


# PID controller
def pid_controller(error, integral, derivative, Kp, Ki, Kd):
    return Kp * error + Ki * integral + Kd * derivative


# Fuzzy PID controller setup
def fuzzy_pid_controller(error, delta_error):
    FS = sf.FuzzySystem()
    FS.add_linguistic_variable("Error", sf.LinguisticVariable([
        sf.FuzzySet(points=[[0, 1], [1, 0]], term="Small"),
        sf.FuzzySet(points=[[0.5, 0], [1, 1.0], [1.5, 0]], term="Medium"),
        sf.FuzzySet(points=[[1, 0], [2, 1.0]], term="Large")
    ]))
    FS.add_linguistic_variable("DeltaError", sf.LinguisticVariable([
        sf.FuzzySet(points=[[0, 1], [1, 0]], term="Small"),
        sf.FuzzySet(points=[[0.5, 0], [1, 1.0], [1.5, 0]], term="Medium"),
        sf.FuzzySet(points=[[1, 0], [2, 1.0]], term="Large")
    ]))
    FS.add_linguistic_variable("Output", sf.LinguisticVariable([
        sf.FuzzySet(points=[[0, 1], [1, 0]], term="Small"),
        sf.FuzzySet(points=[[0.5, 0], [1, 1.0], [1.5, 0]], term="Medium"),
        sf.FuzzySet(points=[[1, 0], [2, 1.0]], term="Large")
    ]))
    FS.add_rules([
        "IF (Error IS Small) AND (DeltaError IS Small) THEN (Output IS Small)",
        "IF (Error IS Medium) THEN (Output IS Medium)",
        "IF (Error IS Large) OR (DeltaError IS Large) THEN (Output IS Large)"
    ])
    FS.set_variable("Error", error)
    FS.set_variable("DeltaError", delta_error)
    return FS.inference()["Output"]


# Simulation function
def simulate_msd(Kp, Ki, Kd, fuzzy=False, setpoint=1.0, duration=10, dt=0.01):
    times = np.arange(0, duration, dt)
    x_vals, force_vals, error_vals = [], [], []
    x, v = 0, 0  # Initial conditions
    integral = 0
    prev_error = setpoint - x

    for t in times:
        error = setpoint - x
        delta_error = error - prev_error
        integral += error * dt
        if fuzzy:
            force = fuzzy_pid_controller(error, delta_error)
        else:
            derivative = delta_error / dt
            force = pid_controller(error, integral, derivative, Kp, Ki, Kd)
        prev_error = error
        sol = solve_ivp(msd_model, [t, t + dt], [x, v], args=(force, MASS, DAMPING, STIFFNESS), t_eval=[t + dt])
        x, v = sol.y[:, 0]
        x_vals.append(x)
        force_vals.append(force)
        error_vals.append(error)
    return times, x_vals, force_vals, error_vals


# Create Dash application
app = Dash(__name__)
app.title = "PID and Fuzzy PID Control"

app.layout = html.Div([
    html.H1("Mass-Spring-Damper System Control", className="header"),
    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="PID Control", value="tab-1", className="tab"),
        dcc.Tab(label="Fuzzy PID Control", value="tab-2", className="tab"),
    ]),
    html.Div(id="control-panel", className="panel"),
    dcc.Graph(id="msd-graph", className="graph"),
], className="container")


@app.callback(
    [Output("control-panel", "children"), Output("msd-graph", "figure")],
    [Input("tabs", "value")],
)
def update_panel_and_graph(tab):  # Only one argument, the tab value, is passed
    ctx = callback_context
    if tab == "tab-1":  # Regular PID control panel
        return [
            html.Div([
                html.Label("Kp:"),
                dcc.Input(id="kp-input", type="number", value=10, step=0.1),
                html.Label("Ki:"),
                dcc.Input(id="ki-input", type="number", value=1, step=0.1),
                html.Label("Kd:"),
                dcc.Input(id="kd-input", type="number", value=0.1, step=0.1),
                html.Label("Setpoint:"),
                dcc.Input(id="setpoint-input", type="number", value=1.0, step=0.1),
                html.Label("Duration (seconds):"),
                dcc.Input(id="duration-input", type="number", value=10, step=1),
                html.Button("Simulate", id="simulate-pid", n_clicks=0),
            ], className="control-panel"),
            go.Figure(),  # Placeholder figure until simulation is run
        ]
    elif tab == "tab-2":  # Fuzzy PID control panel
        return [
            html.Div([
                html.Label("Setpoint:"),
                dcc.Input(id="setpoint-input-fuzzy", type="number", value=1.0, step=0.1),
                html.Label("Duration (seconds):"),
                dcc.Input(id="duration-input-fuzzy", type="number", value=10, step=1),
                html.Button("Simulate", id="simulate-fuzzy", n_clicks=0),
            ], className="control-panel"),
            go.Figure(),  # Placeholder figure for the fuzzy PID tab
        ]

    # Update the graph (this callback runs when control inputs or simulation buttons are clicked)
    if tab == "tab-1":
        kp = float(ctx.inputs["kp-input.value"])
        ki = float(ctx.inputs["ki-input.value"])
        kd = float(ctx.inputs["kd-input.value"])
        setpoint = float(ctx.inputs["setpoint-input.value"])
        duration = float(ctx.inputs["duration-input.value"])
        times, x_vals, force_vals, error_vals = simulate_msd(kp, ki, kd, setpoint=setpoint, duration=duration)

    elif tab == "tab-2":
        setpoint = float(ctx.inputs["setpoint-input-fuzzy.value"])
        duration = float(ctx.inputs["duration-input-fuzzy.value"])
        times, x_vals, force_vals, error_vals = simulate_msd(0, 0, 0, fuzzy=True, setpoint=setpoint, duration=duration)

    # Create the graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=x_vals, mode="lines", name="Position (x)"))
    fig.add_trace(go.Scatter(x=times, y=force_vals, mode="lines", name="Control Force (F)"))
    fig.add_trace(go.Scatter(x=times, y=error_vals, mode="lines", name="Error (e)"))
    fig.update_layout(
        title="Mass-Spring-Damper System Simulation",
        xaxis_title="Time (s)",
        yaxis_title="Values",
        legend_title="Parameters",
    )
    return html.Div(), fig


# Add styles for the app
app.css.append_css({
    "external_url": [
        "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
    ]
})
app.layout.className = "container-fluid"

if __name__ == "__main__":
    app.run_server(debug=True)
