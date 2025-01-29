from dash import Dash, html, dcc, Input, Output
import numpy as np
import plotly.graph_objs as go
from scipy.integrate import solve_ivp
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Constants
MASS = 10
DAMPING = 20
STIFFNESS = 50

def msd_model(t, state, F, m, b, k, distortion_amplitude=5, distortion_frequency=1):
    x, v = state
    dxdt = v
    # sinusoidal_distortion = distortion_amplitude * np.sin(2 * np.pi * distortion_frequency * t)
    sinusoidal_distortion = 0
    dvdt = (F + sinusoidal_distortion - b * v - k * x) / m

    return [dxdt, dvdt]

def pid_controller(error, integral, derivative, Kp, Ti, Td):
    return Kp * error + Ti * integral + Td * derivative # zwraca Kp * (e + /Ti + Td*) wystawiamy Ti>>1 i Td i wyłączyć zakłócenia + dodać zmianę Tp zamiast obecnych

def fuzzy_controller(error, derivative):
    error_universe = np.linspace(-10, 10, 100)
    derivative_universe = np.linspace(-5, 5, 100)
    force_universe = np.linspace(-20, 20, 100)

    error_var = ctrl.Antecedent(error_universe, 'error')
    derivative_var = ctrl.Antecedent(derivative_universe, 'derivative')
    force_var = ctrl.Consequent(force_universe, 'force')

    scale_factor_error = 1

    error_var["NB"] = fuzz.trimf(error_universe, [-2, -1.5, -0.5 ])*scale_factor_error
    error_var["NM"] = fuzz.trimf(error_universe, [-1.5, -0.5, 0])*scale_factor_error
    error_var["Z"] = fuzz.trimf(error_universe, [-0.5, 0, 0.5])*scale_factor_error
    error_var["PM"] = fuzz.trimf(error_universe, [0, 0.5, 1.5])*scale_factor_error
    error_var["PB"] = fuzz.trimf(error_universe, [0.5, 1.5, 2])*scale_factor_error
    
    scale_factor_derivative = 1

    derivative_var["NB"] = fuzz.trimf(derivative_universe, [-2, -1.5, -0.5])*scale_factor_derivative
    derivative_var["NM"] = fuzz.trimf(derivative_universe, [-1.5, -0.5, 0])*scale_factor_derivative
    derivative_var["Z"] = fuzz.trimf(derivative_universe, [-0.5, 0, 0.5])*scale_factor_derivative
    derivative_var["PM"] = fuzz.trimf(derivative_universe, [0, 0.5, 1.5])*scale_factor_derivative
    derivative_var["PB"] = fuzz.trimf(derivative_universe, [0.5, 1.5, 2])*scale_factor_derivative
    
    scale_factor_force = 1
    additional = 4

    force_var["NB"] = fuzz.trimf(force_universe, [9, 9+additional, 12+additional])*scale_factor_force
    force_var["NM"] = fuzz.trimf(force_universe, [9+additional, 12+additional, 15+additional])*scale_factor_force
    force_var["Z"] = fuzz.trimf(force_universe, [15+additional, 15+additional, 15+additional])*scale_factor_force
    force_var["PM"] = fuzz.trimf(force_universe, [10+additional, 12+additional, 14+additional])*scale_factor_force
    force_var["PB"] = fuzz.trimf(force_universe, [12+additional, 14+additional, 16+additional])*scale_factor_force

    rules = [
        ctrl.Rule(error_var['NB'] & derivative_var['NB'], force_var['PB']),
        ctrl.Rule(error_var['NB'] & derivative_var['Z'], force_var['PM']),
        ctrl.Rule(error_var['Z'] & derivative_var['Z'], force_var['Z']),
        ctrl.Rule(error_var['PB'] & derivative_var['Z'], force_var['NM']),
        ctrl.Rule(error_var['PB'] & derivative_var['PB'], force_var['NB']),
    ]

    system = ctrl.ControlSystem(rules)
    controller = ctrl.ControlSystemSimulation(system)

    controller.input['error'] = error
    controller.input['derivative'] = derivative
    controller.compute()

    return controller.output['force']

def simulate_msd(control_type, Kp=10, Ti=1, Td=0.1, setpoint=1.0, duration=10, dt=0.01,
                 distortion_amplitude=5, distortion_frequency=1):
    times = np.arange(0, duration, dt)
    x_vals, v_vals, force_vals, error_vals = [], [], [], []
    x, v = 0, 0
    integral = 0
    prev_error = setpoint - x

    for t in times:
        error = setpoint - x
        integral += error * dt # zamiana
        derivative = (error - prev_error) / dt

        force = pid_controller(error, integral, derivative, Kp, Ti, Td) if control_type == 'PID' else fuzzy_controller(
            error, derivative)
        prev_error = error

        sol = solve_ivp(msd_model, [t, t + dt], [x, v],
                        args=(force, MASS, DAMPING, STIFFNESS, distortion_amplitude, distortion_frequency),
                        t_eval=[t + dt])
        x, v = sol.y[:, 0]

        x_vals.append(x)
        v_vals.append(v)
        force_vals.append(force)
        error_vals.append(error)

    return times, x_vals, force_vals, error_vals

app = Dash(__name__)

# Define global styles
app.layout = html.Div([
    html.H1("Regulator PID i Fuzzy dla układu MSD", style={"textAlign": "center", "fontFamily": "Arial, sans-serif"}),

    dcc.Tabs(id="tabs", value="PID", children=[
        dcc.Tab(label="PID Controller", value="PID", children=[
            html.Div([
                html.Label("Kp:", style={"marginRight": "10px"}),
                dcc.Input(id="kp-input", type="number", value=10, step=0.1, style={"marginBottom": "10px"}),
                html.Label("Ti:", style={"marginRight": "10px"}),
                dcc.Input(id="ki-input", type="number", value=1, step=0.1, style={"marginBottom": "10px"}),
                html.Label("Td:", style={"marginRight": "10px"}),
                dcc.Input(id="kd-input", type="number", value=0.1, step=0.1, style={"marginBottom": "10px"}),
                html.Label("Setpoint:", style={"marginRight": "10px"}),
                dcc.Input(id="setpoint-input", type="number", value=1.0, step=0.1, style={"marginBottom": "10px"}),
                html.Label("Czas trwania [s]:", style={"marginRight": "10px"}),
                dcc.Input(id="duration-input", type="number", value=10, step=1, style={"marginBottom": "10px"}),
            ], style={"padding": "20px", "border": "1px solid #ccc", "borderRadius": "5px", "fontFamily": "Arial, sans-serif"})
        ]),

        dcc.Tab(label="Fuzzy Controller", value="Fuzzy", children=[
            html.Div([
                html.Label("Setpoint:", style={"marginRight": "10px"}),
                dcc.Input(id="fuzzy-setpoint-input", type="number", value=1.0, step=0.1, style={"marginBottom": "10px"}),
                html.Label("Czas trwania [s]:", style={"marginRight": "10px"}),
                dcc.Input(id="fuzzy-duration-input", type="number", value=10, step=1, style={"marginBottom": "10px"}),
            ], style={"padding": "20px", "border": "1px solid #ccc", "borderRadius": "5px", "fontFamily": "Arial, sans-serif"})
        ])
    ]),

    html.Div([
        dcc.Loading(
            id="loading-position-error",
            type="default",
            children=dcc.Graph(id="position-error-graph")
        ),
        dcc.Loading(
            id="loading-force",
            type="default",
            children=dcc.Graph(id="force-graph")
        ),
    ])
], style={"fontFamily": "Arial, sans-serif", "margin": "20px"})

@app.callback(
    [Output("position-error-graph", "figure"),
     Output("force-graph", "figure")],
    [Input("tabs", "value"),
     Input("kp-input", "value"), Input("ki-input", "value"), Input("kd-input", "value"),
     Input("setpoint-input", "value"), Input("duration-input", "value"),
     Input("fuzzy-setpoint-input", "value"), Input("fuzzy-duration-input", "value")]
)
def update_graphs(control_type, kp, ki, kd, pid_setpoint, pid_duration, fuzzy_setpoint, fuzzy_duration):
    if control_type == "PID":
        times, x_vals, force_vals, error_vals = simulate_msd(control_type, kp, ki, kd, pid_setpoint, pid_duration)
    else:
        times, x_vals, force_vals, error_vals = simulate_msd(control_type, setpoint=fuzzy_setpoint,
                                                             duration=fuzzy_duration)

    position_error_fig = go.Figure()
    position_error_fig.add_trace(go.Scatter(x=times, y=x_vals, mode="lines", name="Położenie (x)"))
    position_error_fig.add_trace(go.Scatter(x=times, y=error_vals, mode="lines", name="Błąd sterowania"))
    position_error_fig.update_layout(title="Położenie (x) i Błąd sterowania", xaxis_title="Czas (s)", yaxis_title="Wartość [m]")

    force_fig = go.Figure()
    force_fig.add_trace(go.Scatter(x=times, y=force_vals, mode="lines", name="Siła sterująca (F)"))
    force_fig.update_layout(title="Siła sterująca (F)", xaxis_title="Czas (s)", yaxis_title="Siła [N]")

    return position_error_fig, force_fig

if __name__ == "__main__":
    app.run_server(debug=True)
