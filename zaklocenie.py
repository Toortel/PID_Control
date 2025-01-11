from dash import Dash, html, dcc, Input, Output
import numpy as np
import plotly.graph_objs as go
from scipy.integrate import solve_ivp
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Constants
MASS = 10
DAMPING = 20
STIFFNESS = 100


def msd_model(t, state, F, m, b, k, distortion_amplitude=5, distortion_frequency=1):
    x, v = state
    dxdt = v
    # Add a sinusoidal distortion
    sinusoidal_distortion = distortion_amplitude * np.sin(2 * np.pi * distortion_frequency * t)
    dvdt = (F + sinusoidal_distortion - b * v - k * x) / m
    return [dxdt, dvdt]



def pid_controller(error, integral, derivative, Kp, Ki, Kd):
    return Kp * error + Ki * integral + Kd * derivative


def fuzzy_controller(error, derivative):
    error_universe = np.linspace(-2, 2, 100)
    derivative_universe = np.linspace(-2, 2, 100)
    force_universe = np.linspace(-10, 10, 100)

    error_var = ctrl.Antecedent(error_universe, 'error')
    derivative_var = ctrl.Antecedent(derivative_universe, 'derivative')
    force_var = ctrl.Consequent(force_universe, 'force')

    error_var["NB"] = fuzz.trimf(error_universe, [-2, -2, -1])
    error_var["NM"] = fuzz.trimf(error_universe, [-2, -1, 0])
    error_var["Z"] = fuzz.trimf(error_universe, [-1, 0, 1])
    error_var["PM"] = fuzz.trimf(error_universe, [0, 1, 2])
    error_var["PB"] = fuzz.trimf(error_universe, [1, 2, 2])

    derivative_var["NB"] = fuzz.trimf(derivative_universe, [-2, -2, -1])
    derivative_var["NM"] = fuzz.trimf(derivative_universe, [-2, -1, 0])
    derivative_var["Z"] = fuzz.trimf(derivative_universe, [-1, 0, 1])
    derivative_var["PM"] = fuzz.trimf(derivative_universe, [0, 1, 2])
    derivative_var["PB"] = fuzz.trimf(derivative_universe, [1, 2, 2])

    force_var["NB"] = fuzz.trimf(force_universe, [-10, -10, -5])
    force_var["NM"] = fuzz.trimf(force_universe, [-10, -5, 0])
    force_var["Z"] = fuzz.trimf(force_universe, [-5, 0, 5])
    force_var["PM"] = fuzz.trimf(force_universe, [0, 5, 10])
    force_var["PB"] = fuzz.trimf(force_universe, [5, 10, 10])

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


def simulate_msd(control_type, Kp=10, Ki=1, Kd=0.1, setpoint=1.0, duration=10, dt=0.01,
                 distortion_amplitude=5, distortion_frequency=1):
    times = np.arange(0, duration, dt)
    x_vals, v_vals, force_vals, error_vals = [], [], [], []
    x, v = 0, 0
    integral = 0
    prev_error = setpoint - x

    for t in times:
        error = setpoint - x
        integral += error * dt
        derivative = (error - prev_error) / dt

        force = pid_controller(error, integral, derivative, Kp, Ki, Kd) if control_type == 'PID' else fuzzy_controller(
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

app.layout = html.Div([
    html.H1("Regulator PID i Fuzzy dla układu MSD"),

    dcc.Tabs(id="tabs", value="PID", children=[
        dcc.Tab(label="PID Controller", value="PID", children=[
            html.Div([
                html.Label("Kp:"), dcc.Input(id="kp-input", type="number", value=10, step=0.1),
                html.Label("Ki:"), dcc.Input(id="ki-input", type="number", value=1, step=0.1),
                html.Label("Kd:"), dcc.Input(id="kd-input", type="number", value=0.1, step=0.1),
                html.Label("Setpoint:"), dcc.Input(id="setpoint-input", type="number", value=1.0, step=0.1),
                html.Label("Czas trwania [s]"), dcc.Input(id="duration-input", type="number", value=10, step=1),
            ])
        ]),

        dcc.Tab(label="Fuzzy Controller", value="Fuzzy", children=[
            html.Div([
                html.Label("Setpoint:"), dcc.Input(id="fuzzy-setpoint-input", type="number", value=1.0, step=0.1),
                html.Label("Czas trwania [s]"), dcc.Input(id="fuzzy-duration-input", type="number", value=10, step=1),
            ])
        ])
    ]),

    dcc.Graph(id="msd-graph"),
])


@app.callback(
    Output("msd-graph", "figure"),
    [Input("tabs", "value"),
     Input("kp-input", "value"), Input("ki-input", "value"), Input("kd-input", "value"),
     Input("setpoint-input", "value"), Input("duration-input", "value"),
     Input("fuzzy-setpoint-input", "value"), Input("fuzzy-duration-input", "value")]
)
def update_graph(control_type, kp, ki, kd, pid_setpoint, pid_duration, fuzzy_setpoint, fuzzy_duration):
    if control_type == "PID":
        times, x_vals, force_vals, error_vals = simulate_msd(control_type, kp, ki, kd, pid_setpoint, pid_duration)
    else:
        times, x_vals, force_vals, error_vals = simulate_msd(control_type, setpoint=fuzzy_setpoint,
                                                             duration=fuzzy_duration)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=x_vals, mode="lines", name="Położenie (x)"))
    fig.add_trace(go.Scatter(x=times, y=force_vals, mode="lines", name="Siła sterująca (F)"))
    fig.add_trace(go.Scatter(x=times, y=error_vals, mode="lines", name="Błąd sterowania"))

    fig.update_layout(title=f"Sterowanie: {control_type}", xaxis_title="Czas (s)", yaxis_title="Wartości")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
