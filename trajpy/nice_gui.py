import base64
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from nicegui import ui
from PIL import Image

import trajpy
import trajpy.trajpy as tj

# Konfigurasjon / initialisering
FEATURES = [
    'Anomalous Exponent', 'MSD Ratio', 'Fractal dimension',
    'Anisotropy & Kurtosis', 'Straightness', 'Efficiency',
    'Gaussianity', 'Diffusivity', 'Confinement Prob.'
]

# Tilstand
state = {
    'uploaded_files': [],       # liste av (filename, bytes)
    'trajectories': [],        # liste av tj.Trajectory-objekter
    'selected_features': set(),
    'results': {},
    'last_saved_path': None
}

def load_trajectories_from_uploads(uploads):
    trs = []
    for idx, (name, content) in enumerate(uploads):
        lower = name.lower()
        if lower.endswith('.yaml') or lower.endswith('.yml'):
            from trajpy.auxiliar_functions import parse_lammps_dump_yaml
            try:
                positions = parse_lammps_dump_yaml(BytesIO(content))
            except Exception:
                # Use unique temporary filename with index
                tmp = f'/tmp/{idx}_{name}'
                with open(tmp, 'wb') as f:
                    f.write(content)
                positions = parse_lammps_dump_yaml(tmp)
                os.remove(tmp)
            num_atoms = positions.shape[1]
            for atom_idx in range(num_atoms):
                atom_trajectory = positions[:, atom_idx, :]
                trs.append(tj.Trajectory(atom_trajectory))
        else:
            # Use unique temporary filename with index
            tmp = f'/tmp/{idx}_{name}'
            with open(tmp, 'wb') as f:
                f.write(content)
            try:
                trs.append(tj.Trajectory(tmp, skip_header=1, delimiter=','))
            except Exception as e:
                print(f"Error loading {name}: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    return trs





def plot_trajectories():
    if not state['trajectories']:
        result_box.set_text("Ingen trajectories å plotte")
        return

    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for n, trajectory in enumerate(state['trajectories']):
        r = trajectory._r
        if r.shape[1] >= 2:  # 2D or 3D trajectory
            ax.plot(r[:, 0], r[:, 1], alpha=0.6, label=f'Trajectory {n + 1}')
            ax.scatter(r[0, 0], r[0, 1], marker='o', s=100, label=f'Start {n + 1}')
            ax.scatter(r[-1, 0], r[-1, 1], marker='s', s=100, label=f'End {n + 1}')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save to BytesIO and convert to base64 data URI
    bio = BytesIO()
    fig.savefig(bio, format='png', dpi=100, bbox_inches='tight')
    bio.seek(0)

    # Convert to base64 data URI
    img_base64 = base64.b64encode(bio.read()).decode('utf-8')
    img_src = f'data:image/png;base64,{img_base64}'

    # Clear previous plot if exists
    plot_container.clear()
    with plot_container:
        ui.image(img_src).style('max-width: 30%')

    plt.close(fig)


def normalize_upload_event(event):
    """
    Normalize NiceGUI upload event into a list of (name, bytes).
    Works for different NiceGUI versions / shapes and for single or multiple files.
    """
    if event is None:
        return []

    normalized = []

    # NiceGUI passes event with a 'file' attribute (single file)
    if hasattr(event, 'file'):
        f = event.file
        name = f.name
        # Read the content from the _data attribute
        content = f._data if hasattr(f, '_data') else f.read()
        if hasattr(content, 'read'):
            content = content.read()
        normalized.append((name, content))
    # Handle multiple files if event has 'files' attribute
    elif hasattr(event, 'files'):
        for f in event.files:
            name = f.name
            content = f._data if hasattr(f, '_data') else f.read()
            if hasattr(content, 'read'):
                content = content.read()
            normalized.append((name, content))
    else:
        raise RuntimeError(f'Could not parse upload event: {repr(event)}')

    return normalized

# Example handler using the normalizer
def handle_upload(event):
    try:
        uploaded = normalize_upload_event(event)   # -> list of (name, bytes)
        state['uploaded_files'] = uploaded
        result_box.set_text(f"{len(uploaded)} file(s) uploaded")
        state['trajectories'] = load_trajectories_from_uploads(state['uploaded_files'])
        result_box.set_text(f"{len(state['trajectories'])} trajectory(ies) loaded")
    except Exception as e:
        result_box.set_text(f"Error loading files: {e}")




def compute_selected():
    if not state['trajectories']:
        result_box.set_text("Ingen trajectories lastet inn")
        return

    selected = [feat for feat, cb in checkboxes.items() if cb.value]
    state['results'] = {}
    for n, trajectory in enumerate(state['trajectories']):
        r = trajectory
        state['results'][n] = {}

        if any('Anomalous' in feature for feature in selected):
            r.msd_ea = r.msd_ensemble_averaged_(r._r)
            r.anomalous_exponent = r.anomalous_exponent_(r.msd_ea, r._t)
            state['results'][n]['alpha'] = r.anomalous_exponent

        if any('MSD' in feature for feature in selected):
            r.msd_ta = r.msd_time_averaged_(r._r, np.arange(len(r._r)))
            r.msd_ratio = r.msd_ratio_(r.msd_ta, n1=2, n2=10)
            state['results'][n]['msd_ratio'] = r.msd_ratio

        if any('Fractal' in feature for feature in selected):
            r.fractal_dimension, r._r0 = r.fractal_dimension_(r._r)
            state['results'][n]['df'] = r.fractal_dimension

        if any(item in feature for feature in selected for item in ['Kurtosis', 'Anisotropy']):
            gyration_radius_dict = r.gyration_radius_(r._r)
            r.gyration_radius = gyration_radius_dict['gyration tensor']
            r.eigenvalues = gyration_radius_dict['eigenvalues']
            r.eigenvectors = gyration_radius_dict['eigenvectors']
            r.kurtosis = r.kurtosis_(r._r, r.eigenvectors[:,0])
            r.anisotropy = r.anisotropy_(r.eigenvalues)
            state['results'][n]['anisotropy'] = r.anisotropy
            state['results'][n]['kurtosis'] = r.kurtosis

        if any('Gaussianity' in feature for feature in selected):
            r.gaussianity = r.gaussianity_(r._r)
            state['results'][n]['gaussianity'] = r.gaussianity

        if any('Straightness' in feature for feature in selected):
            r.straightness = r.straightness_(r._r)
            state['results'][n]['straightness'] = r.straightness

        if any('Efficiency' in feature for feature in selected):
            r.efficiency = r.efficiency_(r._r)
            state['results'][n]['efficiency'] = r.efficiency

        if any('Diffusivity' in feature for feature in selected):
            r.velocity = r.velocity_(r._r, r._t)
            r.vacf = r.stationary_velocity_correlation_(r.velocity, r._t, np.arange(int(len(r.velocity))))
            r.diffusivity = r.green_kubo_(r.velocity, r._t, r.vacf)
            state['results'][n]['diffusivity'] = r.diffusivity

        if any('Confinement' in feature for feature in selected):
            if not hasattr(r, 'diffusivity'):
                r.velocity = r.velocity_(r._r, r._t)
                r.vacf = r.stationary_velocity_correlation_(r.velocity, r._t, np.arange(int(len(r.velocity))))
                r.diffusivity = r.green_kubo_(r.velocity, r._t, r.vacf)
            if not hasattr(r, '_r0'):
                r.fractal_dimension, r._r0 = r.fractal_dimension_(r._r)
            r.confinement_probability = r.confinement_probability_(r._r0, r.diffusivity, r._t[-1])
            state['results'][n]['confinement'] = r.confinement_probability

    if state['results']:
        first = state['results'].get(0, {})
        lines = [f"{k}: {v}" for k, v in first.items()]
        result_box.set_text("Resultater (første trajectory):\n" + "\n".join(lines))
        save_btn.props(remove='disabled')  # Enable the button
    else:
        result_box.set_text("Ingen resultater produsert")


def save_results():
    if not state['results']:
        result_box.set_text("No results to save")
        return

    # Create CSV from state['results']
    header = []
    rows = []
    for n in sorted(state['results'].keys()):
        entries = state['results'][n]
        if not header:
            header = list(entries.keys())
        rows.append([entries.get(h, '') for h in header])

    csv_lines = [','.join(header)]
    for row in rows:
        csv_lines.append(','.join(map(str, row)))
    csv_data = '\n'.join(csv_lines)

    # Create download link
    csv_bytes = csv_data.encode('utf-8')
    csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')

    # Trigger download using JavaScript
    ui.run_javascript(f'''
        const link = document.createElement('a');
        link.href = 'data:text/csv;base64,{csv_base64}';
        link.download = 'trajpy_results.csv';
        link.click();
    ''')

    result_box.set_text('Results downloaded as trajpy_results.csv')


# About-dialog
def show_about():
    with ui.dialog() as dialog, ui.card():
        ui.label('TrajPy GUI').style('font-weight: bold; font-size: 18px')
        # Try to show logo.png from same folder if it exists
        logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
        if os.path.exists(logo_path):
            img = Image.open(logo_path)
            img.thumbnail((400, 150))
            bio = BytesIO()
            img.save(bio, format='PNG')
            bio.seek(0)

            # Convert to base64 data URI
            img_base64 = base64.b64encode(bio.read()).decode('utf-8')
            img_src = f'data:image/png;base64,{img_base64}'
            ui.image(img_src).style('max-width: 100%')

        ui.label('Developed by Maurício Moreira-Soares')
        ui.link('phydev.github.io', 'https://phydev.github.io')
        ui.label('trajpy@protonmail.ch')
        ui.button('Close', on_click=lambda: dialog.close())
    dialog.open()


# UI-komponenter
ui.markdown(f"# TrajPy GUI — versjon {trajpy.__version__}")


with ui.row():
    upload = ui.upload(label='Upload one or several files (CSV or YAML)', multiple=True,
                       on_upload=handle_upload)
upload.on_upload(handle_upload)
with ui.row():
    with ui.card().tight():
        ui.label('Velg features').style('font-weight: bold')
        checkboxes = {}
        for feat in FEATURES:
            cb = ui.checkbox(feat, value=False)  # ingen on_change her
            checkboxes[feat] = cb

with ui.row():
    about_btn = ui.button('About', on_click=lambda: show_about())

with ui.row():
    compute_btn = ui.button('Compute!', on_click=lambda: compute_selected())
    plot_btn = ui.button('Plot Trajectories', on_click=lambda: plot_trajectories())
    save_btn = ui.button('Save results (CSV)', on_click=lambda: save_results()).props('disabled')
    result_box = ui.label('No results yet')

# Add plot container
plot_container = ui.column().style('width: 100%')


# Start NiceGUI
if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
