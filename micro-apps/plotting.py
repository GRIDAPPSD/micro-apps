#%%
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import time
#%%
def save_pdf(fig, name):
    go.Figure().write_image("dummy.pdf")
    time.sleep(1)
    fig.write_image(name, format="pdf", width=400, height=400, scale=1, validate=True)


def create_var_plot(primary_path, base_path):
    base_load = pd.read_csv(base_path)
    base_load.p = base_load.p/1e6
    base_load.q = base_load.q/1e6
    base_load.time = pd.to_datetime(base_load['time'],origin="unix", unit='s')
    load = pd.read_csv(primary_path)
    load.p = load.p/1e6
    load.q = load.q/1e6
    load.time = pd.to_datetime(load['time'],origin="unix", unit='s')
    load["base"] = base_load.q
    base_case = "base"
    primary_case = "q"
    # fig = px.line(load, x="time", y=["control"], facet_row="phase", labels={"p": "Active Power", "q": "Reactive Power"} )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", base_case],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="no control",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", primary_case],
            name="controlled",
            line=dict(color="blue", width=0.9),
            showlegend=True,
            legendgroup="controlled",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", base_case],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            legendgroup="no control",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", primary_case],
            name="controlled",
            line=dict(color="blue", width=0.9),
            showlegend=False,
            legendgroup="controlled",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", base_case],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            legendgroup="no control",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", primary_case],
            name="controlled",
            line=dict(color="blue", width=0.9),
            showlegend=False,
            legendgroup="controlled",
        ),
        row=3, col=1,
    )
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 20, "t": 25, "b": 10},
        # title_text='IEEE 123 Node Test Feeder',
        # title_y=0.98,
        # title_x=0.5,
        legend_title_text="",
        font_family="Times New Roman",
        font_size=10,
        font_color="black",
        legend_orientation="h",
        legend_xref="paper",
        legend_xanchor="center",
        legend_bgcolor="rgba(256,256,256,1)",
        legend_borderwidth=1,
        legend_bordercolor="rgba(0,0,0,1)",
        legend_x=0.5,
        legend_y=1.1,
        # legend_yref='container',
        # legend_y=-0.25
        xaxis3_title="Time",
        yaxis2_title="Feeder Reactive Load (MVAr)",
    )
    return fig


def create_power_plot(primary_path, base_path):
    base_load = pd.read_csv(base_path)
    base_load.p = base_load.p/1e6
    base_load.time = pd.to_datetime(base_load['time'],origin="unix", unit='s')
    if isinstance(primary_path, pd.DataFrame):
        load = pd.DataFrame(primary_path)
    else:
        load = pd.read_csv(primary_path)
    load.p = load.p/1e6
    load.time = pd.to_datetime(load['time'],origin="unix", unit='s')
    load["base"] = base_load.p
    # fig = px.line(load, x="time", y=["control"], facet_row="phase", labels={"p": "Active Power", "q": "Reactive Power"} )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", "base"],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="no control",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", "p"],
            name="controlled",
            line=dict(color="blue", width=0.9),
            showlegend=True,
            legendgroup="controlled",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", "base"],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            legendgroup="no control",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", "p"],
            name="controlled",
            line=dict(color="blue", width=0.9),
            showlegend=False,
            legendgroup="controlled",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", "base"],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            legendgroup="no control",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", "p"],
            name="controlled",
            line=dict(color="blue", width=0.9),
            showlegend=False,
            legendgroup="controlled",
        ),
        row=3, col=1,
    )
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 20, "t": 25, "b": 10},
        # title_text='IEEE 123 Node Test Feeder',
        # title_y=0.98,
        # title_x=0.5,
        legend_title_text="",
        font_family="Times New Roman",
        font_size=10,
        font_color="black",
        legend_orientation="h",
        legend_xref="paper",
        legend_xanchor="center",
        legend_bgcolor="rgba(256,256,256,1)",
        legend_borderwidth=1,
        legend_bordercolor="rgba(0,0,0,1)",
        legend_x=0.5,
        legend_y=1.1,
        # legend_yref='container',
        # legend_y=-0.25
        xaxis3_title="Time",
        yaxis2_title="Feeder Load (MW)",
    )
    return fig


def create_power_comparison_plot(base_path, peak_shaving_path, carbon_path):
    load = pd.read_csv(base_path)
    load.p = load.p/1e6
    load.time = pd.to_datetime(load['time'],origin="unix", unit='s')
    if isinstance(peak_shaving_path, pd.DataFrame):
        peak_shaving_load = pd.DataFrame(peak_shaving_path)
    else:
        peak_shaving_load = pd.read_csv(peak_shaving_path)
    if isinstance(carbon_path, pd.DataFrame):
        carbon_load = pd.DataFrame(carbon_path)
    else:
        carbon_load = pd.read_csv(carbon_path)
    carbon_load.time = pd.to_datetime(carbon_load['time'],origin="unix", unit='s')
    peak_shaving_load.time = pd.to_datetime(peak_shaving_load['time'],origin="unix", unit='s')
    load["peak"] = peak_shaving_load.p/1e6
    load["carbon"] = carbon_load.p/1e6
    load.time = pd.to_datetime(load['time'],origin="unix", unit='s')
    load["base"] = load.p
    # fig = px.line(load, x="time", y=["control"], facet_row="phase", labels={"p": "Active Power", "q": "Reactive Power"} )
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", "base"],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=True,
            legendgroup="no control",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", "peak"],
            name="Peak Shaving",
            line=dict(color="blue", width=0.9),
            showlegend=True,
            legendgroup="peak",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="A", "time"],
            y=load.loc[load.phase=="A", "carbon"],
            name="Carbon Management",
            line=dict(color="green", width=0.9),
            showlegend=True,
            legendgroup="carbon",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", "base"],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            legendgroup="no control",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", "peak"],
            name="Peak Shaving",
            line=dict(color="blue", width=0.9),
            showlegend=False,
            legendgroup="peak",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="B", "time"],
            y=load.loc[load.phase=="B", "carbon"],
            name="Carbon Management",
            line=dict(color="green", width=0.9),
            showlegend=False,
            legendgroup="carbon",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", "base"],
            name="no control",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            legendgroup="no control",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", "peak"],
            name="Peak Shaving",
            line=dict(color="blue", width=0.9),
            showlegend=False,
            legendgroup="peak",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=load.loc[load.phase=="C", "time"],
            y=load.loc[load.phase=="C", "carbon"],
            name="Carbon Management",
            line=dict(color="green", width=0.9),
            showlegend=False,
            legendgroup="carbon",
        ),
        row=3, col=1,
    )
    # fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 20, "t": 25, "b": 10},
        # title_text='IEEE 123 Node Test Feeder',
        # title_y=0.98,
        # title_x=0.5,
        legend_title_text="",
        font_family="Times New Roman",
        font_size=10,
        font_color="black",
        legend_orientation="h",
        legend_xref="paper",
        legend_xanchor="center",
        legend_bgcolor="rgba(256,256,256,1)",
        legend_borderwidth=1,
        legend_bordercolor="rgba(0,0,0,1)",
        legend_x=0.5,
        legend_y=1.1,
        # legend_yref='container',
        # legend_y=-0.25
        xaxis3_title="Time",
        yaxis2_title="Feeder Load (MW)",
    )
    return fig

def create_voltage_plot(path):
    v = pd.read_csv(path)
    v.time = pd.to_datetime(v['time'], origin="unix", unit='s')
    v_mean = v.groupby(by="time").voltage.aggregate("mean")
    v_min = v.groupby(by="time").voltage.aggregate("min")
    v_max = v.groupby(by="time").voltage.aggregate("max")
    v_summary = pd.DataFrame(index=v_mean.index)
    v_summary["mean"] = v_mean
    v_summary["min"] = v_min
    v_summary["max"] = v_max
    fig = px.line(v_summary, y=["mean", "min", "max"])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_yaxis(lambda a: a.update(
        # tickformat=".1f",
        dtick=0.05
    ))
    # fig.for_each_xaxis(lambda a: a.update(
    #     title_standoff=0.5,
    # ))
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 20, "t": 25, "b": 10},
        # title_text='IEEE 123 Node Test Feeder',
        # title_y=0.98,
        title_x=0.5,
        legend_title_text="",
        font_family="Times New Roman",
        font_color="black",
        legend_orientation="h",
        legend_xref="paper",
        legend_xanchor="right",
        legend_bgcolor="rgba(256,256,256,1)",
        legend_borderwidth=1,
        legend_bordercolor="rgba(0,0,0,1)",
        legend_x=0.95,
        legend_y=-0.2,
        # legend_yref='container',
        # legend_y=-0.25,
        yaxis_title_standoff=0.0,
        xaxis_title="Time",
        yaxis_title="Voltage (p.u.)",
    )
    fig.add_hline(y=0.95, line_dash='dot')
    fig.add_hline(y=1.05, line_dash='dot')
    return fig

def create_voltage_all_plot(path):
    v = pd.read_csv(path)
    v.time = pd.to_datetime(v['time'], origin="unix", unit='s')
    # v_mean = v.groupby(by="time").voltage.aggregate("mean")
    # v_min = v.groupby(by="time").voltage.aggregate("min")
    # v_max = v.groupby(by="time").voltage.aggregate("max")
    # v_summary = pd.DataFrame(index=v_mean.index)
    # v_summary["mean"] = v_mean
    # v_summary["min"] = v_min
    # v_summary["max"] = v_max
    fig = px.line(v, x="time", y="voltage", color="node", facet_col="phase")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_yaxis(lambda a: a.update(
        # tickformat=".1f",
        dtick=0.05
    ))
    # fig.for_each_xaxis(lambda a: a.update(
    #     title_standoff=0.5,
    # ))
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 20, "t": 25, "b": 10},
        # title_text='IEEE 123 Node Test Feeder',
        # title_y=0.98,
        title_x=0.5,
        legend_title_text="",
        font_family="Times New Roman",
        font_color="black",
        legend_orientation="h",
        legend_xref="paper",
        legend_xanchor="right",
        legend_bgcolor="rgba(256,256,256,1)",
        legend_borderwidth=1,
        legend_bordercolor="rgba(0,0,0,1)",
        legend_x=0.95,
        legend_y=-0.2,
        # legend_yref='container',
        # legend_y=-0.25,
        yaxis_title_standoff=0.0,
        xaxis_title="Time",
        yaxis_title="Voltage (p.u.)",
    )
    fig.add_hline(y=0.95, line_dash='dot')
    fig.add_hline(y=1.05, line_dash='dot')
    return fig


def create_batt_plot(path, units="kW"):
    batt = pd.read_csv(path)
    batt["p"] = batt.p_a + batt.p_b + batt.p_c
    batt.time = pd.to_datetime(batt['time'], origin="unix", unit='s')
    batt_sum = batt.groupby(by="time").p.aggregate("sum")
    batt_soc = batt.groupby(by="time").soc.aggregate("mean")
    batt_agg = pd.DataFrame(index=batt_sum.index)
    if units=="kW":
        batt_agg["p"] = batt_sum/1e3
    if units=="W":
        batt_agg["p"] = batt_sum/1e6

    batt_agg["soc"] = batt_soc
    # fig = px.line(batt, x="time", y=["p", "soc"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=batt_agg.index, y=batt_agg.p, name="Power"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=batt_agg.index, y=batt_agg.soc, name="SOC"),
        secondary_y=True,
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 20, "t": 25, "b": 10},
        # title_text='IEEE 123 Node Test Feeder',
        # title_y=0.98,
        font_size=10,
        title_x=0.5,
        legend_title_text="",
        font_family="Times New Roman",
        font_color="black",
        legend_orientation="h",
        legend_xref="paper",
        legend_xanchor="center",
        legend_bgcolor="rgba(256,256,256,1)",
        legend_borderwidth=1,
        legend_bordercolor="rgba(0,0,0,1)",
        legend_x=0.5,
        legend_y=1.15,
        # legend_yref='container',
        # legend_y=-0.25,
        # yaxis2_title_standoff=0.0,
        xaxis_title_standoff=0,
        xaxis_title="Time",
        yaxis_title="Battery Power (MW)",
        yaxis2_title="SOC (%)",
        yaxis2_range=[0, 100],
        yaxis_range=[-1, 1],
        yaxis_dtick=0.2,
        yaxis2_dtick=20,
    )
    return fig

#%%
plots_path = Path("~/git/micro-apps/micro-apps/plots").expanduser()
no_app_path = Path("~/git/micro-apps/micro-apps/no-app").expanduser()
cvr_path = Path("~/git/micro-apps/micro-apps/conservation-voltage-reduction-app").expanduser()
carbon_path = Path("~/git/micro-apps/micro-apps/carbon-management-app").expanduser()
peak_path = Path("~/git/micro-apps/micro-apps/peak-shaving-app").expanduser()

power_path = Path("output/ieee123_apps/feeder_head.csv")

no_app_power_path = no_app_path / power_path

# Carbon Power
fig = create_power_plot(carbon_path / power_path, no_app_path / power_path)
save_pdf(fig, carbon_path / "output/ieee123_apps/feeder_head.pdf")
fig.write_image(carbon_path / "output/ieee123_apps/feeder_head.png")
fig.write_html(carbon_path / "output/ieee123_apps/feeder_head.html")
# Carbon Batteries
fig = create_batt_plot(carbon_path / "output/ieee123_apps/simulation_table.csv")
fig.write_image(carbon_path / "output/ieee123_apps/batt.png")
save_pdf(fig, carbon_path / "output/ieee123_apps/batt.pdf")

# CVR Power
fig = create_power_plot(cvr_path / power_path, no_app_path / power_path)
save_pdf(fig, cvr_path / "output/ieee123_apps/feeder_head.pdf")
fig.write_image(cvr_path / "output/ieee123_apps/feeder_head.png")
# CVR Q
fig = create_var_plot(cvr_path / power_path, no_app_path / power_path)
save_pdf(fig, cvr_path / "output/ieee123_apps/reactive_power.pdf")
fig.write_image(cvr_path / "output/ieee123_apps/reactive_power.png")
# CVR Voltage Summary
fig = create_voltage_plot(cvr_path / "output/ieee123_apps/voltages.csv")
fig.write_image(cvr_path / "output/ieee123_apps/voltage_summary.png")
save_pdf(fig, cvr_path / "output/ieee123_apps/voltage_summary.pdf")
# CVR Voltage
# fig = create_voltage_all_plot(cvr_path / "output/ieee123_apps/voltages.csv")
# fig.write_image(cvr_path / "output/ieee123_apps/voltages.png")
# save_pdf(fig, cvr_path / "output/ieee123_apps/voltages.pdf")

# Peak Shaving Batteries
fig = create_batt_plot(peak_path / "output/ieee123_apps/battery_power.csv", units="W")
fig.write_image(peak_path / "output/ieee123_apps/batt.png")
save_pdf(fig, peak_path / "output/ieee123_apps/batt.pdf")
# Peak Shaving Voltage
fig = create_voltage_plot(peak_path / "output/ieee123_apps/voltages.csv")
fig.write_image(peak_path / "output/ieee123_apps/voltages.png")
save_pdf(fig, peak_path / "output/ieee123_apps/voltages.pdf")
# Peak Shaving Power
fig = create_power_plot(peak_path / power_path, no_app_path / power_path)
save_pdf(fig, peak_path / "output/ieee123_apps/feeder_head.pdf")
fig.write_image(peak_path / "output/ieee123_apps/feeder_head.png")
# Peak Shaving loads - batteries
loads = pd.read_csv(peak_path / "output/ieee123_apps/loads_no_batt.csv")
loads = loads.melt(id_vars=["time"], value_vars=["a", "b", "c"], var_name="phase", value_name="p")
loads.phase = loads.phase.str.upper()
loads = loads.sort_values(by=["time", "phase"], ignore_index=True)
fig = create_power_plot(loads, no_app_path / power_path)
# save_pdf(fig, peak_path / "output/ieee123_apps/loads_no_batt.pdf")
fig.write_image(peak_path / "output/ieee123_apps/loads_no_batt.png")

fig = create_power_comparison_plot(
    base_path=no_app_path / power_path,
    peak_shaving_path=peak_path / power_path,
    carbon_path=carbon_path / power_path,
)
fig.write_image(plots_path / "carbon_peak_comparison.png")