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


def create_var_plot(path):
    load = pd.read_csv(path)
    load.p = load.p/1e6
    load.q = load.q/1e6
    load.time = pd.to_datetime(load['time'],origin="unix", unit='s')
    fig = px.line(load, x="time", y="q", facet_row="phase", labels={"p": "Active Power", "q": "Reactive Power"} )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_yaxis(lambda a: a.update(
        # tickformat=".1f",
        title="",
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
        yaxis2_title_standoff=0.0,
        xaxis_title="Time",
        yaxis2_title="Feeder Load (MVAr)",
    )
    return fig


def create_power_plot(path):
    load = pd.read_csv(path)
    load.p = load.p/1e6
    load.q = load.q/1e6
    load.time = pd.to_datetime(load['time'],origin="unix", unit='s')
    fig = px.line(load, x="time", y="p", facet_row="phase", labels={"p": "Active Power", "q": "Reactive Power"} )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].upper()))
    fig.for_each_yaxis(lambda a: a.update(
        # tickformat=".1f",
        title="",
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
        yaxis2_title_standoff=0.0,
        xaxis_title="Time",
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

def create_batt_plot(path):
    batt = pd.read_csv(path)
    batt["p"] = batt.p_a + batt.p_b + batt.p_c
    batt.time = pd.to_datetime(batt['time'], origin="unix", unit='s')
    batt_sum = batt.groupby(by="time").p.aggregate("sum")
    batt_soc = batt.groupby(by="time").soc.aggregate("mean")
    batt_agg = pd.DataFrame(index=batt_sum.index)
    batt_agg["p"] = batt_sum
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
        # yaxis2_title_standoff=0.0,
        xaxis_title_standoff=0,
        xaxis_title="Time",
        yaxis_title="Battery Power (MW)",
        yaxis2_title="SOC (%)",
    )
    return fig
#%%
no_app_path = Path("~/git/micro-apps/micro-apps/no-app").expanduser()
cvr_path = Path("~/git/micro-apps/micro-apps/conservation-voltage-reduction-app").expanduser()
carbon_path = Path("~/git/micro-apps/micro-apps/carbon-management-app").expanduser()
peak_path = Path("~/git/micro-apps/micro-apps/peak-shaving-app").expanduser()

path = carbon_path / "output/ieee123_apps/feeder_head.csv"
fig = create_power_plot(path)
save_pdf(fig, carbon_path / "output/ieee123_apps/feeder_head.pdf")
fig.write_image(carbon_path / "output/ieee123_apps/feeder_head.png")

path = carbon_path / "output/ieee123_apps/simulation_table.csv"
fig = create_batt_plot(path)
fig.write_image(carbon_path / "output/ieee123_apps/batt.png")
save_pdf(fig, carbon_path / "output/ieee123_apps/batt.pdf")

path = Path("no-app/output/ieee123_apps/feeder_head.csv")
fig = create_var_plot(path)
save_pdf(fig, "no-app/output/ieee123_apps/cvr_power.pdf")
fig.write_image("no-app/output/ieee123_apps/cvr_power.png")

# path = no_app_path / "output/ieee123_apps/voltages.csv"
# fig = create_voltage_plot(path)
# fig.write_image(no_app_path / "output/ieee123_apps/voltages.png")
# save_pdf(fig, no_app_path / "output/ieee123_apps/voltages.pdf")

path = peak_path / "output/ieee123_apps/battery_power.csv"
fig = create_batt_plot(path)
fig.write_image(peak_path / "output/ieee123_apps/batt.png")
save_pdf(fig, peak_path / "output/ieee123_apps/batt.pdf")

path = peak_path / "output/ieee123_apps/feeder_head.csv"
fig = create_power_plot(path)
path = peak_path / "output/ieee123_apps/feeder_head.csv"
save_pdf(fig, peak_path / "output/ieee123_apps/feeder_head.pdf")
fig.write_image(peak_path / "output/ieee123_apps/feeder_head.png")
