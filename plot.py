import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

def format_figure(figure: go.Figure):
    figure.update_layout(
        plot_bgcolor="white",  # this sets the background color of the plot area
        margin={
            "l": 40,
            "r": 5,
            "t": 5,
            "b": 40,
        },
        font=dict(
            family="Times New Roman",
            size=12,
        ),
        legend=dict(
            xanchor="right",
            x=1,  # this sets the x-position of the legend
            y=1,  # this sets the y-position of the legend
            font_size=12,
            bgcolor="rgba(255, 255, 255, 1)",  # this sets the background color of the legend
            bordercolor="black",
            borderwidth=1,
        ),
    )

    figure.for_each_xaxis(lambda axis: axis.update(
        gridcolor="lightgray",  # this sets the color of the x-axis grid lines
        linecolor="black",  # this sets the color of the x-axis zero line
        ticks="inside",
        mirror=True,
        tickson="boundaries",
        linewidth=2,
        tickwidth=2,
        gridwidth=1,
        zeroline=False,
        # tickformat=".0g",
        # range=[-10, -2],
        # type="log",
        title_standoff=0,
        dtick=30,
    ))
    figure.for_each_yaxis(lambda axis: axis.update(
        gridcolor="lightgray",  # this sets the color of the y-axis grid lines
        linecolor="black",  # this sets the color of the y-axis zero line
        ticks="inside",
        mirror=True,
        tickson="boundaries",
        linewidth=2,
        tickwidth=2,
        # tickformat=".0e",
        zeroline=False,
        # range=[0.59, 0.75],
        # range=[-2.1, -0.3],
        # dtick=0.05,
        # type="log",
        title_standoff=0,
        # title_font=dict(size=5, family="Times New Roman"),

    ))
    # figure.for_each_trace(lambda trace: trace.update(line_width=1))
    return figure


def print_pdf_double_column(figure, plot_name):
    go.Figure().write_image("dummy.pdf")
    time.sleep(1)
    figure.write_image(plot_name, format="pdf", width=672, height=336)


def print_pdf_single_column(figure, plot_name):
    go.Figure().write_image("dummy.pdf")
    time.sleep(1)
    figure.write_image(plot_name, format="pdf", width=336, height=336)


if __name__ == '__main__':
    print("start")
    df = pd.read_csv("micro-apps/carbon-management-app/simulation_table.csv")
    df.time = pd.to_datetime(df.time, unit="s")
    fig = make_subplots(
        subplot_titles=("Phase-A", "Phase-B", "Phase-C"),
        rows=1,
        cols=3,
        y_title="Power Charging (kW)",
        x_title="Time (s)",
        shared_yaxes=False,
        shared_xaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )
    df1 = df.loc[df.battery=="school", :]
    print(df1)
    fig.add_trace(
        go.Scatter(
            x=df.time,
            y=df.p_batt_a
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.time,
            y=df.p_batt_b
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=df.time,
            y=df.p_batt_c
        ),
        row=1, col=3
    )
    fig = format_figure(fig)
    fig.show()
    print("export pdf")
    # print_pdf_double_column(fig, "plot.pdf")