import plotly.graph_objects as go
import plotly.colors as pc

def FigBarras(results):
    probabilidades = results['probabilities']

    labels = list(probabilidades.keys())
    values = list(probabilidades.values())

    sorted_items = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels_sorted, values_sorted = zip(*sorted_items)

    tonos_teal = pc.sequential.Tealgrn
    colores_usados = [tonos_teal[-1] if i == 0 else tonos_teal[-len(labels_sorted) + i] 
                      for i in range(len(labels_sorted))]

    max_val = max(values_sorted)
    x_range = [0, min(1.0, max_val + 0.1)]  

    fig_bar = go.Figure(go.Bar(
        x=values_sorted,
        y=labels_sorted,
        orientation='h',
        marker_color=colores_usados,
        text=[f"{v:.2%}" for v in values_sorted],
        textposition='outside',
        insidetextanchor='start',
        cliponaxis=False,
        hoverinfo='x+y'
    ))

    fig_bar.update_layout(
        title='',
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(family='Fira Sans Extra Condensed, sans-serif', size=15, color='white'),
        xaxis=dict(
            title='Probability',
            range=x_range,
            showgrid=False,
            tickfont=dict(size=13)
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=15)
        ),
        margin=dict(l=140, r=100, t=70, b=50)
    )

    return fig_bar


def FigTarta(results):
    probabilidades = results['probabilities']

    labels = list(probabilidades.keys())
    values = list(probabilidades.values())

    sorted_items = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels_sorted, values_sorted = zip(*sorted_items)

    tonos_teal = pc.sequential.Tealgrn
    colores_usados = tonos_teal[-len(labels_sorted):]

    max_val = max(values_sorted)
    pull_vals = [0.08 if v == max_val else 0 for v in values_sorted]

    fig_pie = go.Figure(go.Pie(
        labels=labels_sorted,
        values=values_sorted,
        hole=0.4,
        pull=pull_vals,
        marker=dict(colors=colores_usados),
        textinfo='percent+label',
        textposition='inside',
        insidetextorientation='auto',
        textfont=dict(size=16, color='white'),
        sort=False,
        direction='clockwise',
        showlegend=True
    ))

    fig_pie.update_layout(
        title='',
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            x=1.05, 
            y=0.5,
            traceorder="normal",
            font=dict(size=14, color="white")
        ),
        margin=dict(t=60, b=20, l=20, r=140),
        font=dict(family='Fira Sans Extra Condensed, sans-serif', size=15, color='white'),
        paper_bgcolor='black'
    )

    return fig_pie
