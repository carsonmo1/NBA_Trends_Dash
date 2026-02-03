"""
NBA Evolution Dashboard (1996-2025)

An interactive Dash application exploring how the NBA has changed over three decades,
focusing on the 3-point revolution, team success metrics, and player performance analysis.

Data sourced from NBA.com API and Basketball Reference via public Google Sheets.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

# =============================================================================
# DATA LOADING
# =============================================================================
SHOTS_URL_1 = 'https://docs.google.com/spreadsheets/d/12OkW3lDR-iYa2ZThZl6c2HeRw0IfDB_nPGYTa-TiwJE/export?format=csv'
SHOTS_URL_2 = 'https://docs.google.com/spreadsheets/d/1bP1-C7-mK34SF-f4sJzOplGlIg1af4ugoBLaTAh1L3c/export?format=csv'
SHOTS_URL_3 = 'https://docs.google.com/spreadsheets/d/1caAWBD7yIyr_ftiJ4SyDK9jDEq3v6bwwVA7CC--uQik/export?format=csv'
SHOTS_URL_4 = 'https://docs.google.com/spreadsheets/d/18NFsYtTCy0fLi6WYbKfZGaqrMHE6IIHcmTtTNGHIauU/export?format=csv'
TEAMS_URL = 'https://docs.google.com/spreadsheets/d/1dpZM6SpwC8ezIsAolEC6bFLnoUi9QyWPCVgK7m0k9mU/export?format=csv'
PLAYERS_URL = 'https://docs.google.com/spreadsheets/d/19lH4qogN_DZ6pjgwMHyeRqD252BpqayJHDk2Yr42ZsY/export?format=csv'


def load_data():
    """Load all datasets from Google Sheets."""
    print("Loading data...")
    teams_df = pd.read_csv(TEAMS_URL)
    players_df = pd.read_csv(PLAYERS_URL)
    shots_df = pd.concat([
        pd.read_csv(SHOTS_URL_1),
        pd.read_csv(SHOTS_URL_2),
        pd.read_csv(SHOTS_URL_3),
        pd.read_csv(SHOTS_URL_4)
    ], ignore_index=True)
    print(f"Loaded: {len(teams_df):,} teams, {len(players_df):,} players, {len(shots_df):,} shots")
    
    # Preprocessing
    players_df['YEAR'] = players_df['SEASON'].str[:4].astype(int)
    players_df['MPG'] = players_df['MIN'] / players_df['GP']
    
    return teams_df, players_df, shots_df


# Load data at startup
teams_df, players_df, shots_df = load_data()

# =============================================================================
# CONSTANTS
# =============================================================================
POSITION_COLORS = {
    'PG': '#0173B2',
    'SG': '#DE8F05',
    'SF': '#029E73',
    'PF': '#CC78BC',
    'C': '#CA9161'
}

PLAYOFF_COLORS = {
    'Champion': '#FFD700',
    'Finals': '#C0C0C0',
    'Conference Finals': '#CD7F32',
    'Conference Semifinals': '#4169E1',
    'First Round': '#32CD32',
    'Missed Playoffs': '#808080'
}

# =============================================================================
# COURT DRAWING UTILITIES
# =============================================================================
def draw_court_lines(fig):
    """
    Add NBA court lines to a plotly figure.
    Draws sidelines, baseline, hoop, backboard, paint, free throw circle,
    restricted area, three-point line, and center court arc.
    """
    # Sidelines
    fig.add_trace(go.Scatter(x=[-250, -250], y=[-47.5, 422.5], mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[250, 250], y=[-47.5, 422.5], mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    
    # Half court and baseline
    fig.add_trace(go.Scatter(x=[-250, 250], y=[422.5, 422.5], mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-250, 250], y=[-47.5, -47.5], mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    
    # Hoop
    hoop_theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=7.5*np.cos(hoop_theta), y=7.5*np.sin(hoop_theta), mode='lines',
                             line=dict(color='orange', width=2), showlegend=False, hoverinfo='skip'))
    
    # Backboard
    fig.add_trace(go.Scatter(x=[-30, 30], y=[-7.5, -7.5], mode='lines',
                             line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    
    # Paint (outer and inner box)
    fig.add_trace(go.Scatter(x=[-80, -80, 80, 80, -80], y=[-47.5, 143.5, 143.5, -47.5, -47.5], mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-60, -60, 60, 60, -60], y=[-47.5, 143.5, 143.5, -47.5, -47.5], mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    
    # Free throw circle (top half solid, bottom half dashed)
    ft_theta = np.linspace(0, np.pi, 100)
    ft_x = 60 * np.cos(ft_theta)
    ft_y = 60 * np.sin(ft_theta) + 143.5
    fig.add_trace(go.Scatter(x=ft_x, y=ft_y, mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    
    ft_theta_bottom = np.linspace(np.pi, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=60*np.cos(ft_theta_bottom), y=60*np.sin(ft_theta_bottom)+143.5, mode='lines',
                             line=dict(color='black', width=1, dash='dash'), showlegend=False, hoverinfo='skip'))
    
    # Restricted area arc
    ra_theta = np.linspace(0, np.pi, 100)
    fig.add_trace(go.Scatter(x=40*np.cos(ra_theta), y=40*np.sin(ra_theta), mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    
    # Three-point line (arc and corners)
    three_theta = np.linspace(np.arccos(220/237.5), np.pi - np.arccos(220/237.5), 100)
    fig.add_trace(go.Scatter(x=237.5*np.cos(three_theta), y=237.5*np.sin(three_theta), mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-220, -220], y=[-47.5, 88.5], mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[220, 220], y=[-47.5, 88.5], mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    
    # Center court arc
    center_theta = np.linspace(np.pi, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=60*np.cos(center_theta), y=60*np.sin(center_theta)+422.5, mode='lines',
                             line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    
    return fig


# =============================================================================
# CHART FUNCTIONS
# =============================================================================
def create_league_evolution_chart(df, y_stat='FG3A'):
    """
    Section 1: Player scatter plot showing how a stat has evolved across the league.
    Includes trend line and position-based coloring.
    """
    df = df.copy()
    pct_stats = ['FG3_PCT', 'FG_PCT', 'FT_PCT']
    
    if y_stat in pct_stats:
        df[y_stat] = df[y_stat] * 100

    fig = px.scatter(
        df, x='YEAR', y=y_stat, color='POSITION',
        color_discrete_map=POSITION_COLORS,
        custom_data=['PLAYER', 'TEAM', 'POSITION', 'SEASON'],
        opacity=0.5
    )

    y_label = f"{y_stat} (%)" if y_stat in pct_stats else y_stat
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Team: %{customdata[1]}<br>"
            "Position: %{customdata[2]}<br>"
            f"{y_label}: %{{y:.1f}}<br>"
            "<extra></extra>"
        )
    )

    # Trend line
    z = np.polyfit(df['YEAR'], df[y_stat], 1)
    p = np.poly1d(z)
    years_trend = np.array(sorted(df['YEAR'].unique()))
    fig.add_trace(go.Scatter(
        x=years_trend, y=p(years_trend),
        mode='lines', name='Trend',
        line=dict(color='black', width=3, dash='dash'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(text=f"<b>NBA League Evolution: {y_stat} (1996-2025)</b>", x=0.5, font=dict(size=18)),
        xaxis=dict(title="Season", gridcolor='lightgray', dtick=2),
        yaxis=dict(title=y_label, gridcolor='lightgray'),
        plot_bgcolor='white',
        height=500,
        legend=dict(title="Position", x=0.01, y=0.99)
    )
    return fig


def create_team_success_chart(df, season='2016-17'):
    """Section 2: Team scatter plot showing 3PA vs Wins with playoff results."""
    season_df = df[df['SEASON'] == season].copy()
    season_df['FG3A_PG'] = season_df['FG3A'] / season_df['GP']

    fig = go.Figure()

    # Trend line
    if len(season_df) > 1:
        z = np.polyfit(season_df['FG3A_PG'], season_df['W'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(season_df['FG3A_PG'].min(), season_df['FG3A_PG'].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_trend, y=p(x_trend), mode='lines', name='Trend',
            line=dict(color='rgba(0,0,0,0.3)', width=2, dash='dash'),
            hoverinfo='skip'
        ))

    # Team markers with logos
    for _, row in season_df.iterrows():
        color = PLAYOFF_COLORS.get(row['playoff_result'], '#808080')
        fig.add_trace(go.Scatter(
            x=[row['FG3A_PG']], y=[row['W']],
            mode='markers',
            marker=dict(size=40, color=color, opacity=0.7, line=dict(color=color, width=3)),
            name=row['TEAM_NAME'],
            hovertemplate=(
                f"<b>{row['TEAM_NAME']}</b><br>"
                f"3PA/game: {row['FG3A_PG']:.1f}<br>"
                f"Wins: {row['W']}<br>"
                f"Result: {row['playoff_result']}<br>"
                "<extra></extra>"
            ),
            showlegend=False
        ))
        if pd.notna(row.get('LOGO_URL')):
            fig.add_layout_image(dict(
                source=row['LOGO_URL'],
                x=row['FG3A_PG'], y=row['W'],
                xref="x", yref="y",
                sizex=2, sizey=4,
                xanchor="center", yanchor="middle",
                layer="above"
            ))

    # Legend for playoff results
    for result in ['Champion', 'Finals', 'Conference Finals', 'Conference Semifinals', 'First Round', 'Missed Playoffs']:
        if result in season_df['playoff_result'].values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=PLAYOFF_COLORS[result]),
                name=result, showlegend=True
            ))

    correlation = season_df['FG3A_PG'].corr(season_df['W']) if len(season_df) > 1 else 0
    fig.update_layout(
        title=dict(
            text=f"<b>Team Success vs 3-Point Attempts ({season})</b><br><sup>Correlation: {correlation:.3f}</sup>",
            x=0.5, font=dict(size=18)
        ),
        xaxis=dict(title="3-Point Attempts per Game", gridcolor='lightgray'),
        yaxis=dict(title="Wins", gridcolor='lightgray'),
        plot_bgcolor='white',
        height=500,
        legend=dict(title="Playoff Result", x=0.99, y=0.01, xanchor='right', yanchor='bottom')
    )
    return fig


def create_four_factors_radar(df, team_name, season):
    """Section 2: Four Factors radar chart comparing team to league average."""
    team_data = df[(df['TEAM_NAME'] == team_name) & (df['SEASON'] == season)]
    if team_data.empty:
        return go.Figure()
    team_data = team_data.iloc[0]

    season_df = df[df['SEASON'] == season]
    league_avg = season_df.mean(numeric_only=True)

    categories = ['eFG%', 'TOV%', 'ORB%', 'FT Rate', '3P%']
    column_map = {
        'eFG%': 'EFG_PCT', 'TOV%': 'TM_TOV_PCT', 'ORB%': 'OREB_PCT',
        'FT Rate': 'FT_RATE', '3P%': 'FG3_PCT'
    }

    team_values, league_values = [], []
    for cat in categories:
        col = column_map[cat]
        team_val = team_data[col] if col in team_data else 0
        league_val = league_avg[col] if col in league_avg else 0
        team_val = team_val * 100 if team_val < 1 else team_val
        league_val = league_val * 100 if league_val < 1 else league_val
        
        # Invert TOV% (lower is better)
        if cat == 'TOV%':
            team_val = 100 - (team_val * 5)
            league_val = 100 - (league_val * 5)
        
        team_values.append(team_val)
        league_values.append(league_val)

    # Close polygon
    team_values += [team_values[0]]
    league_values += [league_values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=league_values, theta=categories_closed, name='League Avg',
        line=dict(color='rgba(150,150,150,0.8)', width=2, dash='dash'),
        fill='toself', fillcolor='rgba(150,150,150,0.1)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=team_values, theta=categories_closed, name=team_name,
        line=dict(color='rgba(67,160,71,1)', width=3),
        fill='toself', fillcolor='rgba(67,160,71,0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 65], gridcolor='rgba(255,255,255,0.3)', tickfont=dict(color='white')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.3)', tickfont=dict(color='white')),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(text=f"{team_name} - Four Factors", x=0.5, font=dict(size=14, color='white')),
        height=350,
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h', font=dict(size=10))
    )
    return fig


def create_shot_chart(df, player_name, season):
    """Section 3: Player shot chart showing makes and misses."""
    player_shots = df[(df['PLAYER_NAME'] == player_name) & (df['SEASON'] == season)]

    if player_shots.empty:
        fig = go.Figure()
        fig.add_annotation(text="No shot data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=450)
        return fig

    makes = player_shots[player_shots['SHOT_MADE_FLAG'] == 1]
    misses = player_shots[player_shots['SHOT_MADE_FLAG'] == 0]

    fig = go.Figure()
    fig = draw_court_lines(fig)

    # Shots - misses first so makes render on top
    fig.add_trace(go.Scatter(
        x=misses['LOC_X'], y=misses['LOC_Y'],
        mode='markers', marker=dict(color='red', size=5, opacity=0.5),
        name='Miss', hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=makes['LOC_X'], y=makes['LOC_Y'],
        mode='markers', marker=dict(color='green', size=5, opacity=0.5),
        name='Make', hoverinfo='skip'
    ))

    fg_pct = len(makes) / len(player_shots) * 100 if len(player_shots) > 0 else 0
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name} Shot Chart ({season})</b><br><sup>{len(player_shots):,} attempts | {fg_pct:.1f}% FG</sup>",
            x=0.5, font=dict(size=14)
        ),
        xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-55, 445], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='#d2b48c',
        height=450, width=470,
        legend=dict(x=0.02, y=0.95, bgcolor='rgba(255,255,255,0.7)', font=dict(color='black')),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def create_shot_chart_heatmap(df, player_name, season):
    """Section 3: Player shot chart heatmap showing efficiency by zone vs league average."""
    player_shots = df[(df['PLAYER_NAME'] == player_name) & (df['SEASON'] == season)]

    if player_shots.empty:
        fig = go.Figure()
        fig.add_annotation(text="No shot data available", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        fig.update_layout(height=450)
        return fig

    # Calculate zone stats
    season_shots = df[df['SEASON'] == season]
    league_zone_stats = season_shots.groupby('SHOT_ZONE_BASIC').agg(
        lg_makes=('SHOT_MADE_FLAG', 'sum'),
        lg_attempts=('SHOT_MADE_FLAG', 'count')
    )
    league_zone_stats['lg_pct'] = league_zone_stats['lg_makes'] / league_zone_stats['lg_attempts'] * 100

    player_stats = player_shots.groupby('SHOT_ZONE_BASIC').agg(
        makes=('SHOT_MADE_FLAG', 'sum'),
        attempts=('SHOT_MADE_FLAG', 'count')
    ).reset_index()
    player_stats['fg_pct'] = player_stats['makes'] / player_stats['attempts'] * 100
    player_stats = player_stats.merge(league_zone_stats[['lg_pct']], left_on='SHOT_ZONE_BASIC', right_index=True, how='left')
    player_stats['diff'] = player_stats['fg_pct'] - player_stats['lg_pct']

    def diff_to_color(diff):
        """Convert percentage difference to green (above avg) or red (below avg)."""
        if diff >= 0:
            intensity = min(diff / 10, 1)
            r, g = int(255 * (1 - intensity)), 255
            b = int(255 * (1 - intensity))
        else:
            intensity = min(abs(diff) / 10, 1)
            r, g, b = 255, int(255 * (1 - intensity)), int(255 * (1 - intensity))
        return f'rgba({r}, {g}, {b}, 0.8)'

    def get_zone_data(zone_name):
        row = player_stats[player_stats['SHOT_ZONE_BASIC'] == zone_name]
        if len(row) == 0:
            return 0, 0, 0, 'rgba(128,128,128,0.3)'
        return row['fg_pct'].values[0], row['lg_pct'].values[0], row['attempts'].values[0], diff_to_color(row['diff'].values[0])

    # Build zone polygons
    corner_y = np.sqrt(237.5**2 - 220**2)
    three_theta = np.linspace(np.pi - np.arccos(220/237.5), np.arccos(220/237.5), 200)
    arc_points = [(round(237.5*np.cos(t), 2), round(237.5*np.sin(t), 2)) for t in three_theta]

    inside_3pt_points = [(-220, -47.5)] + arc_points + [(220, -47.5)]
    three_pt_area = Polygon(inside_3pt_points).buffer(0)

    ra_theta = np.linspace(0, np.pi, 100)
    ra_points = [(round(40*np.cos(t), 2), round(40*np.sin(t), 2)) for t in ra_theta]
    restricted = Polygon(ra_points).buffer(0)

    court = Polygon([(-250, -47.5), (250, -47.5), (250, 422.5), (-250, 422.5)]).buffer(0)
    paint = Polygon([(-80, -47.5), (80, -47.5), (80, 143.5), (-80, 143.5)]).buffer(0)
    left_corner = Polygon([(-250, -47.5), (-220, -47.5), (-220, corner_y), (-250, corner_y)]).buffer(0)
    right_corner = Polygon([(220, -47.5), (250, -47.5), (250, corner_y), (220, corner_y)]).buffer(0)

    above_break_3 = court.difference(three_pt_area).difference(left_corner).difference(right_corner)
    mid_range = three_pt_area.difference(paint)
    paint_non_ra = paint.difference(restricted)

    fig = go.Figure()

    def add_polygon_to_fig(fig, poly, color):
        if poly.is_empty:
            return
        if poly.geom_type == 'MultiPolygon':
            for geom in poly.geoms:
                x, y = geom.exterior.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), fill='toself', fillcolor=color,
                                         line=dict(width=0), showlegend=False, hoverinfo='skip'))
        else:
            x, y = poly.exterior.xy
            fig.add_trace(go.Scatter(x=list(x), y=list(y), fill='toself', fillcolor=color,
                                     line=dict(width=0), showlegend=False, hoverinfo='skip'))

    # Get zone colors and add polygons
    ab3_pct, _, ab3_att, ab3_color = get_zone_data('Above the Break 3')
    mid_pct, _, mid_att, mid_color = get_zone_data('Mid-Range')
    lc3_pct, _, lc3_att, lc3_color = get_zone_data('Left Corner 3')
    rc3_pct, _, rc3_att, rc3_color = get_zone_data('Right Corner 3')
    paint_pct, _, paint_att, paint_color = get_zone_data('In The Paint (Non-RA)')
    ra_pct, _, ra_att, ra_color = get_zone_data('Restricted Area')

    add_polygon_to_fig(fig, above_break_3, ab3_color)
    add_polygon_to_fig(fig, left_corner, lc3_color)
    add_polygon_to_fig(fig, right_corner, rc3_color)
    add_polygon_to_fig(fig, mid_range, mid_color)
    add_polygon_to_fig(fig, paint_non_ra, paint_color)
    add_polygon_to_fig(fig, restricted, ra_color)

    # Court lines (simplified for heatmap)
    fig.add_trace(go.Scatter(x=[-250, -250], y=[-47.5, 422.5], mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[250, 250], y=[-47.5, 422.5], mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-250, 250], y=[422.5, 422.5], mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-250, 250], y=[-47.5, -47.5], mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))

    arc_line_theta = np.linspace(np.arccos(220/237.5), np.pi - np.arccos(220/237.5), 100)
    fig.add_trace(go.Scatter(x=237.5*np.cos(arc_line_theta), y=237.5*np.sin(arc_line_theta), mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-220, -220], y=[-47.5, corner_y], mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[220, 220], y=[-47.5, corner_y], mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[-80, -80, 80, 80, -80], y=[-47.5, 143.5, 143.5, -47.5, -47.5], mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))

    ra_x = 40 * np.cos(np.linspace(0, np.pi, 100))
    ra_y = 40 * np.sin(np.linspace(0, np.pi, 100))
    fig.add_trace(go.Scatter(x=ra_x, y=ra_y, mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'))

    hoop_theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=7.5*np.cos(hoop_theta), y=7.5*np.sin(hoop_theta), mode='lines', line=dict(color='orange', width=2), showlegend=False, hoverinfo='skip'))

    # Zone labels
    for x, y, pct, att in [(0, 18, ra_pct, ra_att), (0, 95, paint_pct, paint_att), (-140, 50, mid_pct, mid_att),
                           (0, 280, ab3_pct, ab3_att), (-235, 20, lc3_pct, lc3_att), (235, 20, rc3_pct, rc3_att)]:
        fig.add_annotation(x=x, y=y, text=f"{pct:.1f}%<br>({int(att)})", showarrow=False,
                          font=dict(size=9, color='black', family='Arial Black'))

    fig.update_layout(
        title=dict(
            text=f"<b>{player_name} Zone Heatmap ({season})</b><br><sup>Green = Above League Avg | Red = Below League Avg</sup>",
            x=0.5, font=dict(size=14)
        ),
        xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-55, 445], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='#d2b48c',
        height=450, width=470,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def create_player_radar(df, player_name, season, selected_stats=None):
    """Section 3: Player performance radar chart comparing to position average."""
    if selected_stats is None:
        selected_stats = ['PTS', 'AST', 'REB', 'STL', 'FG3M']

    player_data = df[(df['PLAYER'] == player_name) & (df['SEASON'] == season)]
    if player_data.empty:
        return go.Figure()
    player_data = player_data.iloc[0]

    position = player_data['POSITION']
    position_df = df[(df['SEASON'] == season) & (df['POSITION'] == position)]

    no_divide_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST_TOV', 'STL_TOV', 'EFF']

    player_values, pos_avg_values = [], []
    for stat in selected_stats:
        if stat in no_divide_stats:
            player_val = player_data[stat]
            pos_max = position_df[stat].max()
            pos_avg = position_df[stat].mean()
        else:
            player_val = player_data[stat] / player_data['GP']
            pos_max = (position_df[stat] / position_df['GP']).max()
            pos_avg = (position_df[stat] / position_df['GP']).mean()

        pos_max = pos_max if pos_max != 0 else 1
        player_values.append((player_val / pos_max) * 100)
        pos_avg_values.append((pos_avg / pos_max) * 100)

    # Close polygon
    player_values += [player_values[0]]
    pos_avg_values += [pos_avg_values[0]]
    stat_labels_closed = selected_stats + [selected_stats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=pos_avg_values, theta=stat_labels_closed, name=f'{position} Avg',
        line=dict(color='rgba(150,150,150,0.8)', width=2, dash='dash'),
        fill='toself', fillcolor='rgba(150,150,150,0.1)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=player_values, theta=stat_labels_closed, name=player_name,
        line=dict(color='rgba(67,160,71,1)', width=3),
        fill='toself', fillcolor='rgba(67,160,71,0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.3)', tickfont=dict(color='white')),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.3)', tickfont=dict(color='white')),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title=dict(text=f"{player_name} vs {position} Avg", x=0.5, font=dict(size=14, color='white')),
        height=350,
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h', font=dict(size=10))
    )
    return fig


def create_player_stats_panel(player_data):
    """Section 3: Player stats card with headshot, bio, and key stats."""
    p = player_data
    player_name = p['PLAYER']
    gp = p['GP']

    # Per-game stats
    ppg, apg, rpg = p['PTS']/gp, p['AST']/gp, p['REB']/gp
    spg, bpg, topg = p['STL']/gp, p['BLK']/gp, p['TOV']/gp
    fg3m_pg, mpg = p['FG3M']/gp, p['MIN']/gp

    # Percentages
    fg_pct, fg3_pct, ft_pct = p['FG_PCT']*100, p['FG3_PCT']*100, p['FT_PCT']*100

    age = p['Age'] if pd.notna(p.get('Age')) else None
    awards = p['Awards'] if pd.notna(p.get('Awards')) and p.get('Awards') != '' else None
    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{int(p['PLAYER_ID'])}.png"

    components = [
        html.Img(src=headshot_url, style={'height': '120px', 'borderRadius': '10px', 'marginBottom': '10px', 'border': '2px solid #444'}),
        html.H4(player_name, style={'margin': '5px 0', 'color': 'white'}),
        html.P(f"{p['POSITION']} | {p['TEAM']}", style={'margin': '2px 0', 'color': '#aaa', 'fontSize': '14px'}),
    ]

    if age is not None:
        components.append(html.P(f"Age: {int(age)} | {p['SEASON']}", style={'margin': '2px 0 10px 0', 'color': '#aaa', 'fontSize': '14px'}))
    else:
        components.append(html.P(f"{p['SEASON']}", style={'margin': '2px 0 10px 0', 'color': '#aaa', 'fontSize': '14px'}))

    components.append(html.Div([
        html.Span(f"GP: {int(gp)}", style={'marginRight': '15px'}),
        html.Span(f"MPG: {mpg:.1f}")
    ], style={'marginBottom': '10px', 'fontSize': '13px', 'color': '#ccc'}))

    # Per game stats table
    components.append(html.Div([
        html.Table([
            html.Tr([html.Td(s, style={'color': '#888', 'fontSize': '11px'}) for s in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']]),
            html.Tr([html.Td(f"{v:.1f}", style={'fontWeight': 'bold', 'fontSize': '14px'}) for v in [ppg, apg, rpg, spg, bpg, topg]])
        ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '10px'})
    ]))

    # Shooting percentages table
    components.append(html.Div([
        html.Table([
            html.Tr([html.Td(s, style={'color': '#888', 'fontSize': '11px'}) for s in ['FG%', '3P%', 'FT%', '3PM']]),
            html.Tr([html.Td(f"{v:.1f}", style={'fontWeight': 'bold', 'fontSize': '14px'}) for v in [fg_pct, fg3_pct, ft_pct, fg3m_pg]])
        ], style={'width': '100%', 'textAlign': 'center', 'marginBottom': '10px'})
    ]))

    # Efficiency
    components.append(html.Div([
        html.Span("EFF: ", style={'color': '#888', 'fontSize': '12px'}),
        html.Span(f"{p['EFF']:.1f}", style={'fontWeight': 'bold', 'fontSize': '14px'})
    ], style={'marginBottom': '10px'}))

    # Awards
    if awards:
        components.append(html.Div([
            html.Hr(style={'borderColor': '#444', 'margin': '10px 0'}),
            html.P("Awards", style={'color': '#888', 'fontSize': '11px', 'margin': '5px 0'}),
            html.P(awards, style={'color': '#ffd700', 'fontSize': '12px', 'fontWeight': 'bold'})
        ]))

    return html.Div(components)


# =============================================================================
# APP INITIALIZATION
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server  # Required for deployment

# Custom CSS to fix dropdown text color in dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>NBA Evolution Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .Select-menu-outer { background-color: #303030 !important; }
            .VirtualizedSelectOption { color: white !important; background-color: #303030 !important; }
            .VirtualizedSelectFocusedOption { background-color: #505050 !important; color: white !important; }
            .Select-option { color: white !important; background-color: #303030 !important; }
            .Select-option.is-focused { background-color: #505050 !important; color: white !important; }
            .dash-dropdown .Select-menu-outer { background-color: #303030 !important; }
            .dash-dropdown .Select-option { color: white !important; }
            .dash-dropdown .Select-option:hover { background-color: #505050 !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Dropdown options
seasons = sorted(players_df['SEASON'].unique(), reverse=True)
positions = ['All'] + sorted(players_df['POSITION'].dropna().unique())
stat_options = ['FG3A', 'PTS', 'AST', 'REB', 'FG3M', 'FG3_PCT', 'FG_PCT', 'FT_PCT', 'EFF']

player_seasons = players_df[['PLAYER', 'SEASON']].drop_duplicates()
player_options = [{'label': f"{row['PLAYER']} ({row['SEASON']})", 'value': f"{row['PLAYER']}|{row['SEASON']}"}
                  for _, row in player_seasons.iterrows()]

# =============================================================================
# APP LAYOUT
# =============================================================================
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Img(src='https://cdn.nba.com/logos/nba/nba-logoman-75.svg', style={'height': '70px', 'marginRight': '20px'}),
            html.Div([
                html.H1("NBA Evolution Dashboard", style={'margin': '0'}),
                html.P("Exploring how the NBA has changed from 1996-2025", style={'margin': '0', 'color': 'gray'})
            ])
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ], style={'textAlign': 'center', 'marginTop': '20px', 'marginBottom': '20px'}),

    html.Hr(),

    # Section 1: League Evolution
    html.Div([html.H3("Section 1: League Evolution", style={'marginBottom': '15px'})]),
    html.Div([
        html.Div([
            html.Label("Y-Axis Stat:"),
            dcc.Dropdown(id='stat-dropdown', options=[{'label': s, 'value': s} for s in stat_options], value='FG3A', clearable=False)
        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Position:"),
            dcc.Dropdown(id='position-dropdown', options=[{'label': p, 'value': p} for p in positions], value='All', clearable=False)
        ], style={'width': '15%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Year Range:"),
            dcc.RangeSlider(id='year-slider', min=1996, max=2024, step=1, value=[1996, 2024], marks={y: str(y) for y in range(1996, 2025, 4)})
        ], style={'width': '35%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
        html.Div([
            html.Label("Min MPG:"),
            dcc.Slider(id='mpg-slider', min=0, max=30, step=5, value=0, marks={i: str(i) for i in range(0, 35, 5)})
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '20px'}),
    html.Div([dcc.Graph(id='league-evolution-chart')]),
    html.Div([
        html.Div(id='insight-text', style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '5px', 'color': 'white'})
    ], style={'marginBottom': '20px'}),

    html.Hr(),

    # Section 2: Team Success
    html.Div([html.H3("Section 2: Team Success Analysis", style={'marginBottom': '15px'})]),
    html.Div([
        html.Div([
            html.Label("Season:"),
            dcc.Dropdown(id='team-season-dropdown', options=[{'label': s, 'value': s} for s in seasons], value='2016-17', clearable=False)
        ], style={'width': '25%'})
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Div([dcc.Graph(id='team-success-chart')], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([dcc.Graph(id='four-factors-radar')], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'marginBottom': '20px'}),

    html.Hr(),

    # Section 3: Player Spotlight
    html.Div([html.H3("Section 3: Player Spotlight", style={'marginBottom': '15px'})]),
    html.Div([
        html.Div([
            html.Label("Search Player:"),
            dcc.Dropdown(id='player-search', options=player_options, value='Stephen Curry|2016-17', placeholder="Type to search players...", searchable=True)
        ], style={'width': '50%'})
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Div([
            html.Label("Radar Stats:"),
            dcc.Dropdown(
                id='radar-stats-dropdown',
                options=[{'label': s, 'value': s} for s in ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                         'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'EFF', 'AST_TOV', 'STL_TOV']],
                value=['PTS', 'AST', 'REB', 'STL', 'FG3M'],
                multi=True, placeholder="Select stats for radar chart..."
            )
        ], style={'width': '50%'})
    ], style={'marginBottom': '15px'}),
    html.Div([
        html.Div([
            html.Div([
                dbc.RadioItems(id='shot-chart-toggle', options=[{'label': 'Scatter', 'value': 'scatter'}, {'label': 'Heatmap', 'value': 'heatmap'}],
                              value='scatter', inline=True, style={'marginBottom': '5px'})
            ]),
            dcc.Graph(id='shot-chart')
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div(id='player-stats-panel', style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'textAlign': 'center'}),
        html.Div([dcc.Graph(id='player-radar')], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ]),

    # Footer
    html.Hr(),
    html.Div([html.P("Data: NBA Stats API & Basketball Reference (1996-2025)", style={'textAlign': 'center', 'color': 'gray'})])

], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


# =============================================================================
# CALLBACKS
# =============================================================================
@app.callback(
    Output('league-evolution-chart', 'figure'),
    [Input('stat-dropdown', 'value'), Input('position-dropdown', 'value'),
     Input('year-slider', 'value'), Input('mpg-slider', 'value')]
)
def update_league_evolution(stat, position, year_range, min_mpg):
    df = players_df.copy()
    df = df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]
    df = df[df['MPG'] >= min_mpg]
    if position != 'All':
        df = df[df['POSITION'] == position]
    return create_league_evolution_chart(df, stat)


@app.callback(
    Output('insight-text', 'children'),
    [Input('stat-dropdown', 'value'), Input('position-dropdown', 'value'),
     Input('year-slider', 'value'), Input('mpg-slider', 'value')]
)
def update_insight_text(stat, position, year_range, min_mpg):
    df = players_df.copy()
    df = df[(df['YEAR'] >= year_range[0]) & (df['YEAR'] <= year_range[1])]
    df = df[df['MPG'] >= min_mpg]
    if position != 'All':
        df = df[df['POSITION'] == position]

    start_avg = df[df['YEAR'] == year_range[0]][stat].mean()
    end_avg = df[df['YEAR'] == year_range[1]][stat].mean()
    change = end_avg - start_avg
    direction = "increased" if change > 0 else "decreased"
    pos_text = position if position != 'All' else "all positions"

    pct_stats = ['FG3_PCT', 'FG_PCT', 'FT_PCT']
    change_display = f"{abs(change) * 100:.1f}%" if stat in pct_stats else f"{abs(change):.1f}"

    return f"From {year_range[0]} to {year_range[1]}, the average {stat} for {pos_text} {direction} by {change_display}."


@app.callback(Output('team-success-chart', 'figure'), [Input('team-season-dropdown', 'value')])
def update_team_success(season):
    return create_team_success_chart(teams_df, season)


@app.callback(Output('four-factors-radar', 'figure'), [Input('team-season-dropdown', 'value')])
def update_four_factors(season):
    season_df = teams_df[teams_df['SEASON'] == season]
    champ = season_df[season_df['playoff_result'] == 'Champion']
    team_name = champ.iloc[0]['TEAM_NAME'] if not champ.empty else (season_df.iloc[0]['TEAM_NAME'] if not season_df.empty else 'Golden State Warriors')
    return create_four_factors_radar(teams_df, team_name, season)


@app.callback(
    Output('shot-chart', 'figure'),
    [Input('player-search', 'value'), Input('shot-chart-toggle', 'value')]
)
def update_shot_chart(player_season, chart_type):
    if not player_season:
        return go.Figure()
    player_name, season = player_season.split('|')
    return create_shot_chart_heatmap(shots_df, player_name, season) if chart_type == 'heatmap' else create_shot_chart(shots_df, player_name, season)


@app.callback(
    Output('player-radar', 'figure'),
    [Input('player-search', 'value'), Input('radar-stats-dropdown', 'value')]
)
def update_player_radar(player_season, selected_stats):
    if not player_season or not selected_stats or len(selected_stats) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Select at least 3 stats", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color='white'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        return fig
    player_name, season = player_season.split('|')
    return create_player_radar(players_df, player_name, season, selected_stats)


@app.callback(Output('player-stats-panel', 'children'), [Input('player-search', 'value')])
def update_player_stats(player_season):
    if not player_season:
        return html.Div("Select a player to view stats")
    player_name, season = player_season.split('|')
    player_data = players_df[(players_df['PLAYER'] == player_name) & (players_df['SEASON'] == season)]
    if player_data.empty:
        return html.Div("Player data not found")
    return create_player_stats_panel(player_data.iloc[0])


# =============================================================================
# RUN
# =============================================================================
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
