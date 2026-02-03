# NBA Evolution Dashboard (1996-2025)

An interactive dashboard exploring how the NBA has transformed over three decades, with a focus on the 3-point revolution, team success metrics, and individual player performance.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Live Demo

🏀 **[View Live Dashboard](https://nba-trends-dash.onrender.com)**

## Overview

This dashboard visualizes nearly 30 years of NBA data to answer questions like:
- How has the 3-point shot changed the game?
- Is there a correlation between 3-point shooting and team success?
- How do individual players compare to their positional peers?

The project demonstrates proficiency in interactive data visualization, dashboard development, and working with sports analytics data.

## Features

### Section 1: League Evolution
A scatter plot showing how any selected statistic has changed league-wide from 1996-2025. Filter by position, year range, and minimum minutes played. Includes trend line analysis and dynamic insight generation.

### Section 2: Team Success Analysis
Visualizes the relationship between 3-point attempts and wins for any season. Teams are color-coded by playoff result (Champion → Missed Playoffs), with team logos displayed on hover. Includes a Four Factors radar chart comparing the season champion to league average.

### Section 3: Player Spotlight
Search any player-season combination to view:
- **Shot Chart**: Toggle between scatter plot (every shot) and zone heatmap (efficiency vs league average)
- **Player Card**: Headshot, bio, per-game stats, shooting percentages, and awards
- **Radar Chart**: Customizable stat comparison against positional average

## Data Sources

All data is sourced from public APIs and hosted on Google Sheets for easy access:
- **Player Stats**: NBA.com Stats API (1996-2025)
- **Team Stats**: NBA.com Stats API with playoff results
- **Shot Data**: NBA.com Shot Chart API (~3M shots)
- **Awards & Age**: Basketball Reference

Data loads directly from Google Sheets on app startup—no local files required.

## Tech Stack

- **Framework**: Dash (Plotly)
- **Styling**: Dash Bootstrap Components (DARKLY theme)
- **Visualizations**: Plotly Express & Graph Objects
- **Data Processing**: Pandas, NumPy
- **Geometry**: Shapely (for shot chart zones)
- **Deployment**: Render

## Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/nba-evolution-dashboard.git
cd nba-evolution-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open http://localhost:8050 in your browser.

Note: First startup takes 30-60 seconds as data loads from Google Sheets.

## Deploy to Render

This app is configured for one-click deployment to Render's free tier.

### Steps:
1. Push this repo to your GitHub account
2. Go to [render.com](https://render.com) and sign up/login
3. Click **New** → **Web Service**
4. Connect your GitHub repo
5. Configure:
   - **Name**: `nba-evolution-dashboard` (or your choice)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`
6. Click **Create Web Service**

Render will automatically deploy and provide you with a public URL. Note: Free tier apps spin down after inactivity, so first load may take ~30 seconds.

## Project Structure

```
nba-evolution-dashboard/
├── app.py              # Main application (single file)
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Key Implementation Details

- **Modular Chart Functions**: Each visualization is a standalone function for readability and testing
- **Extracted Court Drawing**: `draw_court_lines()` utility prevents code duplication between shot chart types
- **Responsive Layout**: Uses percentage-based widths for flexible display
- **Dark Theme**: Custom CSS fixes dropdown visibility in DARKLY Bootstrap theme
- **Production Ready**: `server = app.server` exposes Flask server for Gunicorn
