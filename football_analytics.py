# https://ameybagwe12-miniproject-footballanaly-football-analytics-203tqk.streamlit.app/ --> streamlit website link

import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from copy import deepcopy
import plotly.graph_objects as go


st.set_page_config(layout="wide")

analysis = ['Home', 'Heat Maps', 'Radar Plot', 'Radar Plot Comparison']
analysis_choice = st.selectbox("Select choice", analysis)

x_lims = [0, 1.15]
y_lims = [0, 0.74]

x_mid = x_lims[1] / 2
y_mid = y_lims[1] / 2


background_color = "green"
line_color = "white"
line_width = 2.
fig = 0

header = st.container()


def radar_comp(scrape_url):
    pg_connect = urlopen(scrape_url)
    pg_html = BeautifulSoup(pg_connect, "html.parser")
    json_raw_string = pg_html.findAll(name="script")[3].text
    start_ind = json_raw_string.index("\\")  # slicing of code
    stop_ind = json_raw_string.index("')")  # the start index is included and stop index is excluded
    json_data = json_raw_string[start_ind: stop_ind]
    json_data = json_data.encode("utf8").decode("unicode_escape")
    final_json_df = pd.json_normalize(json.loads(json_data))
    top_10_df = deepcopy(final_json_df.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    top_10_df = top_10_df.apply(pd.to_numeric, errors="ignore")
    top_10_df = top_10_df[top_10_df["time"] > 0]
    per90Cols = ["goals", "assists", "key_passes", "time", "xG", "xA", "shots", "npxG", "xGChain", "xGBuildup"]
    for col in per90Cols:
        top_10_df[col + "Per90"] = top_10_df[col].divide(top_10_df["time"]).multiply(90)
    cols_for_radar1 = []
    for i in per90Cols:
        cols_for_radar1.append(i)
    scaler = MinMaxScaler()
    top_10_df[cols_for_radar1] = scaler.fit_transform(top_10_df[cols_for_radar1])
    top_10_league = top_10_df.drop_duplicates(subset=["id"])[["id", "player_name"]]
    df_max = top_10_df[cols_for_radar1].max().max()
    player_name = st.text_input("Player Name", 'Lionel Messi')
    st.header('Given Dataset - ')
    st.dataframe(top_10_df)
    for i, row in top_10_league.iterrows():

        if row["player_name"] == player_name:
            continue
        print(row["player_name"])
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=top_10_df.loc[(top_10_df["player_name"] == player_name), cols_for_radar1].sum(),
            theta=cols_for_radar1,
            fill='toself',
            name=player_name))
        fig.add_trace(go.Scatterpolar(
            r=top_10_df.loc[(top_10_df["id"] == row["id"]), cols_for_radar1].sum(),
            theta=cols_for_radar1,
            fill='toself',
            name=row["player_name"]))
        fig.update_layout(
            title=player_name + " vs " + row["player_name"],
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df_max]
                )),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)


def create_full_pitch(x_lims, y_lims, background_color="green", line_color="white", line_width=2.):
    global fig
    fig = plt.figure(facecolor=background_color, figsize=(16, 10))
    ax = fig.add_subplot(111, facecolor=background_color)

    # Pitch Outline & Centre Line
    plt.plot([x_lims[0], x_lims[0]], [y_lims[0], y_lims[1]], linewidth=line_width, color=line_color)  # left goal-line
    plt.plot([x_lims[0], x_lims[1]], [y_lims[1], y_lims[1]], linewidth=line_width, color=line_color)  # Upper side-line
    plt.plot([x_lims[1], x_lims[1]], [y_lims[1], y_lims[0]], linewidth=line_width, color=line_color)  # Right goal-line
    plt.plot([x_lims[1], x_lims[0]], [y_lims[0], y_lims[0]], linewidth=line_width, color=line_color)  # Lower side-line
    plt.plot([x_mid, x_mid], [y_lims[0], y_lims[1]], linewidth=line_width, color=line_color)  # Center line

    # Left Penalty Area
    plt.plot([x_lims[0] + .18, x_lims[0] + .18], [y_mid - .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[0], x_lims[0] + .18], [y_mid + .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[0], x_lims[0] + .18], [y_mid - .22, y_mid - .22], color=line_color)

    # Right Penalty Area
    plt.plot([x_lims[1] - .18, x_lims[1] - .18], [y_mid - .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .18], [y_mid + .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .18], [y_mid - .22, y_mid - .22], color=line_color)

    # Left 6yd box Area
    plt.plot([x_lims[0] + .06, x_lims[0] + .06], [y_mid - .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[0], x_lims[0] + .06], [y_mid + .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[0], x_lims[0] + .06], [y_mid - .06, y_mid - .06], color=line_color)

    # # Right 6yd box Area
    plt.plot([x_lims[1] - .06, x_lims[1] - .06], [y_mid - .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .06], [y_mid + .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .06], [y_mid - .06, y_mid - .06], color=line_color)

    # Prepare Circles
    centre_circle = plt.Circle((x_mid, y_mid), .1, color=line_color, fill=False)
    centre_spot = plt.Circle((x_mid, y_mid), 0.005, color=line_color)
    left_pen_spot = plt.Circle((x_lims[0] + 0.12, y_mid), 0.005, color=line_color)
    right_pen_spot = plt.Circle((x_lims[1] - 0.12, y_mid), 0.005, color=line_color)

    # Draw Circles
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)
    ax.add_patch(left_pen_spot)
    ax.add_patch(right_pen_spot)

    # Prepare Arcs
    left_arc = Arc((x_lims[0] + .12, y_mid), height=.183, width=.183, angle=0, theta1=310, theta2=50, color=line_color)
    right_arc = Arc((x_lims[1] - .12, y_mid), height=.183, width=.183, angle=0, theta1=130, theta2=230,
                    color=line_color)

    # Draw Arcs
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    plt.axis("off")


def build_heatmap(scrape_url):
    global fig
    page_connect = urlopen(scrape_url)
    page_html = BeautifulSoup(page_connect, "html.parser")

    json_raw_string = page_html.findAll(name="script")[3].text
    start_ind = json_raw_string.index("\\")
    stop_ind = json_raw_string.index("')")

    json_data = json_raw_string[start_ind:stop_ind]
    json_data = json_data.encode("utf8").decode("unicode_escape")

    final_json_df = pd.json_normalize(json.loads(json_data)).head(10)
    player_id_list = final_json_df["id"].to_list()
    print(player_id_list)
    player_shot_df_list = []
    for p_id in player_id_list:
        # Scrape player stats:
        scrape_url = "https://understat.com/player/{}".format(p_id)
        page_connect = urlopen(scrape_url)
        page_html = BeautifulSoup(page_connect, "html.parser")

        json_raw_string = page_html.findAll(name="script")[3].text
        start_ind = json_raw_string.index("\\")
        stop_ind = json_raw_string.index("')")

        json_data = json_raw_string[start_ind:stop_ind]
        json_data = json_data.encode("utf8").decode("unicode_escape")

        shots_df = pd.json_normalize(json.loads(json_data))
        shots_df = shots_df.apply(pd.to_numeric, errors="ignore")

        full_pitch_shots_df = deepcopy(shots_df)
        full_pitch_shots_df["X"] = full_pitch_shots_df["X"].multiply(x_lims[1])
        full_pitch_shots_df["Y"] = full_pitch_shots_df["Y"].multiply(y_lims[1])
        player_shot_df_list.append(full_pitch_shots_df)

    player_shot_df_list[9].head()
    for p_df in player_shot_df_list:
        # Generate image of field:
        create_full_pitch(x_lims, y_lims)

        # Plot the heat-map:
        ax = sns.kdeplot(x=p_df["X"], y=p_df["Y"], shade=True, cmap="YlOrRd", n_levels=10)
        # Set axis tick limits:
        plt.xlim(x_lims)
        plt.ylim(y_lims)
        plt.title(p_df["player"].unique().item())

        st.pyplot(fig)


def radar_plot(scrape_url, n):
    urlopen(scrape_url)
    page_connect = urlopen(scrape_url)
    page_html = BeautifulSoup(page_connect, "html.parser")
    json_raw_string = page_html.findAll(name="script")[3].text  # script block from html
    start_ind = json_raw_string.index("\\")  # slicing the data getting only essential detail
    stop_ind = json_raw_string.index("')")
    json_data = json_raw_string[start_ind:stop_ind]
    json_data = json_data.encode("utf8").decode("unicode_escape")
    final_json_df = pd.json_normalize(json.loads(json_data))
    players_df = final_json_df.apply(pd.to_numeric, errors="ignore")
    players_df = players_df.head(n)  # top n  players

    # Algorithm for per 90 stats
    players_df = players_df[players_df["time"] > 0]
    per90Cols = ["goals", "assists", "shots", "yellow_cards", "xG", "key_passes", "red_cards"]
    for col in per90Cols:
        players_df[col + "Per90"] = players_df[col].divide(players_df["time"]).multiply(90)
    cols_for_radar = [i + "Per90" for i in per90Cols]
    scaler = MinMaxScaler()
    players_df[cols_for_radar] = scaler.fit_transform(players_df[cols_for_radar])
    st.header('Given Dataset - ')
    st.dataframe(players_df)
    for i, row in players_df.iterrows():
        print(row["player_name"])
        fig = px.line_polar(players_df, r=players_df.loc[(players_df["id"] == row["id"]), cols_for_radar].sum(),
                            theta=cols_for_radar, line_close=True,
                            title=row["player_name"])
        fig.update_traces(fill='toself', mode='lines', line_color='indigo')
        st.plotly_chart(fig, use_container_width=True)


if analysis_choice == "Home":
    with header:
        st.markdown("<h1 style='text-align: left; color: orange;'>Football Analytics - Group 13</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: left; color: orange;'>Group Members : </h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: yellow;'>Amey Nitin Bagwe</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: yellow;'>Vailantan Fernandes</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: yellow;'>Wesley Lewis</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: yellow;'>Sandesh Raut</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: right; color: orange;'>Under Guidance Of - Dipali Maam</h1>", unsafe_allow_html=True)

elif analysis_choice == 'Heat Maps':
    st.title("Heat maps")
    user_choice = ['La Liga', 'Ligue ', 'English Premier League', 'Bundesliga', 'Serie A']
    select_user_choice = st.selectbox("Select choice", user_choice)

    if select_user_choice == 'La Liga':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            build_heatmap("https://understat.com/league/La_liga/2017")
        elif select_season == '2018/19':
            build_heatmap("https://understat.com/league/La_liga/2018")
        elif select_season == '2019/20':
            build_heatmap("https://understat.com/league/La_liga/2019")
        elif select_season == '2020/21':
            build_heatmap("https://understat.com/league/La_liga/2020")
        elif select_season == '2021/22':
            build_heatmap("https://understat.com/league/La_liga/2021")

    elif select_user_choice == 'Ligue 1':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            build_heatmap("https://understat.com/league/Ligue_1/2017")
        elif select_season == '2018/19':
            build_heatmap("https://understat.com/league/Ligue_1/2018")
        elif select_season == '2019/20':
            build_heatmap("https://understat.com/league/Ligue_1/2019")
        elif select_season == '2020/21':
            build_heatmap("https://understat.com/league/Ligue_1/2020")
        elif select_season == '2021/22':
            build_heatmap("https://understat.com/league/Ligue_1/2021")

    elif select_user_choice == 'English Premier League':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            build_heatmap("https://understat.com/league/EPL/2017")
        elif select_season == '2018/19':
            build_heatmap("https://understat.com/league/EPL/2018")
        elif select_season == '2019/20':
            build_heatmap("https://understat.com/league/EPL/2019")
        elif select_season == '2020/21':
            build_heatmap("https://understat.com/league/EPL/2020")
        elif select_season == '2021/22':
            build_heatmap("https://understat.com/league/EPL/2021")

    elif select_user_choice == 'Bundesliga':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            build_heatmap("https://understat.com/league/Bundesliga/2017")
        elif select_season == '2018/19':
            build_heatmap("https://understat.com/league/Bundesliga/2018")
        elif select_season == '2019/20':
            build_heatmap("https://understat.com/league/Bundesliga/2019")
        elif select_season == '2020/21':
            build_heatmap("https://understat.com/league/Bundesliga/2020")
        elif select_season == '2021/22':
            build_heatmap("https://understat.com/league/Bundesliga/2021")

    elif select_user_choice == 'Serie A':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            build_heatmap("https://understat.com/league/Serie_A/2017")
        elif select_season == '2018/19':
            build_heatmap("https://understat.com/league/Serie_A/2018")
        elif select_season == '2019/20':
            build_heatmap("https://understat.com/league/Serie_A/2019")
        elif select_season == '2020/21':
            build_heatmap("https://understat.com/league/Serie_A/2020")
        elif select_season == '2021/22':
            build_heatmap("https://understat.com/league/Serie_A/2021")

elif analysis_choice == 'Radar Plot Comparison':
    st.title("Radar plot Comparison")
    user_choice = ['La Liga', 'Ligue ', 'English Premier League', 'Bundesliga', 'Serie A']
    select_user_choice = st.selectbox("Select choice", user_choice)

    if select_user_choice == 'La Liga':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_comp("https://understat.com/league/La_liga/2017")
        elif select_season == '2018/19':
            radar_comp("https://understat.com/league/La_liga/2018")
        elif select_season == '2019/20':
            radar_comp("https://understat.com/league/La_liga/2019")
        elif select_season == '2020/21':
            radar_comp("https://understat.com/league/La_liga/2020")
        elif select_season == '2021/22':
            radar_comp("https://understat.com/league/La_liga/2021")

    elif select_user_choice == 'Ligue 1':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_comp("https://understat.com/league/Ligue_1/2017")
        elif select_season == '2018/19':
            radar_comp("https://understat.com/league/Ligue_1/2018")
        elif select_season == '2019/20':
            radar_comp("https://understat.com/league/Ligue_1/2019")
        elif select_season == '2020/21':
            radar_comp("https://understat.com/league/Ligue_1/2020")
        elif select_season == '2021/22':
            radar_comp("https://understat.com/league/Ligue_1/2021")

    elif select_user_choice == 'English Premier League':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_comp("https://understat.com/league/EPL/2017")
        elif select_season == '2018/19':
            radar_comp("https://understat.com/league/EPL/2018")
        elif select_season == '2019/20':
            radar_comp("https://understat.com/league/EPL/2019")
        elif select_season == '2020/21':
            radar_comp("https://understat.com/league/EPL/2020")
        elif select_season == '2021/22':
            radar_comp("https://understat.com/league/EPL/2021")

    elif select_user_choice == 'Bundesliga':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_comp("https://understat.com/league/Bundesliga/2017")
        elif select_season == '2018/19':
            radar_comp("https://understat.com/league/Bundesliga/2018")
        elif select_season == '2019/20':
            radar_comp("https://understat.com/league/Bundesliga/2019")
        elif select_season == '2020/21':
            radar_comp("https://understat.com/league/Bundesliga/2020")
        elif select_season == '2021/22':
            radar_comp("https://understat.com/league/Bundesliga/2021")

    elif select_user_choice == 'Serie A':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_comp("https://understat.com/league/Serie_A/2017")
        elif select_season == '2018/19':
            radar_comp("https://understat.com/league/Serie_A/2018")
        elif select_season == '2019/20':
            radar_comp("https://understat.com/league/Serie_A/2019")
        elif select_season == '2020/21':
            radar_comp("https://understat.com/league/Serie_A/2020")
        elif select_season == '2021/22':
            radar_comp("https://understat.com/league/Serie_A/2021")

elif analysis_choice == 'Radar Plot':
    st.title("Radar plot")
    top = st.text_input("Number of players ", 10)
    user_choice = ['La Liga', 'Ligue 1', 'English Premier League', 'Bundesliga', 'Serie A']
    select_user_choice = st.selectbox("Select choice", user_choice)

    if select_user_choice == 'La Liga':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_plot("https://understat.com/league/La_liga/2017", int(top))
        elif select_season == '2018/19':
            radar_plot("https://understat.com/league/La_liga/2018", int(top))
        elif select_season == '2019/20':
            radar_plot("https://understat.com/league/La_liga/2019", int(top))
        elif select_season == '2020/21':
            radar_plot("https://understat.com/league/La_liga/2020", int(top))
        elif select_season == '2021/22':
            radar_plot("https://understat.com/league/La_liga/2021", int(top))

    elif select_user_choice == 'Ligue 1':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_plot("https://understat.com/league/Ligue_1/2017", int(top))
        elif select_season == '2018/19':
            radar_plot("https://understat.com/league/Ligue_1/2018", int(top))
        elif select_season == '2019/20':
            radar_plot("https://understat.com/league/Ligue_1/2019", int(top))
        elif select_season == '2020/21':
            radar_plot("https://understat.com/league/Ligue_1/2020", int(top))
        elif select_season == '2021/22':
            radar_plot("https://understat.com/league/Ligue_1/2021", int(top))

    elif select_user_choice == 'English Premier League':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_plot("https://understat.com/league/EPL/2017", int(top))
        elif select_season == '2018/19':
            radar_plot("https://understat.com/league/EPL/2018", int(top))
        elif select_season == '2019/20':
            radar_plot("https://understat.com/league/EPL/2019", int(top))
        elif select_season == '2020/21':
            radar_plot("https://understat.com/league/EPL/2020", int(top))
        elif select_season == '2021/22':
            radar_plot("https://understat.com/league/EPL/2021", int(top))

    elif select_user_choice == 'Bundesliga':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_plot("https://understat.com/league/Bundesliga/2017", int(top))
        elif select_season == '2018/19':
            radar_plot("https://understat.com/league/Bundesliga/2018", int(top))
        elif select_season == '2019/20':
            radar_plot("https://understat.com/league/Bundesliga/2019", int(top))
        elif select_season == '2020/21':
            radar_plot("https://understat.com/league/Bundesliga/2020", int(top))
        elif select_season == '2021/22':
            radar_plot("https://understat.com/league/Bundesliga/2021", int(top))

    elif select_user_choice == 'Serie A':
        season = ['2017/18', '2018/19', '2019/20', '2020/21', '2021/22']
        select_season = st.selectbox("Select choice", season)
        if select_season == '2017/18':
            radar_plot("https://understat.com/league/Serie_A/2017", int(top))
        elif select_season == '2018/19':
            radar_plot("https://understat.com/league/Serie_A/2018", int(top))
        elif select_season == '2019/20':
            radar_plot("https://understat.com/league/Serie_A/2019", int(top))
        elif select_season == '2020/21':
            radar_plot("https://understat.com/league/Serie_A/2020", int(top))
        elif select_season == '2021/22':
            radar_plot("https://understat.com/league/Serie_A/2021", int(top))
