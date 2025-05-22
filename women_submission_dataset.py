import math
import random
import time
import warnings

import numpy as np
from torch.utils.data import Dataset
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

import os
import chardet



def load_csv(csv_file):
    with open(csv_file, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)

    # print(result)  # Check detected encoding
    df = pd.read_csv(csv_file, encoding=result["encoding"])
    return df
dirname = "/home/lm/Downloads/proje/marchmania/march-machine-learning-mania-2025"
mens_df = load_csv(os.path.join(dirname, 'MTeams.csv'))
womens_df = load_csv(os.path.join(dirname, 'WTeams.csv'))

mens_df_seasons = load_csv(os.path.join(dirname, 'MSeasons.csv'))
womens_df_seasons = load_csv(os.path.join(dirname, 'WSeasons.csv'))

mens_df_seeds = load_csv(os.path.join(dirname, 'MNCAATourneySeeds.csv'))
womens_df_seeds = load_csv(os.path.join(dirname, 'WNCAATourneySeeds.csv'))
mens_df_seeds["Seed"] = mens_df_seeds["Seed"].str.extract("(\d+)").astype(int)
womens_df_seeds["Seed"] = womens_df_seeds["Seed"].str.extract("(\d+)").astype(int)

mens_df_regular_compact = load_csv(os.path.join(dirname, 'MRegularSeasonCompactResults.csv'))
womens_df_regular_compact = load_csv(os.path.join(dirname, 'WRegularSeasonCompactResults.csv'))
wins = mens_df_regular_compact["WTeamID"].value_counts().rename("Wins")
losses = mens_df_regular_compact["LTeamID"].value_counts().rename("Losses")

mens_df_tourney_compact = load_csv(os.path.join(dirname, 'MNCAATourneyCompactResults.csv'))
womens_df_tourney_compact = load_csv(os.path.join(dirname, 'WNCAATourneyCompactResults.csv'))

mens_df_regular_detailed = load_csv(os.path.join(dirname, 'MRegularSeasonDetailedResults.csv'))
womens_df_regular_detailed = load_csv(os.path.join(dirname, 'WRegularSeasonDetailedResults.csv'))

mens_df_tourney_detailed = load_csv(os.path.join(dirname, 'MNCAATourneyDetailedResults.csv'))
womens_df_tourney_detailed = load_csv(os.path.join(dirname, 'WNCAATourneyDetailedResults.csv'))

sample_submission = load_csv(os.path.join(dirname, 'SampleSubmissionStage1.csv'))

mcoaches = load_csv(os.path.join(dirname, 'MTeamCoaches.csv'))

class MMarchManiaDataset(Dataset):
    def __init__(self):
        dirname = "/home/lm/Downloads/proje/marchmania/march-machine-learning-mania-20252"

        submission_file = load_csv(
            "/home/lm/Downloads/proje/marchmania/march-machine-learning-mania-20252/SampleSubmissionStage2.csv")
        submission_file[['Year', 'Team1', 'Team2']] = submission_file['ID'].str.split('_', expand=True)

        df = submission_file[['Year', 'Team1', 'Team2']].astype(int)
        self.team1_idxs = df[df["Team1"] > 2000]["Team1"].values
        self.team2_idxs = df[df["Team2"] > 2000]["Team2"].values


        self.mens_df = load_csv(os.path.join(dirname, 'MTeams.csv'))

        self.mens_df_seasons = load_csv(os.path.join(dirname, 'MSeasons.csv'))

        self.mens_df_seeds = load_csv(os.path.join(dirname, 'MNCAATourneySeeds.csv'))
        #self.mens_df_seeds["Seed"] = mens_df_seeds["Seed"].str.extract("(\d+)").astype(int)
        self.mens_df_seeds = mens_df_seeds
        self.womens_df_seeds = womens_df_seeds


        self.mens_df_regular_compact = load_csv(os.path.join(dirname, 'MRegularSeasonCompactResults.csv'))
        self.wins = mens_df_regular_compact["WTeamID"].value_counts().rename("Wins")
        self.losses = mens_df_regular_compact["LTeamID"].value_counts().rename("Losses")

        self.mens_df_tourney_compact = load_csv(os.path.join(dirname, 'MNCAATourneyCompactResults.csv'))

        self.mens_df_regular_detailed = load_csv(os.path.join(dirname, 'MRegularSeasonDetailedResults.csv'))
        self.womens_df_regular_detailed = load_csv(os.path.join(dirname, 'WRegularSeasonDetailedResults.csv'))


        self.mens_df_tourney_detailed = load_csv(os.path.join(dirname, 'MNCAATourneyDetailedResults.csv'))
        self.womens_df_tourney_detailed = load_csv(os.path.join(dirname, 'WNCAATourneyDetailedResults.csv'))

        self.sample_submission = load_csv(os.path.join(dirname, 'SampleSubmissionStage1.csv'))

        self.mcoaches = load_csv(os.path.join(dirname, 'MTeamCoaches.csv'))

        self.games = pd.concat((self.womens_df_tourney_detailed, self.womens_df_regular_detailed))

        self.k = 5
        self.global_winrate = 0.5
        self.labels = []
        self.data = []
        self.seeds = []

    def __len__(self):
        return len(self.team1_idxs)

    def calculate_diffs(self, wseed, lseed, wwrate, lwrate, wavgscore, lavgscore, wavg_score_eaten, lavg_score_eaten,
                        wwfgm, lwfgm, wwfgm3, lwfgm3, wwfga, lwfga, wwfga3,
                        lwfga3, wwfta, lwfta, wwor, lwor, wwdr, lwdr, wwast, lwast, wwto, lwto, wwstl, lwstl, wwblk,
                        lwblk, wwpf, lwpf, wwftm, lwftm):
        """Takımların özellik farklarını hesaplar ve verileri rastgele ters çevirerek dengesizlikten kaçınır."""

        flip = random.choice([True, False])  # %50 ihtimalle ters çevir

        if flip:
            seed_diff = wseed - lseed
            winrate_diff = wwrate - lwrate
            avgscore_diff = wavgscore.mean() - lavgscore.mean()
            avg_score_eaten_diff = wavg_score_eaten.mean() - lavg_score_eaten.mean()
            wfgm_diff = wwfgm.mean() - lwfgm.mean()
            wfgm3_diff = wwfgm3.mean() - lwfgm3.mean()
            wwfga_diff = wwfga.mean() - lwfga.mean()
            wwfga3_diff = wwfga3.mean() - lwfga3.mean()
            wfta_diff = wwfta.mean() - lwfta.mean()
            wwor_diff = wwor.mean() - lwor.mean()
            wdr_diff = wwdr.mean() - lwdr.mean()
            wast_diff = wwast.mean() - lwast.mean()
            wto_diff = wwto.mean() - lwto.mean()
            wstl_diff = wwstl.mean() - lwstl.mean()
            wblk_diff = wwblk.mean() - lwblk.mean()
            wpf_diff = wwpf.mean() - lwpf.mean()
            wftm = wwftm.mean() - lwftm.mean()
            avgscore_diff2 = wavgscore.std() - lavgscore.std()
            avg_score_eaten_diff2 = wavg_score_eaten.std() - lavg_score_eaten.std()
            wfgm_diff2 = wwfgm.std() - lwfgm.std()
            wfgm3_diff2 = wwfgm3.std() - lwfgm3.std()
            wwfga_diff2 = wwfga.std() - lwfga.std()
            wwfga3_diff2 = wwfga3.std() - lwfga3.std()
            wfta_diff2 = wwfta.std() - lwfta.std()
            wwor_diff2 = wwor.std() - lwor.std()
            wdr_diff2 = wwdr.std() - lwdr.std()
            wast_diff2 = wwast.std() - lwast.std()
            wto_diff2 = wwto.std() - lwto.std()
            wstl_diff2 = wwstl.std() - lwstl.std()
            wblk_diff2 = wwblk.std() - lwblk.std()
            wpf_diff2 = wwpf.std() - lwpf.std()
            wftm2 = wwftm.std() - lwftm.std()
            avgscore_diff3 = wavgscore.median() - lavgscore.median()
            avg_score_eaten_diff3 = wavg_score_eaten.median() - lavg_score_eaten.median()
            wfgm_diff3 = wwfgm.median() - lwfgm.median()
            wfgm3_diff3 = wwfgm3.median() - lwfgm3.median()
            wwfga_diff3 = wwfga.median() - lwfga.median()
            wwfga3_diff3 = wwfga3.median() - lwfga3.median()
            wfta_diff3 = wwfta.median() - lwfta.median()
            wwor_diff3 = wwor.median() - lwor.median()
            wdr_diff3 = wwdr.median() - lwdr.median()
            wast_diff3 = wwast.median() - lwast.median()
            wto_diff3 = wwto.median() - lwto.median()
            wstl_diff3 = wwstl.median() - lwstl.median()
            wblk_diff3 = wwblk.median() - lwblk.median()
            wpf_diff3 = wwpf.median() - lwpf.median()
            wftm3 = wwftm.median() - lwftm.median()
            fgm_per_fga = (wwfgm.mean() / wwfga.mean()) - (lwfgm.mean() / lwfga.mean())
            fgm3_per_fga3 = (wwfgm3.mean() / wwfga3.mean()) - (lwfgm3.mean() / lwfga3.mean())
            ftm_per_fta = (wwftm.mean() / wwfta.mean()) - (lwftm.mean() / lwfta.mean())
            ast_to_ratio = (wwast.mean() / wwto.mean()) - (lwast.mean() / lwto.mean())
            defensive_efficiency = (wwdr.mean() + wwblk.mean() + wwstl.mean()) - (
                    lwdr.mean() + lwblk.mean() + lwstl.mean())

            # İleri düzey istatistikler
            possessions = (wwfga.mean() + 0.44 * wwfta.mean() - wwor.mean() + wwto.mean()) - (
                    lwfga.mean() + 0.44 * lwfta.mean() - lwor.mean() + lwto.mean())
            true_shooting = ((wavgscore.mean() / (2 * (wwfga.mean() + 0.44 * wwfta.mean()))) - (
                    lavgscore.mean() / (2 * (lwfga.mean() + 0.44 * lwfta.mean()))))
            net_rating = ((wavgscore.mean() - wavg_score_eaten.mean()) - (lavgscore.mean() - lavg_score_eaten.mean()))

            pace = (2 * possessions) / (wwfga.mean() + lwfga.mean())
            offensive_rating = (wavgscore.mean() / possessions) - (lavgscore.mean() / possessions)
            defensive_rating = (wavg_score_eaten.mean() / possessions) - (lavg_score_eaten.mean() / possessions)
            efg_percentage = ((wwfgm.mean() + 0.5 * wwfgm3.mean()) / wwfga.mean()) - (
                    (lwfgm.mean() + 0.5 * lwfgm3.mean()) / lwfga.mean())
            ft_rate = (wwftm.mean() / wwfga.mean()) - (lwftm.mean() / lwfga.mean())
            to_percentage = (wwto.mean() / possessions) - (lwto.mean() / possessions)
            reb_percentage = ((wwor.mean() + wwdr.mean()) / (wwor.mean() + wwdr.mean() + lwor.mean() + lwdr.mean())) - \
                             ((lwor.mean() + lwdr.mean()) / (wwor.mean() + wwdr.mean() + lwor.mean() + lwdr.mean()))
            ast_percentage = (wwast.mean() / wwfgm.mean()) - (lwast.mean() / lwfgm.mean())
            stl_percentage = (wwstl.mean() / possessions) - (lwstl.mean() / possessions)
            blk_percentage = (wwblk.mean() / possessions) - (lwblk.mean() / possessions)

            # Eklenen yeni özellikler
            pace_consistency = (wwfga.std() / wwfga.mean()) - (lwfga.std() / lwfga.mean())
            ft_efficiency = ((wwftm.mean() / wavgscore.mean()) - (lwftm.mean() / lavgscore.mean()))
            inside_outside_balance = ((wwfga.mean() - wwfga3.mean()) / wwfgm.mean()) - (
                        (lwfga.mean() - lwfga3.mean()) / lwfgm.mean())
            defensive_pressure = ((wwstl.mean() + wwblk.mean()) / wavg_score_eaten.mean()) - (
                        (lwstl.mean() + lwblk.mean()) / lavg_score_eaten.mean())
            offensive_reb_efficiency = (wwor.mean() / (wwfga.mean() - wwfgm.mean())) - (
                        lwor.mean() / (lwfga.mean() - lwfgm.mean()))
            assist_quality = ((wwast.mean() / wwfgm.mean()) * (wwfgm.mean() / wwfga.mean())) - (
                        (lwast.mean() / lwfgm.mean()) * (lwfgm.mean() / lwfga.mean()))
            game_control = ((wwast.mean() - wwto.mean()) / wwfga.mean()) - ((lwast.mean() - lwto.mean()) / lwfga.mean())
            foul_efficiency = (wwpf.mean() / lwftm.mean()) - (lwpf.mean() / wwftm.mean())
            perimeter_defense = (wwblk.mean() / (lwfga3.mean())) - (lwblk.mean() / (wwfga3.mean()))
            defense_to_offense = ((wwstl.mean() + wwblk.mean()) / wwfgm.mean()) - (
                        (lwstl.mean() + lwblk.mean()) / lwfgm.mean())
            q25_diff = wwfgm.quantile(0.25) - lwfgm.quantile(0.25)
            q75_diff = wwfgm.quantile(0.75) - lwfgm.quantile(0.75)
            q25_diff3 = wavgscore.quantile(0.25) - lavgscore.quantile(0.25)
            q75_diff3 = wavgscore.quantile(0.75) - lavgscore.quantile(0.75)
            q25_diff4 = wavg_score_eaten.quantile(0.25) - lavg_score_eaten.quantile(0.25)
            q75_diff4 = wavg_score_eaten.quantile(0.75) - lavg_score_eaten.quantile(0.75)
            turnover_ratio = wwto.mean() / (wwfga.mean() + 0.44 * wwfta.mean() + wwto.mean()) - \
                             lwto.mean() / (lwfga.mean() + 0.44 * lwfta.mean() + lwto.mean())
            offensive_rebound_percentage = (wwor.mean() / (wwor.mean() + lwdr.mean())) - \
                                           (lwor.mean() / (lwor.mean() + wwdr.mean()))

            shot_selection_quality = ((wwfgm.mean() / wwfga.mean()) * (wwfgm3.mean() / wwfga.mean())) - (
                        (lwfgm.mean() / lwfga.mean()) * (lwfgm3.mean() / lwfga.mean()))
            ft_drawing_ability = (wwfta.mean() / wwfga.mean()) - (lwfta.mean() / lwfga.mean())
            defensive_discipline = ((wwblk.mean() + wwstl.mean()) / wwpf.mean()) - (
                        (lwblk.mean() + lwstl.mean()) / lwpf.mean())
            inside_outside_variance = (wwfga.std() - wwfga3.std()) - (lwfga.std() - lwfga3.std())
            offensive_efficiency_combo = ((wwast.mean() / wwto.mean()) * (wwfgm.mean() / wwfga.mean())) - (
                        (lwast.mean() / lwto.mean()) * (lwfgm.mean() / lwfga.mean()))
            game_control_consistency = ((wwast.std() - wwto.std()) / wwfga.std()) - (
                        (lwast.std() - lwto.std()) / lwfga.std())
            steal_to_points_efficiency = (wwstl.mean() / wavgscore.mean()) - (lwstl.mean() / lavgscore.mean())
            score_volatility = wavgscore.std() / wavgscore.mean() - lavgscore.std() / lavgscore.mean()
            second_chance_efficiency = (wwor.mean() / wavgscore.mean()) - (lwor.mean() / lavgscore.mean())
            transition_effectiveness = ((wwstl.mean() + wwblk.mean()) / wavgscore.mean()) - (
                        (lwstl.mean() + lwblk.mean()) / lavgscore.mean())

            # İlave kuantil analizleri
            iqr_fgm = (wwfgm.quantile(0.75) - wwfgm.quantile(0.25)) - (lwfgm.quantile(0.75) - lwfgm.quantile(0.25))
            iqr_score = (wavgscore.quantile(0.75) - wavgscore.quantile(0.25)) - (
                        lavgscore.quantile(0.75) - lavgscore.quantile(0.25))
            q10_q90_range = (wwfgm.quantile(0.9) - wwfgm.quantile(0.1)) - (lwfgm.quantile(0.9) - lwfgm.quantile(0.1))

            # Şut bölgesi ve pozisyon etkiliği
            interior_scoring_efficiency = ((wwfgm.mean() - wwfgm3.mean()) / (wwfga.mean() - wwfga3.mean())) - (
                        (lwfgm.mean() - lwfgm3.mean()) / (lwfga.mean() - lwfga3.mean()))

            self.labels.append(1)
        else:
            seed_diff = lseed - wseed
            winrate_diff = lwrate - wwrate
            avgscore_diff = lavgscore.mean() - wavgscore.mean()
            avg_score_eaten_diff = lavg_score_eaten.mean() - wavg_score_eaten.mean()
            wfgm_diff = lwfgm.mean() - wwfgm.mean()
            wfgm3_diff = lwfgm3.mean() - wwfgm3.mean()
            wwfga_diff = lwfga.mean() - wwfga.mean()
            wwfga3_diff = lwfga3.mean() - wwfga3.mean()
            wfta_diff = lwfta.mean() - wwfta.mean()
            wwor_diff = lwor.mean() - wwor.mean()
            wdr_diff = lwdr.mean() - wwdr.mean()
            wast_diff = lwast.mean() - wwast.mean()
            wto_diff = lwto.mean() - wwto.mean()
            wstl_diff = lwstl.mean() - wwstl.mean()
            wblk_diff = lwblk.mean() - wwblk.mean()
            wpf_diff = lwpf.mean() - wwpf.mean()
            wftm = lwftm.mean() - wwftm.mean()
            avgscore_diff2 = lavgscore.std() - wavgscore.std()
            avg_score_eaten_diff2 = lavg_score_eaten.std() - wavg_score_eaten.std()
            wfgm_diff2 = lwfgm.std() - wwfgm.std()
            wfgm3_diff2 = lwfgm3.std() - wwfgm3.std()
            wwfga_diff2 = lwfga.std() - wwfga.std()
            wwfga3_diff2 = lwfga3.std() - wwfga3.std()
            wfta_diff2 = lwfta.std() - wwfta.std()
            wwor_diff2 = lwor.std() - wwor.std()
            wdr_diff2 = lwdr.std() - wwdr.std()
            wast_diff2 = lwast.std() - wwast.std()
            wto_diff2 = lwto.std() - wwto.std()
            wstl_diff2 = lwstl.std() - wwstl.std()
            wblk_diff2 = lwblk.std() - wwblk.std()
            wpf_diff2 = lwpf.std() - wwpf.std()
            wftm2 = lwftm.std() - wwftm.std()
            avgscore_diff3 = lavgscore.median() - wavgscore.median()
            avg_score_eaten_diff3 = lavg_score_eaten.median() - wavg_score_eaten.median()
            wfgm_diff3 = lwfgm.median() - wwfgm.median()
            wfgm3_diff3 = lwfgm3.median() - wwfgm3.median()
            wwfga_diff3 = lwfga.median() - wwfga.median()
            wwfga3_diff3 = lwfga3.median() - wwfga3.median()
            wfta_diff3 = lwfta.median() - wwfta.median()
            wwor_diff3 = lwor.median() - wwor.median()
            wdr_diff3 = lwdr.median() - wwdr.median()
            wast_diff3 = lwast.median() - wwast.median()
            wto_diff3 = lwto.median() - wwto.median()
            wstl_diff3 = lwstl.median() - wwstl.median()
            wblk_diff3 = lwblk.median() - wwblk.median()
            wpf_diff3 = lwpf.median() - wwpf.median()
            wftm3 = lwftm.median() - wwftm.median()
            fgm_per_fga = (lwfgm.mean() / lwfga.mean()) - (wwfgm.mean() / wwfga.mean())
            fgm3_per_fga3 = (lwfgm3.mean() / lwfga3.mean()) - (wwfgm3.mean() / wwfga3.mean())
            ftm_per_fta = (lwftm.mean() / lwfta.mean()) - (wwftm.mean() / wwfta.mean())
            ast_to_ratio = (lwast.mean() / lwto.mean()) - (wwast.mean() / wwto.mean())
            defensive_efficiency = (lwdr.mean() + lwblk.mean() + lwstl.mean()) - (
                    wwdr.mean() + wwblk.mean() + wwstl.mean())

            possessions = (lwfga.mean() + 0.44 * lwfta.mean() - lwor.mean() + lwto.mean()) - (
                    wwfga.mean() + 0.44 * wwfta.mean() - wwor.mean() + wwto.mean())
            true_shooting = ((lavgscore.mean() / (2 * (lwfga.mean() + 0.44 * lwfta.mean()))) - (
                    wavgscore.mean() / (2 * (wwfga.mean() + 0.44 * wwfta.mean()))))
            net_rating = ((lavgscore.mean() - lavg_score_eaten.mean()) - (wavgscore.mean() - wavg_score_eaten.mean()))
            pace = (2 * possessions) / (lwfga.mean() + wwfga.mean())
            offensive_rating = (lavgscore.mean() / possessions) - (wavgscore.mean() / possessions)
            defensive_rating = (lavg_score_eaten.mean() / possessions) - (wavg_score_eaten.mean() / possessions)
            efg_percentage = ((lwfgm.mean() + 0.5 * lwfgm3.mean()) / lwfga.mean()) - (
                    (wwfgm.mean() + 0.5 * wwfgm3.mean()) / wwfga.mean())
            ft_rate = (lwftm.mean() / lwfga.mean()) - (wwftm.mean() / wwfga.mean())
            to_percentage = (lwto.mean() / possessions) - (wwto.mean() / possessions)
            reb_percentage = ((lwor.mean() + lwdr.mean()) / (lwor.mean() + lwdr.mean() + wwor.mean() + wwdr.mean())) - \
                             ((wwor.mean() + wwdr.mean()) / (lwor.mean() + lwdr.mean() + wwor.mean() + wwdr.mean()))
            ast_percentage = (lwast.mean() / lwfgm.mean()) - (wwast.mean() / wwfgm.mean())
            stl_percentage = (lwstl.mean() / possessions) - (wwstl.mean() / possessions)
            blk_percentage = (lwblk.mean() / possessions) - (wwblk.mean() / possessions)

            # Eklenen yeni özellikler
            pace_consistency = (lwfga.std() / lwfga.mean()) - (wwfga.std() / wwfga.mean())
            ft_efficiency = ((lwftm.mean() / lavgscore.mean()) - (wwftm.mean() / wavgscore.mean()))
            inside_outside_balance = ((lwfga.mean() - lwfga3.mean()) / lwfgm.mean()) - (
                        (wwfga.mean() - wwfga3.mean()) / wwfgm.mean())
            defensive_pressure = ((lwstl.mean() + lwblk.mean()) / lavg_score_eaten.mean()) - (
                        (wwstl.mean() + wwblk.mean()) / wavg_score_eaten.mean())
            offensive_reb_efficiency = (lwor.mean() / (lwfga.mean() - lwfgm.mean())) - (
                        wwor.mean() / (wwfga.mean() - wwfgm.mean()))
            assist_quality = ((lwast.mean() / lwfgm.mean()) * (lwfgm.mean() / lwfga.mean())) - (
                        (wwast.mean() / wwfgm.mean()) * (wwfgm.mean() / wwfga.mean()))
            game_control = ((lwast.mean() - lwto.mean()) / lwfga.mean()) - ((wwast.mean() - wwto.mean()) / wwfga.mean())
            foul_efficiency = (lwpf.mean() / wwftm.mean()) - (wwpf.mean() / lwftm.mean())
            perimeter_defense = (lwblk.mean() / (wwfga3.mean())) - (wwblk.mean() / (lwfga3.mean()))
            defense_to_offense = ((lwstl.mean() + lwblk.mean()) / lwfgm.mean()) - (
                        (wwstl.mean() + wwblk.mean()) / wwfgm.mean())
            q25_diff = lwfgm.quantile(0.25) - wwfgm.quantile(0.25)
            q75_diff = lwfgm.quantile(0.75) - wwfgm.quantile(0.75)
            q25_diff3 = lavgscore.quantile(0.25) - wavgscore.quantile(0.25)
            q75_diff3 = lavgscore.quantile(0.75) - wavgscore.quantile(0.75)
            q25_diff4 = lavg_score_eaten.quantile(0.25) - wavg_score_eaten.quantile(0.25)
            q75_diff4 = lavg_score_eaten.quantile(0.75) - wavg_score_eaten.quantile(0.75)
            turnover_ratio = lwto.mean() / (lwfga.mean() + 0.44 * lwfta.mean() + lwto.mean()) - \
                             wwto.mean() / (wwfga.mean() + 0.44 * wwfta.mean() + wwto.mean())
            offensive_rebound_percentage = (lwor.mean() / (lwor.mean() + wwdr.mean())) - \
                                           (wwor.mean() / (wwor.mean() + lwdr.mean()))

            shot_selection_quality = ((lwfgm.mean() / lwfga.mean()) * (lwfgm3.mean() / lwfga.mean())) - (
                        (wwfgm.mean() / wwfga.mean()) * (wwfgm3.mean() / wwfga.mean()))
            ft_drawing_ability = (lwfta.mean() / lwfga.mean()) - (wwfta.mean() / wwfga.mean())
            defensive_discipline = ((lwblk.mean() + lwstl.mean()) / lwpf.mean()) - (
                        (wwblk.mean() + wwstl.mean()) / wwpf.mean())
            inside_outside_variance = (lwfga.std() - lwfga3.std()) - (wwfga.std() - wwfga3.std())
            offensive_efficiency_combo = ((lwast.mean() / lwto.mean()) * (lwfgm.mean() / lwfga.mean())) - (
                        (wwast.mean() / wwto.mean()) * (wwfgm.mean() / wwfga.mean()))
            game_control_consistency = ((lwast.std() - lwto.std()) / lwfga.std()) - (
                        (wwast.std() - wwto.std()) / wwfga.std())
            steal_to_points_efficiency = (lwstl.mean() / lavgscore.mean()) - (wwstl.mean() / wavgscore.mean())
            score_volatility = lavgscore.std() / lavgscore.mean() - wavgscore.std() / wavgscore.mean()
            second_chance_efficiency = (lwor.mean() / lavgscore.mean()) - (wwor.mean() / wavgscore.mean())
            transition_effectiveness = ((lwstl.mean() + lwblk.mean()) / lavgscore.mean()) - (
                        (wwstl.mean() + wwblk.mean()) / wavgscore.mean())

            # İlave kuantil analizleri
            iqr_fgm = (lwfgm.quantile(0.75) - lwfgm.quantile(0.25)) - (wwfgm.quantile(0.75) - wwfgm.quantile(0.25))
            iqr_score = (lavgscore.quantile(0.75) - lavgscore.quantile(0.25)) - (
                        wavgscore.quantile(0.75) - wavgscore.quantile(0.25))
            q10_q90_range = (lwfgm.quantile(0.9) - lwfgm.quantile(0.1)) - (wwfgm.quantile(0.9) - wwfgm.quantile(0.1))

            # Şut bölgesi ve pozisyon etkiliği
            interior_scoring_efficiency = ((lwfgm.mean() - lwfgm3.mean()) / (lwfga.mean() - lwfga3.mean())) - (
                        (wwfgm.mean() - wwfgm3.mean()) / (wwfga.mean() - wwfga3.mean()))




            self.labels.append(0)

        self.data.append(
            [seed_diff, winrate_diff, avgscore_diff, avg_score_eaten_diff, wfgm_diff, wfgm3_diff, wwfga_diff,
             wwfga3_diff, wfta_diff, wwor_diff, wdr_diff, wast_diff,
             wto_diff, wstl_diff, wblk_diff, wpf_diff, wftm, avgscore_diff2, avg_score_eaten_diff2, wfgm_diff2,
             wfgm3_diff2, wwfga_diff2, wwfga3_diff2, wfta_diff2, wwor_diff2, wdr_diff2, wast_diff2,
             wto_diff2, wstl_diff2, wblk_diff2, wpf_diff2, wftm2, avgscore_diff3, avg_score_eaten_diff3, wfgm_diff3,
             wfgm3_diff3, wwfga_diff3, wwfga3_diff3, wfta_diff3, wwor_diff3, wdr_diff3, wast_diff3, wto_diff3,
             wstl_diff3, wblk_diff3, wpf_diff3, wftm3,
             fgm_per_fga, fgm3_per_fga3,
             ftm_per_fta, ast_to_ratio, defensive_efficiency, possessions, true_shooting, net_rating, pace,
             offensive_rating, defensive_rating, efg_percentage, ft_rate,
             to_percentage, reb_percentage, ast_percentage, stl_percentage, blk_percentage,
             # Yeni özellikler aşağıda eklenmiştir
             pace_consistency, ft_efficiency, inside_outside_balance, defensive_pressure, offensive_reb_efficiency,
             assist_quality, game_control, foul_efficiency, perimeter_defense, defense_to_offense,q25_diff,q75_diff,q25_diff3,q75_diff3,q25_diff4,
             q75_diff4, turnover_ratio, offensive_rebound_percentage,shot_selection_quality,ft_drawing_ability,defensive_discipline,inside_outside_variance,
             offensive_efficiency_combo,game_control_consistency,steal_to_points_efficiency,score_volatility,second_chance_efficiency,transition_effectiveness,
             iqr_fgm,iqr_score,q10_q90_range,interior_scoring_efficiency
             ])


        return seed_diff, winrate_diff, avgscore_diff


    def __getitem__(self, idx):
        prev_games = self.games

        team1 = self.team1_idxs[idx]
        team2 = self.team2_idxs[idx]


        #team1 = row["WTeamID"]
        #team2 = row["LTeamID"]



        team1_wgames = prev_games[prev_games["WTeamID"] == team1]

        team1_lgames = prev_games[prev_games["LTeamID"] == team1]

        team2_wgames = prev_games[prev_games["WTeamID"] == team2]

        team2_lgames = prev_games[prev_games["LTeamID"] == team2]

        #team1_wavg_score = team1_wgames["WScore"].mean()
        #team1_lavg_score = team1_lgames["LScore"].mean()
        team1_avg_score = pd.concat([team1_wgames["WScore"], team1_lgames["LScore"]])
        team1_avg_eaten_score = pd.concat([team1_wgames["LScore"], team1_lgames["WScore"]])
        team1_WFGM = pd.concat([team1_wgames["WFGM"], team1_lgames["LFGM"]])
        team1_WFGM3 = pd.concat([team1_wgames["WFGM3"], team1_lgames["LFGM3"]])
        team1_WFGA = pd.concat([team1_wgames["WFGA"], team1_lgames["LFGA"]])
        team1_WFGA3 = pd.concat([team1_wgames["WFGA3"], team1_lgames["LFGA3"]])
        team1_WFTM = pd.concat([team1_wgames["WFTM"], team1_lgames["LFTM"]])
        team1_WFTA = pd.concat([team1_wgames["WFTA"], team1_lgames["LFTA"]])
        team1_WOR = pd.concat([team1_wgames["WOR"], team1_lgames["LOR"]])
        team1_WDR = pd.concat([team1_wgames["WDR"], team1_lgames["LDR"]])
        team1_WAst = pd.concat([team1_wgames["WAst"], team1_lgames["LAst"]])
        team1_WTO = pd.concat([team1_wgames["WTO"], team1_lgames["LTO"]])
        team1_WStl = pd.concat([team1_wgames["WStl"], team1_lgames["LStl"]])
        team1_WBlk = pd.concat([team1_wgames["WBlk"], team1_lgames["LBlk"]])
        team1_WPF = pd.concat([team1_wgames["WPF"], team1_lgames["LPF"]])

        '''_,metrics,_ = self.hesapla_team1_karakteristik_metrikleri(team1_avg_score,team1_avg_eaten_score,team1_WFGM,team1_WFGM3,team1_WFGA,team1_WFGA3,
                                                    team1_WFTM,team1_WFTA,team1_WOR,team1_WDR,team1_WAst,team1_WTO,team1_WStl,team1_WBlk,team1_WPF
                                                    )'''



        #team2_wavg_score = team2_wgames["WScore"].mean()
        #team2_lavg_score = team2_lgames["LScore"].mean()
        team2_avg_score = pd.concat([team2_wgames["WScore"], team2_lgames["LScore"]])
        team2_avg_score_eaten = pd.concat([team2_wgames["LScore"], team2_lgames["WScore"]])

        team2_WFGM = pd.concat([team2_wgames["WFGM"], team2_lgames["LFGM"]])
        team2_WFGM3 = pd.concat([team2_wgames["WFGM3"], team2_lgames["LFGM3"]])
        team2_WFGA = pd.concat([team2_wgames["WFGA"], team2_lgames["LFGA"]])
        team2_WFGA3 = pd.concat([team2_wgames["WFGA3"], team2_lgames["LFGA3"]])
        team2_WFTM = pd.concat([team2_wgames["WFTM"], team2_lgames["LFTM"]])
        team2_WFTA = pd.concat([team2_wgames["WFTA"], team2_lgames["LFTA"]])
        team2_WOR = pd.concat([team2_wgames["WOR"], team2_lgames["LOR"]])
        team2_WDR = pd.concat([team2_wgames["WDR"], team2_lgames["LDR"]])
        team2_WAst = pd.concat([team2_wgames["WAst"], team2_lgames["LAst"]])
        team2_WTO = pd.concat([team2_wgames["WTO"], team2_lgames["LTO"]])
        team2_WStl = pd.concat([team2_wgames["WStl"], team2_lgames["LStl"]])
        team2_WBlk = pd.concat([team2_wgames["WBlk"], team2_lgames["LBlk"]])
        team2_WPF = pd.concat([team2_wgames["WPF"], team2_lgames["LPF"]])

        '''_,metrics2,_ = self.hesapla_team1_karakteristik_metrikleri(team2_avg_score, team2_avg_score_eaten, team2_WFGM,
                                                              team2_WFGM3, team2_WFGA, team2_WFGA3,
                                                              team2_WFTM, team2_WFTA, team2_WOR, team2_WDR, team2_WAst,
                                                              team2_WTO, team2_WStl, team2_WBlk, team2_WPF
                                                              )'''



        team1_wgames_len = len(team1_wgames)
        team1_lgames_len = len(team1_lgames)

        team2_wgames_len = len(team2_wgames)
        team2_lgames_len = len(team2_lgames)

        team1_total_games= team1_wgames_len + team1_lgames_len
        team2_total_games= team2_wgames_len + team2_lgames_len

        if team1_total_games == 0:
            team1_win_rate = 0.5
        else:
            if team1_total_games != 0 and team1_wgames_len == 0:
                team1_win_rate = max(0.1,0.5 - (team1_lgames_len * 0.05))
            else:
                team1_win_rate = (team1_wgames_len + self.k * self.global_winrate) / (team1_total_games + self.k)

        if team2_total_games == 0:
            team2_win_rate = 0.5
        else:
            if team2_total_games != 0 and team2_wgames_len == 0:
                team2_win_rate = max(0.1, 0.5 - (team2_lgames_len * 0.05))
            else:
                team2_win_rate = (team2_wgames_len + self.k * self.global_winrate) / (team2_total_games + self.k)



        team1_seeds = self.womens_df_seeds[self.womens_df_seeds["TeamID"] == team1]
        team1_avg_seed = team1_seeds["Seed"].mean()


        team2_seeds = self.womens_df_seeds[self.womens_df_seeds["TeamID"] == team2]
        team2_avg_seed = team2_seeds["Seed"].mean()

        '''if np.isnan(team1_avg_seed):
            team1_avg_seed = 8

        if np.isnan(team2_avg_seed):
            team2_avg_seed = 8'''
        if np.isnan(team1_avg_seed):
            team1_avg_seed = 9.888282820954

        if np.isnan(team2_avg_seed):
            team2_avg_seed = 9.888282820954



        self.calculate_diffs(team1_avg_seed, team2_avg_seed, team1_win_rate, team2_win_rate, team1_avg_score,
                             team2_avg_score, team1_avg_eaten_score, team2_avg_score_eaten,team1_WFGM, team2_WFGM,team1_WFGM3,team2_WFGM3,team1_WFGA,team2_WFGA,
                               team1_WFGA3, team2_WFGA3, team1_WFTA,team2_WFTA, team1_WOR,team2_WOR,team1_WDR,team2_WDR,team1_WAst,team2_WAst,team1_WTO,team2_WTO,
                               team1_WStl, team2_WStl, team1_WBlk,team2_WBlk,team1_WPF,team2_WPF, team1_WFTM, team2_WFTM)



        return [team1_avg_score.mean(), team2_avg_score.mean(), team1_avg_seed, team2_avg_seed]

dataset = MMarchManiaDataset()
for x in range(0, len(dataset)):
    print(x)
    output = dataset[x]


np.save("sub_dataw.npy",np.array(dataset.data))
np.save("sub_labelsw.npy",np.array(dataset.labels))


print(dataset[50009])



