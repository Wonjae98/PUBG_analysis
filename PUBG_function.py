import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PUBGDataPreprocessor:
    def __init__(self, df):
        self.df = df

    def drop_measure_fault(self):
        conditions = (self.df['killPlace'] > 100) | (self.df['kills'] > 100) | (self.df['killStreaks'] > 100) | (self.df['revives'] > 100) | (self.df['teamKills'] > 100)
        self.df = self.df[~conditions]
        return self

    def drop_user_match(self):
        self.df = self.df[~self.df['matchType'].str.contains('normal')]
        return self

    def drop_event_match(self):
        event_type = ['crashfpp', 'flaretpp', 'flarefpp', 'crashtpp']
        self.df = self.df[~self.df['matchType'].isin(event_type)]
        return self

    def unite_match_type(self):
        self.df['matchType'] = self.df['matchType'].str.replace('-fpp', '').str.strip()
        return self

    def make_teamWork(self):
        def note_teamWork(x):
            if 'squad' in x['matchType'] or 'duo' in x['matchType']:
                return x['revives'] + x['assists'] - x['teamKills']
            else:
                return 0

        self.df['teamWork'] = self.df.apply(note_teamWork, axis=1)
        return self

    
    def make_headshotRatio(self):
        self.df['headshotRatio'] = self.df.apply(lambda x: 0 if x['kills'] == 0 else x['headshotKills'] / x['kills'], axis=1)
        return self

    def make_killRatio(self):
        userCnt = self.df.groupby('matchId')['Id'].transform('count').rename('userCnt')
        memberCnt = self.df.groupby(['matchId', 'groupId'])['Id'].transform('count').rename('memberCnt')
        
        self.df = self.df.assign(userCnt=userCnt, memberCnt=memberCnt)
        self.df['killRatio'] = self.df['kills'] / (self.df['userCnt'] - self.df['memberCnt'])
        return self

    def drop_columns(self):
        columns_to_drop = ['groupId', 'matchId', 'assists', 'headshotKills', 'kills', 'revives', 'teamKills', 'memberCnt', 'userCnt']
        self.df.drop(columns=columns_to_drop, inplace=True)
        return self

    def run_pipeline(self):
        return (
            self.drop_measure_fault()
                .drop_user_match()
                .drop_event_match()
                .unite_match_type()
                .make_teamWork()
                .make_headshotRatio()
                .make_killRatio()
                .drop_columns()
                .df
        )


def show_kdeplot(df):
    df = df.select_dtypes(include=['int', 'float'])
    plt.figure(figsize=(len(df.columns), len(df.columns)))
    row = int(len(df.columns) ** 0.5) + 1
    column = int(len(df.columns) ** 0.5)
    for idx, value in enumerate(df.columns):
        plt.subplot(row, column, idx + 1)
        plt.xlabel(value, fontsize=20)
        sns.kdeplot(data=df[value])
    plt.tight_layout()
    plt.show()


def categorical_barchart(df, colors=None):
    df = df.select_dtypes(include=['object'])
    if colors is None:
        colors = ['blue'] * len(df['matchType'].unique())  # Default color
    x = df['matchType'].value_counts().keys()
    y = df['matchType'].value_counts().values
    plt.figure(figsize=(14, 10))
    plt.bar(x=x, height=y, color=colors)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ticklabel_format(style='plain', axis='y')
    plt.xlim(-0.5, len(x) - 0.5)
    plt.xlabel('Match Type')
    plt.ylabel('Count')
    plt.show()


def show_boxplot(df):
    df = df.select_dtypes(include=['int', 'float'])
    plt.figure(figsize=(len(df.columns), len(df.columns)))
    row = int(len(df.columns) ** 0.5) + 1
    column = int(len(df.columns) ** 0.5)
    for idx, value in enumerate(df.columns):
        plt.subplot(row, column, idx + 1)
        plt.xlabel(value, fontsize=20)
        sns.boxplot(data=df[value])
    plt.tight_layout()
    plt.show()


def show_corr_matrix(df, threshold=0.5):
    corr_matrix = df.select_dtypes(include=['int', 'float']).corr()
    filtered_corr_matrix = corr_matrix[(corr_matrix.abs() >= threshold) & (corr_matrix != 1.0)].dropna(how='all', axis=0).dropna(how='all', axis=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Filtered Correlation Matrix (|corr| >= {threshold})')
    plt.show()


def show_histogram(df, color):
    df = df.select_dtypes(include=['int', 'float'])
    plt.figure(figsize=(len(df.columns), len(df.columns)))
    row = int(len(df.columns) ** 0.5) + 1
    column = int(len(df.columns) ** 0.5)
    for idx, value in enumerate(df.columns):
        plt.subplot(row, column, idx + 1)
        plt.xlabel(value, fontsize=20)
        plt.hist(df[value], color=color)
    plt.tight_layout()
    plt.show()


