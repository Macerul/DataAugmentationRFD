import itertools
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("aggregated2.csv", sep=";")


metrics = ["Precision", "Recall", "F1-Score", "G mean", "Balanced Accuracy"]

def boxplot():
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), constrained_layout=True)

    for i, m in enumerate(metrics):
        ax = axes.flatten()[i]
        df.boxplot(column=m, by="method", grid=False, ax=ax)
        ax.set_title(m)
        ax.set_xlabel("Metodo")
        ax.set_ylabel(m)

    fig.suptitle("Distribuzione delle metriche per metodo", fontsize=14)
    plt.show()

def barplot():
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", context="talk")

    # === Mappatura nomi e ordine fisso ===
    method_order = ["smote", "SMOTECDNN", "casTGAN", "2", "4", "8","llama","deepseek"]
    method_labels = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "CasTGAN",
        "2": "SYRFD ($\phi$ = 2)",
        "4": "SYRFD ($\phi$ = 4)",
        "8": "SYRFD ($\phi$ = 8)",
        "llama": "llama",
        "deepseek": "deepseek"
    }

    # Applica mappatura dei nomi leggibili
    df_plot = df.copy()
    df_plot["method_label"] = df_plot["method"].map(method_labels)

    # 🔹 Figura più larga ma con grafici più bassi
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22, 4))
    axes = axes.flatten()

    # Palette coerente
    palette = sns.color_palette("crest", n_colors=len(method_order))

    for i, m in enumerate(metrics):
        ax = axes[i]

        # Calcola la media per metodo
        mean_scores = (
            df_plot.groupby(["method", "method_label"])[m]
            .mean()
            .reindex(method_order, level="method")
            .reset_index()
        )
        print(mean_scores)
        sns.barplot(
            data=mean_scores,
            x="method_label",
            y=m,
            order=[method_labels[o] for o in method_order],
            ax=ax,
            palette=palette
        )
        # Trova il valore massimo della metrica
        max_val = mean_scores[m].max()
        max_row = mean_scores.loc[mean_scores[m].idxmax()]
        max_label = max_row["method_label"]
        max_x = list(mean_scores["method_label"]).index(max_label)

        # Aggiungi una linea orizzontale tratteggiata rossa al valore massimo
        ax.axhline(y=max_val, color="red", linestyle="--", linewidth=1.5, alpha=0.8)

        # === 🔢 Valore numerico in corrispondenza del metodo massimo ===
        ax.text(
            max_x, max_val + 0.02,  # posizionato sopra la barra corretta
            f"{max_val:.3f}",
            color="red", fontsize=14, fontweight="bold",
            ha="center", va="bottom"
        )

        # Titoli e assi
        ax.set_title(m, fontsize=20, fontweight="bold")

        '''# Asse Y
        if m == "balanced_accuracy":
            ax.set_ylabel("Balanced Accuracy", fontsize=11)
        else:
            ax.set_ylabel(m, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        '''
        # Etichette asse X: verticali, più visibili
        ax.tick_params(axis="x", labelrotation=90, labelsize=18)
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        # Asse Y fisso da 0 a 1
        ax.set_ylim(0, 1)

        # Rimuovo label asse X individuali
        ax.set_xlabel("")
        ax.set_ylabel("")

        # 🔹 Riduco lo spazio verticale tra barre e label
        ax.margins(y=0.1)

    # Layout compatto ma più leggibile
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)
    #plt.savefig(f'./classification_results_barplot.pdf', bbox_inches='tight')
    plt.show()

def barplot_horizontal():
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", context="talk")

    # === Mappatura nomi e ordine fisso ===
    method_order = ["smote", "SMOTECDNN", "casTGAN", "2", "4", "8"]
    method_labels = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "CasTGAN",
        "2": "SYRFD ($\phi$ = 2)",
        "4": "SYRFD ($\phi$ = 4)",
        "8": "SYRFD ($\phi$ = 8)"
    }

    # Applica mappatura dei nomi leggibili
    df_plot = df.copy()
    df_plot["method_label"] = df_plot["method"].map(method_labels)

    # 🔹 Figura più larga ma con grafici più bassi
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22, 4))
    axes = axes.flatten()

    # Palette coerente
    palette = sns.color_palette("crest", n_colors=len(method_order))

    for i, m in enumerate(metrics):
        ax = axes[i]

        # Calcola la media per metodo
        mean_scores = (
            df_plot.groupby(["method", "method_label"])[m]
            .mean()
            .reindex(method_order, level="method")
            .reset_index()
        )
        print(mean_scores)
        sns.barplot(
            data=mean_scores,
            x="method_label",
            y=m,
            order=[method_labels[o] for o in method_order],
            ax=ax,
            palette=palette
        )

        # Titoli e assi
        ax.set_title(m, fontsize=15, fontweight="bold")

        '''# Asse Y
        if m == "balanced_accuracy":
            ax.set_ylabel("Balanced Accuracy", fontsize=11)
        else:
            ax.set_ylabel(m, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        '''
        # Etichette asse X: verticali, più visibili
        ax.tick_params(axis="x", labelrotation=90, labelsize=13)
        # Asse Y fisso da 0 a 1
        ax.set_ylim(0, 1)

        # Rimuovo label asse X individuali
        ax.set_xlabel("")
        ax.set_ylabel("")

        # 🔹 Riduco lo spazio verticale tra barre e label
        ax.margins(y=0.1)

    # Layout compatto ma più leggibile
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)
    plt.savefig(f'./classification_results_barplot.pdf', bbox_inches='tight')
    plt.show()


def plot_syrfd_heatmap(csv_file, metriche=None, thresholds=[2,4,8], figsize=(10,6), cmap="viridis"):
    """
    Genera una heatmap dei valori delle metriche per i diversi threshold di SYRFD.

    Args:
        csv_file (str): Percorso del CSV contenente le metriche (media ± std).
        metriche (list, optional): Lista di metriche da visualizzare. Default: tutte.
        thresholds (list, optional): Lista dei threshold di SYRFD da confrontare. Default: [2,4,8].
        figsize (tuple, optional): Dimensioni della figura. Default: (10,6).
        cmap (str, optional): Colormap per la heatmap. Default: "viridis".

    Returns:
        None. Mostra la heatmap.
    """
    # Leggi CSV
    df = pd.read_csv(csv_file, sep=";")

    # Se non specificato, prendi tutte le metriche (escluse colonne "metodo")
    if metriche is None:
        metriche = [c for c in df.columns if c != "metodo"]

    # Funzione per estrarre solo la media dai valori "media ± std"
    def estrai_media(val):
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        else:
            return float(val)

    for m in metriche:
        df[m+"_mean"] = df[m].apply(estrai_media)

    # Filtra solo SYRFD
    df_syrfd = df[df["metodo"].str.contains("SYRFD")]

    # Crea dataframe per heatmap: threshold come righe, metriche come colonne
    df_heatmap = pd.DataFrame(
        {thr: df_syrfd[df_syrfd["metodo"]==f"SYRFD_thr{thr}"][[m+"_mean" for m in metriche]].values.flatten()
         for thr in thresholds}
    ).T
    df_heatmap.columns = metriche

    # Normalizzazione 0-1 per comparabilità
    df_heatmap_norm = (df_heatmap - df_heatmap.min()) / (df_heatmap.max() - df_heatmap.min())

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df_heatmap_norm, annot=True, cmap=cmap, cbar_kws={'label': 'Normalized value'})
    plt.title("SYRFD: Sensitivity of metrics to thresholds")
    plt.ylabel("Threshold")
    plt.xlabel("Metrics")
    plt.show()


def plot_metrics_per_threshold(csv_file, method_prefix="SYRFD_thr", metrics=None):
    """
    Crea un unico plot con subplot separati per ogni metrica, mostrando i valori dei threshold per SYRFD.

    Args:
        csv_file (str): percorso del file CSV con le metriche (colonna 'metodo' inclusa).
        method_prefix (str): prefisso dei metodi SYRFD nel CSV.
        metrics (list of str, optional): lista delle metriche da plottare. Se None, plotta tutte tranne 'metodo'.
    """
    df = pd.read_csv(csv_file, sep=';')

    # Filtra solo le righe dei threshold SYRFD
    df_syrfd = df[df['metodo'].str.startswith(method_prefix)].copy()

    # Rimuovi il ± dai valori e converti a float
    for col in df_syrfd.columns:
        if col != "metodo":
            df_syrfd[col] = df_syrfd[col].str.split(' ±').str[0].astype(float)

    # Se non specificato, usa tutte le metriche tranne 'metodo'
    if metrics is None:
        metrics = [c for c in df_syrfd.columns if c != "metodo"]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    thresholds = df_syrfd['metodo'].tolist()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df_syrfd[metric].tolist()
        sns.barplot(x=thresholds, y=values, palette="viridis", ax=ax)
        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.set_xlabel("Threshold")
        ax.set_xticklabels(thresholds, rotation=45, ha='right')

    # Nascondi eventuali subplot vuoti
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()


def plot_heatmaps_per_metric(csv_file, method_prefix="SYRFD_thr", metrics=None, normalize=False):
    """
    Crea un unico plot con subplot, una heatmap per ogni metrica,
    mostrando i valori dei threshold per SYRFD.

    Args:
        csv_file (str): percorso del file CSV con le metriche (colonna 'metodo' inclusa).
        method_prefix (str): prefisso dei metodi SYRFD nel CSV.
        metrics (list of str, optional): lista delle metriche da plottare. Se None, plottare tutte tranne 'metodo'.
        normalize (bool): se True normalizza i valori per colonna tra 0 e 1.
    """
    df = pd.read_csv(csv_file, sep=';')

    # Filtra solo le righe dei threshold SYRFD
    df_syrfd = df[df['metodo'].str.startswith(method_prefix)].copy()

    # Rimuovi il ± dai valori e converti a float
    for col in df_syrfd.columns:
        if col != "metodo":
            df_syrfd[col] = df_syrfd[col].str.split(' ±').str[0].astype(float)

    # Se non specificato, usa tutte le metriche tranne 'metodo'
    if metrics is None:
        metrics = [c for c in df_syrfd.columns if c != "metodo"]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    # Indici: threshold, colonne: valori (1 colonna per "metrica")
    thresholds = df_syrfd['metodo'].tolist()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = df_syrfd[[metric]].copy()
        data.index = thresholds

        if normalize:
            # Normalizza tra 0 e 1 per rendere comparabili le scale
            min_val = data[metric].min()
            max_val = data[metric].max()
            if max_val - min_val != 0:
                data[metric] = (data[metric] - min_val) / (max_val - min_val)
            else:
                data[metric] = 0.5  # caso costante

        sns.heatmap(data.T, annot=True, fmt=".2f", cmap="viridis", cbar=True, ax=ax)
        ax.set_title(metric)
        ax.set_ylabel("")
        ax.set_xlabel("Threshold")

    # Nascondi eventuali subplot vuoti
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()


def plot_grouped_bar(csv_file, metrics=None):
    """
    Crea un grouped bar plot per ciascuna metrica (2 righe x 3 colonne).

    Args:
        csv_file (str): percorso del CSV con colonne: metodo;metric1;metric2;...
        metrics (list of str, optional): metriche da plottare. Se None, usa tutte tranne 'metodo'.
    """
    df = pd.read_csv(csv_file, sep=";")
    df = df[df["metodo"].str.startswith("SYRFD")]  # filtra solo SYRFD
    if metrics is None:
        metrics = [c for c in df.columns if c != "metodo"]

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # Estrae la media dal formato "mean ± std"
        df[metric + "_mean"] = df[metric].str.split(" ± ").str[0].astype(float)

        sns.barplot(x="metodo", y=metric + "_mean", data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(metric)
        axes[i].set_xticklabels(df["metodo"], rotation=45, ha="right")
        axes[i].set_ylabel("Value")

    # rimuove assi vuoti se metriche < 6
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_line_metrics(csv_file, metrics=None):
    """
    Crea line plot con marker per ogni metrica (2 righe x 3 colonne).

    Args:
        csv_file (str): percorso del CSV con colonne: metodo;metric1;metric2;...
        metrics (list of str, optional): metriche da plottare. Se None, usa tutte tranne 'metodo'.
    """
    df = pd.read_csv(csv_file, sep=";")
    df = df[df["metodo"].str.startswith("SYRFD")]  # filtra solo SYRFD
    if metrics is None:
        metrics = [c for c in df.columns if c != "metodo"]

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # Estrae la media dal formato "mean ± std"
        df[metric + "_mean"] = df[metric].str.split(" ± ").str[0].astype(float)

        axes[i].plot(df["metodo"], df[metric + "_mean"], marker="o", linestyle="-", color="tab:blue")
        axes[i].set_title(metric)
        axes[i].set_xticklabels(df["metodo"], rotation=45, ha="right")
        axes[i].set_ylabel("Value")

    # rimuove assi vuoti se metriche < 6
    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def barplot_all_results():
    # === Lettura dati ===
    df = pd.read_csv("aggregated.csv", sep=";")

    # === Parametri di base ===
    metrics = ["Precision", "Recall", "F1-Score", "G mean", "Balanced Accuracy"]
    datasets = sorted(df["dataset"].unique())
    methods = ["smote", "SMOTECDNN", "casTGAN", "2", "4", "8"]
    models = sorted(df["model"].unique())

    method_labels = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "CasTGAN",
        "2": "SYRFD (ϕ=2)",
        "4": "SYRFD (ϕ=4)",
        "8": "SYRFD (ϕ=8)"
    }

    palette = sns.color_palette("tab10", n_colors=len(models))

    # === 🔧 PARAMETRI MODIFICABILI ===
    subplot_width = 4   # larghezza singolo subplot
    subplot_height = 1  # altezza singolo subplot
    bar_width = 0.5     # larghezza barre
    share_y_axes = True # condividere asse Y tra metriche

    # Calcolo dimensione totale figura
    fig_width = subplot_width * len(datasets)
    fig_height = subplot_height * len(metrics)

    print(f"📐 Figura totale: {fig_width:.1f} x {fig_height:.1f} pollici")

    # === Figura complessiva ===
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(datasets),
        figsize=(fig_width, fig_height),
        sharey='row' if share_y_axes else False
    )

    if len(metrics) == 1:
        axes = np.expand_dims(axes, axis=0)

    # === Loop su metriche e dataset ===
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]

            # Filtra i dati del dataset
            df_sub = df[df["dataset"] == dataset]

            sns.barplot(
                data=df_sub,
                x="method",
                y=metric,
                hue="model",
                hue_order=models,
                order=methods,
                ax=ax,
                palette=palette,
                width=bar_width
            )

            # === Layout e testi ===
            if i == 0:
                ax.set_title(dataset, fontsize=6, fontweight="bold", rotation=45, pad=6)
            else:
                ax.set_title("")

            # Mostra la label Y solo nella prima colonna
            if j == 0:
                ax.set_ylabel(metric, fontsize=8, fontweight="bold")
            else:
                ax.set_ylabel("")

            # === Mostra le label X solo nell’ultima riga ===
            if i == len(metrics) - 1:
                ax.set_xticklabels(
                    [method_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                    rotation=90, fontsize=5, fontweight="bold"
                )
            else:
                ax.set_xticklabels([])

            ax.set_xlabel("")
            ax.tick_params(axis="x", pad=1)
            ax.set_ylim(0, 1)
            ax.margins(x=0.05)

            # Rimuove tutte le legende locali
            ax.get_legend().remove()

    # === Legenda globale unica (in alto) ===
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(models),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=6,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'./classification_results_all_dataset.pdf', bbox_inches='tight')
    plt.show()

def barplot_mean_over_datasets():
    # === Lettura dati ===
    df = pd.read_csv("aggregated.csv", sep=";")

    # === Parametri base ===
    metrics = ["Precision", "Recall", "F1-Score", "G mean", "Balanced Accuracy"]
    methods = ["smote", "SMOTECDNN", "casTGAN", "2", "4", "8"]
    models = sorted(df["model"].unique())

    method_labels = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "CasTGAN",
        "2": "SYRFD (ϕ=2)",
        "4": "SYRFD (ϕ=4)",
        "8": "SYRFD (ϕ=8)"
    }

    # === 🔹 Colori coerenti per metodo ===
    method_palette = sns.color_palette("tab10", n_colors=len(methods))
    color_map = dict(zip(methods, method_palette))

    # === 🔹 Calcolo media sui dataset ===
    df_mean = (
        df.groupby(["method", "model"])
        [metrics].mean()
        .reset_index()
    )

    # === 🔧 Parametri grafici ===
    subplot_width = 1.8
    subplot_height = 1.6
    bar_width = 0.8

    fig_width = subplot_width * len(models)
    fig_height = subplot_height * len(metrics)

    print(f"📐 Figura totale: {fig_width:.1f} x {fig_height:.1f} pollici")

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(models),
        figsize=(fig_width, fig_height),
        sharey='row'
    )

    if len(metrics) == 1:
        axes = np.expand_dims(axes, axis=0)

    # === Loop su metriche e modelli ===
    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            ax = axes[i, j]
            df_sub = df_mean[df_mean["model"] == model]

            sns.barplot(
                data=df_sub,
                x="method",
                y=metric,
                order=methods,
                ax=ax,
                palette=[color_map[m] for m in methods],
                width=bar_width
            )

            # === Layout ===
            if i == 0:
                ax.set_title(model, fontsize=8, fontweight="bold", pad=6)
            else:
                ax.set_title("")

            if j == 0:
                ax.set_ylabel(metric, fontsize=8, fontweight="bold")
            else:
                ax.set_ylabel("")

            # Label asse X solo nell’ultima riga
            if i == len(metrics) - 1:
                ax.set_xticklabels(
                    [method_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                    rotation=90, fontsize=6, fontweight="bold"
                )
            else:
                ax.set_xticklabels([])

            ax.set_xlabel("")
            ax.set_ylim(0, 1)
            ax.margins(x=0.05)

    # === Legenda globale unica in alto (6 colonne) ===
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[m]) for m in methods
    ]
    fig.legend(
        legend_handles,
        [method_labels[m] for m in methods],
        loc='upper center',
        ncol=len(methods),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    #plt.savefig(f'./classification_results_for_model.pdf', bbox_inches='tight')
    plt.show()

def barplot_mean_over_models():
    # === Lettura dati ===
    df = pd.read_csv("aggregated.csv", sep=";")

    # === Parametri base ===
    metrics = ["Precision", "Recall", "F1-Score", "G mean", "Balanced Accuracy"]
    methods = ["smote", "SMOTECDNN", "casTGAN", "2", "4", "8"]
    datasets = sorted(df["dataset"].unique())

    method_labels = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "CasTGAN",
        "2": "SYRFD (ϕ=2)",
        "4": "SYRFD (ϕ=4)",
        "8": "SYRFD (ϕ=8)"
    }

    # === 🎨 Palette coerente per metodo ===
    method_palette = sns.color_palette("tab10", n_colors=len(methods))
    color_map = dict(zip(methods, method_palette))

    # === 🔹 Calcolo media sui modelli ===
    df_mean = (
        df.groupby(["dataset", "method"])
        [metrics].mean()
        .reset_index()
    )

    # === 🔧 Parametri grafici ===
    subplot_width = 0.7   # larghezza singolo subplot
    subplot_height = 1.2  # altezza singolo subplot
    bar_width = 0.4       # spessore barre

    fig_width = subplot_width * len(datasets)
    fig_height = subplot_height * len(metrics)

    print(f"📐 Figura totale: {fig_width:.1f} x {fig_height:.1f} pollici")

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(datasets),
        figsize=(fig_width, fig_height),
        sharey='row',

    )


    if len(metrics) == 1:
        axes = np.expand_dims(axes, axis=0)

    # === Loop su metriche e dataset ===
    for i, metric in enumerate(metrics):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            df_sub = df_mean[df_mean["dataset"] == dataset]

            sns.barplot(
                data=df_sub,
                x="method",
                y=metric,
                order=methods,
                ax=ax,
                palette=[color_map[m] for m in methods],
                width=bar_width
            )

            # === Layout ===
            if i == 0:
                ax.set_title(dataset, fontsize=6, fontweight="bold", rotation=45, pad=4)
            else:
                ax.set_title("")

            if j == 0:
                ax.set_ylabel(metric, fontsize=8, fontweight="bold")
            else:
                ax.set_ylabel("")

            # Label asse X solo nell’ultima riga
            if i == len(metrics) - 1:
                ax.set_xticklabels(
                    [method_labels.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                    rotation=90, fontsize=6, fontweight="bold"
                )
            else:
                ax.set_xticklabels([])

            ax.set_xlabel("")
            ax.set_ylim(0.4, 1.05)
            ax.margins(x=0.05)

    # === Legenda globale unica (in alto, 6 colonne) ===
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[m]) for m in methods
    ]
    fig.legend(
        legend_handles,
        [method_labels[m] for m in methods],
        loc='upper center',
        ncol=len(methods),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'./classification_results_for_dataset.pdf', bbox_inches='tight')
    plt.show()


def scatter_precision_recall_per_dataset():
    # === Lettura dati ===
    df = pd.read_csv("aggregated.csv", sep=";")

    # === Parametri base ===
    datasets = sorted(df["dataset"].unique())
    methods = ["smote", "SMOTECDNN", "casTGAN", "2", "4", "8"]
    models = sorted(df["model"].unique())

    method_labels = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "CasTGAN",
        "2": "SYRFD (ϕ=2)",
        "4": "SYRFD (ϕ=4)",
        "8": "SYRFD (ϕ=8)"
    }
    # Marker distinti per ogni metodo
    method_markers = ["o", "s", "D", "^", "v", "P"]  # 6 marker diversi
    marker_map = dict(zip(methods, method_markers))

    # Colore per metodo
    base_palette = sns.color_palette("tab10", n_colors=len(methods))
    color_map = dict(zip(methods, base_palette))

    # Lettere per i modelli
    model_letters = dict(zip(models, string.ascii_uppercase))  # A, B, C, ...

    # Parametri grafici
    n_datasets = len(datasets)
    n_cols = 7
    n_rows = int(np.ceil(n_datasets / n_cols))
    subplot_width = 2.2
    subplot_height = 2.2

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(subplot_width * n_cols, subplot_height * n_rows),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # Loop su dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_sub = df[df["dataset"] == dataset]

        for method in methods:
            df_m = df_sub[df_sub["method"] == method]

            for model in models:
                df_mod = df_m[df_m["model"] == model]
                # Scatter base
                ax.scatter(
                    df_mod["Precision"],
                    df_mod["Recall"],
                    color=color_map[method],
                    marker=marker_map[method],
                    s=25,
                    edgecolor='black',
                    linewidth=0.2,
                    alpha=0.9
                )
                # Aggiunge la lettera del modello dentro il marker
                for x, y in zip(df_mod["Precision"], df_mod["Recall"]):
                    ax.text(
                        x, y, model_letters[model],
                        color='black',
                        fontsize=6,
                        ha='center',
                        va='center',
                        fontweight='bold'
                    )

        # Titoli e assi
        ax.set_title(dataset, fontsize=8, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)

        if i % n_cols == 0:
            ax.set_ylabel("Recall", fontsize=8, fontweight="bold")
        else:
            ax.set_ylabel("")

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Precision", fontsize=8, fontweight="bold")
        else:
            ax.set_xlabel("")

    # Rimuove subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Legenda superiore per i metodi
    legend_handles_methods = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color_map[m],
            markeredgecolor='black',
            markersize=8,
            label=method_labels[m]
        ) for m in methods
    ]
    fig.legend(
        handles=legend_handles_methods,
        loc='upper center',
        ncol=len(methods),
        bbox_to_anchor=(0.5, 1.03),
        fontsize=7,
        frameon=False,
        title="Metodologie",
        title_fontsize=8
    )

    # Legenda inferiore per i modelli (lettere)
    legend_handles_models = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='grey',
            markeredgecolor='black',
            markersize=8,
            label=f"{model_letters[m]} = {m}"
        ) for m in models
    ]
    fig.legend(
        handles=legend_handles_models,
        loc='lower center',
        ncol=len(models),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=6.5,
        frameon=False,
        title="Modelli (lettera nel marker)",
        title_fontsize=8
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.subplots_adjust(wspace=0.0, hspace=0.1)
    plt.savefig(f'./classification_results_scatter.pdf', bbox_inches='tight')
    plt.show()


def boxplot_metrics_per_dataset(df=None, figsize_per_subplot=(4,3)):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import math

    # ==========================
    # Parametri modificabili
    # ==========================
    csv_file = 'aggregated2.csv'  # path al CSV
    plots_per_row = 12             # numero di plot per riga
    plot_width = 0.5                # larghezza di ogni subplot
    plot_height = 1               # altezza di ogni subplot

    # Mappe personalizzate per rinominare dataset e metodi
    dataset_names = {
        'iris0': '$D_{1}$',
        'cleveland-0_vs_4': '$D_{2}$',
        'new-thyroid1': '$D_{3}$',
        'newthyroid2': '$D_{4}$',
        'ecoli-0_vs_1': '$D_{5}$',
        'ecoli1': '$D_{6}$',
        'dermatology-6': '$D_{7}$',
        'Migraine_onevsrest_0': '$D_{8}$',
        'Migraine_onevsrest_1': '$D_{9}$',
        'Migraine_onevsrest_2': '$D_{10}$',
        'Migraine_onevsrest_3': '$D_{11}$',
        'Migraine_onevsrest_4': '$D_{12}$',
        'Migraine_onevsrest_5': '$D_{13}$',
        'abalone9-18': '$D_{14}$',
        'transfusion': '$D_{15}$',
        'pima':   '$D_{16}$',
        'vowel0': '$D_{17}$',
        'yeast1':'$D_{18}$',
        'yeast3': '$D_{19}$',
        'kddcup-guess_passwd_vs_satan': '$D_{20}$',
        'Obesity_onevsrest_0':  '$D_{21}$',
        'Obesity_onevsrest_1': '$D_{22}$',
        'Obesity_onevsrest_2': '$D_{23}$',
        'Obesity_onevsrest_3': '$D_{24}$',
        'Obesity_onevsrest_4': '$D_{25}$',
        'Obesity_onevsrest_5':  '$D_{26}$',
        'Obesity_onevsrest_6': '$D_{27}$',
        'page-blocks-1-3_vs_4':'$D_{28}$'
    }

    method_names = {
        "smote": "[4]",
        "SMOTECDNN": "[39]",
        "casTGAN": "[38]",
        "2": "$\phi$=2",
        "4": "$\phi$=4",
        "8": "$\phi$=8"
    }

    method_names = {
        "smote": "SMOTE",
        "SMOTECDNN": "SMOTE-CDNN",
        "casTGAN": "casTGAN",
        "2": "SyRFD $\phi$=2",
        "4": "SyRFD $\phi$=4",
        "8": "SyRFD $\phi$=8"
    }
    # ==========================

    # Carica il CSV
    df = pd.read_csv(csv_file, sep=';')

    # Applica le mappe
    df['dataset'] = df['dataset'].map(lambda x: dataset_names.get(x, x))
    df['method'] = df['method'].map(lambda x: method_names.get(x, x))
    # (Aggiungi qui il filtro dei dataset)
    #datasets_to_plot = ['$D_{2}$', '$D_{10}$', '$D_{12}$', '$D_{13}$', '$D_{24}$', '$D_{28}$']
    #df = df[df['dataset'].isin(datasets_to_plot)]
    print(df["model"])
    # Metriche da considerare
    metrics = ['F1-Score', 'G mean', 'Balanced Accuracy']
    # Modelli e marker
    models = df['model'].unique()
    print(models)
    markers = ['$\\bigtriangledown$', 'X','$\\bigtriangleup$', '$\\bigoplus$', '.', '*', 'd', '$\spadesuit$','$\imath$']
    colors = [
        '#1f77b4',  # blu
        '#ff7f0e',  # arancione
        '#2ca02c',  # verde
        '#d62728',  # rosso
        '#9467bd',  # viola
        '#8c564b',  # marrone
        '#7f7f7f',  # grigio
        '#e377c2'  # rosa
    ]

    colors = ['#777777', '#770088', '#0000b1', '#029fcf', '#00a353', '#16f316', '#e1f309', '#ffa915', '#cc0000']
    model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}
    #model_markers = {model: markers[i % len(markers)] for i, model in enumerate(models)}

    # Metodi
    methods = df['method'].unique()
    gray_scale = ["#e0e0e0", "#c0c0c0", "#a0a0a0", "#808080", "#606060", "#404040"]
    method_edgecolors = {m: gray_scale[i % len(gray_scale)] for i, m in enumerate(methods)}

    # Ordina i dataset secondo la mappa
    ordered_datasets = [dataset_names[d] for d in dataset_names if dataset_names[d] in df['dataset'].values]

    # Numero totale di subplot
    total_plots = len(ordered_datasets) * len(metrics)
    rows = math.ceil(total_plots / plots_per_row)
    cols = plots_per_row

    # Crea figura
    fig, axes = plt.subplots(rows, cols, figsize=(20,12),#(5,15)cols * plot_width, rows * plot_height
         sharex=True, sharey=True)
    axes_flat = axes.flatten()

    plot_idx = 0

    for dataset in ordered_datasets:
        df_dataset = df[df['dataset'] == dataset]

        for metric in metrics:
            ax = axes_flat[plot_idx]
            plot_idx += 1

            # Boxplot palette="Set2",
            sns.boxplot(x='method', y=metric, data=df_dataset, ax=ax, width=0.7,boxprops=dict(facecolor='none', edgecolor='gray', linewidth=0.5),
                        whiskerprops=dict(color='gray', linewidth=0.5),
                        capprops=dict(color='gray', linewidth=0.5),
                        medianprops=dict(color='gray', linewidth=0.8),
                        flierprops=dict(marker='.', color='gray', markersize=2, alpha=0.9))

            x_center = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
            y_center = 0.2 * (ax.get_ylim()[0] + ax.get_ylim()[1])
            #ax.set_title(f"{dataset} - {metrics_dict[metric]}", fontsize=6)
            # Titolo e label
            metrics_dict = {'F1-Score':"F1", 'G mean':"G-mean", 'Balanced Accuracy': "Bal. Acc."}
            ax.text(x_center, y_center, f"{dataset} - {metrics_dict[metric]}", fontsize=16, color='grey', ha='center', va='center',zorder=1)

            ax.set_yticks([0.0,0.15,0.3, 0.45, 0.60, 0.75, 0.90, 1.0])
            # Colorazione dei bordi in scala di grigi
            for i, artist in enumerate(ax.artists):
                if i < len(methods):  # una box per metodo
                    artist.set_edgecolor(method_edgecolors[methods[i]])
                    artist.set_linewidth(1.2)

            # Sovrappone i punti dei modelli
            for k, method in enumerate(methods):
                df_subset = df_dataset[df_dataset['method'] == method]
                x = np.full(len(df_subset), k)
                for l, row in df_subset.iterrows():
                    ax.scatter(x[l - df_subset.index[0]], row[metric],
                               marker=model_markers[row['model']],
                               facecolors='none',
                               edgecolors=model_colors[row['model']],
                               linewidth=0.5, alpha=0.5,
                               s=14, zorder=5)



            #ax.set_title(f"{dataset} - {metrics_dict[metric]}", fontsize=6)
            ax.set_xlabel("")
            ax.set_ylabel("")  # rimuove label y
            # Mostra i tick y solo se è il primo plot della riga
            if plot_idx % plots_per_row != 1:  # se non è il primo plot della riga
                ax.tick_params(axis='y', labelleft=False,left=False)
            else:
                ax.tick_params(axis='y', labelsize=8)

            ax.tick_params(axis='x', labelsize=11, rotation=90)
            ax.tick_params(axis='y', labelsize=11)

    # Rimuove assi vuoti se ce ne sono
    for i in range(plot_idx, len(axes_flat)):
        fig.delaxes(axes_flat[i])
    '''
    handles = [
        plt.Line2D([0], [0],
                   marker=model_markers[m],
                   color='w',
                   label=m,
                   markerfacecolor='none',
                   markeredgecolor=model_colors[m],  # bordo colorato
                   markersize=8,
                   linewidth=0)
        for m in models
    ]
    fig.legend(handles=handles,
               loc='upper center',
               ncol=3,  # legenda su una riga
               title="Model",
               bbox_to_anchor=(0.5, 1.05),
               frameon=False)
    '''
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'./classification_results_boxplot_total.pdf', bbox_inches='tight')
    #plt.show()

#plot_syrfd_heatmap("./statistiche_metriche_per_metodo_mean_std.csv")
#barplot()
#plot_heatmaps_per_metric("./statistiche_metriche_per_metodo_mean_std.csv")
#plot_grouped_bar("./statistiche_metriche_per_metodo_mean_std.csv")
#plot_line_metrics("./statistiche_metriche_per_metodo_mean_std.csv")
#barplot_all_results()
#barplot_mean_over_datasets()
#barplot_mean_over_models()
#scatter_precision_recall_per_dataset()
boxplot_metrics_per_dataset()

