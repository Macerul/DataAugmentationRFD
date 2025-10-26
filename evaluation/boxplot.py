import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("aggregated.csv", sep=";")


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



#plot_syrfd_heatmap("./statistiche_metriche_per_metodo_mean_std.csv")
barplot()
#plot_heatmaps_per_metric("./statistiche_metriche_per_metodo_mean_std.csv")
#plot_grouped_bar("./statistiche_metriche_per_metodo_mean_std.csv")
#plot_line_metrics("./statistiche_metriche_per_metodo_mean_std.csv")