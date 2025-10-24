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




barplot()