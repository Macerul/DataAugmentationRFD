import re
from pathlib import Path


def extract_attributes_from_csv(csv_path):
    """Estrae tutti gli attributi dal CSV (header)"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            # Rimuovi eventuali spazi e split per virgola
            attributes = [attr.strip() for attr in header.split(',')]
            return attributes
    except Exception as e:
        print(f"Errore nella lettura del CSV {csv_path}: {e}")
        return []


def parse_rfd_file(log_path, csv_path):
    """Parse del file log e conversione nel nuovo formato"""

    # Leggi il file log
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Trova la sezione RFDs
    rfd_section_start = -1
    for i, line in enumerate(lines):
        if line.strip() == "RFDs:":
            rfd_section_start = i + 1
            break

    if rfd_section_start == -1:
        print(f"Sezione RFDs non trovata in {log_path}")
        return []

    # Estrai tutte le RFDs
    rfds = []
    empty_lhs_rfds = []  # RFDs con lato sinistro vuoto (-> AttrX)

    for i in range(rfd_section_start, len(lines)):
        line = lines[i].strip()

        # Fermati quando trovi altre sezioni
        if line.startswith("Time limit:") or line.startswith("Out of memory:") or \
                line.startswith("Memory consumption:") or line.startswith("RFDs count:"):
            break

        if not line or line == "":
            continue

        # Parse della RFD
        if "->" in line:
            parts = line.split("->")
            lhs = parts[0].strip()
            rhs = parts[1].strip()

            # Ignora se rhs contiene "class"
            if "class" in rhs.lower():
                continue

            if lhs == "":
                # Caso speciale: -> AttrX
                empty_lhs_rfds.append(rhs)
            else:
                # Caso normale: separa gli attributi (possono essere separati da tab o spazi)
                lhs_attrs = [attr.strip() for attr in re.split(r'\s+', lhs) if attr.strip()]

                # Ignora se qualche attributo in lhs contiene "class"
                if any("class" in attr.lower() for attr in lhs_attrs):
                    continue

                rfds.append((lhs_attrs, rhs))

    # Gestisci le RFDs con lhs vuoto
    if empty_lhs_rfds:
        # Estrai tutti gli attributi dal CSV
        all_attrs = extract_attributes_from_csv(csv_path)

        if all_attrs:
            # Per ogni attributo target con lhs vuoto
            for target_attr in empty_lhs_rfds:
                # Estrai il numero dell'attributo (es: Attr1 da Attr1@2.0)
                target_num = re.search(r'Attr(\d+)', target_attr)
                if target_num:
                    target_num = target_num.group(1)

                    # Crea RFDs per tutti gli altri attributi
                    for csv_attr in all_attrs:
                        # Converti l'attributo CSV nel formato AttrX@Y.0
                        # Estrai la soglia dal target_attr
                        threshold = re.search(r'@([\d.]+)', target_attr)
                        threshold_val = threshold.group(1) if threshold else "2.0"

                        # Determina il numero dell'attributo dal CSV
                        # Assumiamo che gli attributi siano Attr1, Attr2, etc.
                        # o prendiamo il nome direttamente se già in formato corretto
                        if csv_attr.startswith('Attr'):
                            attr_num = re.search(r'Attr(\d+)', csv_attr)
                            if attr_num and attr_num.group(1) != target_num:
                                source_attr = f"Attr{attr_num.group(1)}@{threshold_val}"
                                rfds.append(([source_attr], target_attr))
                        # Ignora class completamente
                        # elif csv_attr.lower() == 'class':
                        #     if target_num != 'class':
                        #         source_attr = f"class@{threshold_val}"
                        #         rfds.append(([source_attr], target_attr))

    return rfds


def format_output(rfds):
    """Formatta le RFDs nel nuovo formato"""
    output_lines = ["****** DISCOVERED RFDs *******"]

    for lhs_attrs, rhs in rfds:
        # Unisci gli attributi del lhs con virgole
        lhs_formatted = ",".join(lhs_attrs) + ","
        output_lines.append(f"{lhs_formatted} -> {rhs}")

    return "\n".join(output_lines)


def process_files(input_dir, csv_dir, output_dir=None):
    """Processa tutti i file log nella directory"""

    if output_dir is None:
        output_dir = input_dir

    # Crea la directory di output se non esiste
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Trova tutti i file .log
    log_files = list(Path(input_dir).glob("*_true.log"))

    print(f"Trovati {len(log_files)} file da processare")

    for log_path in log_files:
        print(f"\nProcessando: {log_path.name}")

        # Estrai il nome del dataset dal nome del file log
        # Formato: LiverCirrhosis_onevsrest_1_min.csv_2.0_0.0_true.log
        log_name = log_path.name
        match = re.match(r'(.+\.csv)_([\d.]+)_0\.0_true\.log', log_name)

        if not match:
            print(f"  Formato nome file non riconosciuto: {log_name}")
            continue

        csv_name = match.group(1)
        threshold = match.group(2)

        # Costruisci il path del CSV nella cartella imbalanced_datasets
        csv_path = Path(csv_dir) / csv_name

        if not csv_path.exists():
            print(f"  ATTENZIONE: CSV non trovato: {csv_path}")
            print(f"  Continuo senza gestire le RFDs con lhs vuoto")

        # Parse del file
        rfds = parse_rfd_file(str(log_path), str(csv_path) if csv_path.exists() else None)

        if not rfds:
            print(f"  Nessuna RFD trovata")
            continue

        # Formatta l'output
        output_content = format_output(rfds)

        # Crea il nome del file di output
        output_name = f"RFD{threshold.split('.')[0]}_E0.0_{csv_name.split('.csv')[0]}.txt"
        output_path = Path(output_dir) / output_name

        # Scrivi il file di output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)

        print(f"  ✓ Creato: {output_name} ({len(rfds)} RFDs)")


# Esempio d'uso
if __name__ == "__main__":
    # Specifica le directory
    input_directory = "discovered_rfds"  # Directory con i file .log, modifica se necessario
    csv_directory = "./imbalanced_datasets"  # Directory con i file CSV
    output_directory = "./discovered_rfds/discovered_rfds_processed"  # Directory di output, modifica se necessario

    process_files(input_directory, csv_directory, output_directory)

    print("\n✓ Conversione completata!")