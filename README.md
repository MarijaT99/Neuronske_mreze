# Hijerarhijska klasifikacija biljnih bolesti korišćenjem dubokih neuronskih mreža

Projekat iz predmeta *Neuronske mreže*, Fakultet tehničkih nauka, Univerzitet u Novom Sadu
(master studije, Mašinsko učenje i veštačka inteligencija). Autor: Marija Tadić, E9 12/2024.

Hijerarhijska klasifikacija biljnih bolesti sa fotografija listova na skupu **PlantVillage**
(*color* verzija, 54.306 RGB slika, 14 vrsta, 38 (vrsta, bolest) klasa).

- **Nivo 1 (vrsta):** klasifikacija biljne vrste (14 klasa).
- **Nivo 2 (bolest):** tip bolesti unutar prepoznate vrste (jedan *disease* klasifikator po vrsti).
- Kao referenca trenira se i **flat** klasifikator sa svih 38 klasa - poređenje pokazuje da li
  hijerarhija donosi poboljšanje (i odsustvo poboljšanja je validan rezultat).

## Evaluacija

Skup je neuravnotežen na oba nivoa hijerarhije, pa accuracy nije primarna metrika. Korišćene metrike:

- **macro F1** (tretira sve klase jednako) - primarna metrika,
- **weighted F1** (ponderisan po broju primera),
- precision/recall po klasi + matrice konfuzije,
- **balanced accuracy** kao accuracy-alternativa prilagođena disbalansu,
- end-to-end metrike hijerarhije uz analizu propagacije greške (*error propagation*).

Accuracy se navodi isključivo kao sporedna metrika.

## Struktura projekta

```
configs/        YAML konfiguracije po eksperimentu
data/splits/    Perzistirani stratifikovani train/val/test indeksi
data/raw/       Sirove slike (van verzionisanja)
notebooks/      EDA, provera podataka, analiza grešaka
src/            Paketi: data / models / training / evaluation / utils
scripts/        prepare_splits, train_*, evaluate_pipeline, make_figures, ...
experiments/    Rezultati treniranja (istorije i metrike; težine van verzionisanja)
results/        Figure i tabele za rad
paper/          Izvorni kod rada (LaTeX, IEEE)
kaggle/         Notebook za treniranje na Kaggle platformi
Report.pdf      Finalna verzija rada
```

## Zavisnosti

- Python 3.10+ i biblioteke iz `requirements.txt` (PyTorch, torchvision, timm, albumentations,
  scikit-learn, matplotlib, seaborn, pytorch-grad-cam, ...).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate    |    Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
```

## Pokretanje

Treniranje je rađeno na **Kaggle** GPU okruženju (Tesla T4); razvoj koda lokalno.

```bash
# 1. Podaci (samo color verzija) u data/raw/plantvillage dataset/color/
#    (PlantVillage je dostupan na Kaggle platformi)

# 2. Podela je već perzistirana u data/splits/ i ne regeneriše se.
#    (Po potrebi: python scripts/prepare_splits.py --data-root "<...>/color" --out-dir data/splits --seed 42)

# 3. Flat modeli (baseline CNN i ResNet-50)
python scripts/train_flat.py --config configs/baseline_cnn_flat.yaml --device cuda
python scripts/train_flat.py --config configs/resnet50_flat.yaml --device cuda

# 4. Hijerarhija: model za vrstu + modeli za bolest po vrsti
python scripts/train_species.py --config configs/resnet50_hierarchical.yaml --device cuda
python scripts/train_disease.py --config configs/resnet50_hierarchical.yaml --device cuda

# 5. End-to-end evaluacija i poređenje flat vs hijerarhija
python scripts/evaluate_pipeline.py \
    --config configs/resnet50_hierarchical.yaml \
    --flat-config configs/resnet50_flat.yaml --device cuda

# 6. Figure za rad
python scripts/make_figures.py
python scripts/make_error_analysis.py
```

Za pokretanje na Kaggle platformi koristi se `kaggle/train_kaggle.ipynb`.

## Rezultati

Na test skupu, flat ResNet-50 postiže **macro F1 = 0,9843**, dok hijerarhijski pristup postiže
**0,9741** - flat pristup je blago bolji. Analiza propagacije greške pokazuje da oko petine grešaka
hijerarhije potiče od pogrešno prepoznate vrste. Detaljni rezultati, tabele i figure nalaze se u
`results/`, a kompletna analiza u `Report.pdf`.

## Reproducibilnost

- Fiksni seed-ovi (`torch`, `numpy`, `random`); stratifikovana podela 70/15/15 perzistirana u CSV-u.
- Svaki eksperiment ima sopstvenu YAML konfiguraciju.
- Najbolji model se čuva prema validacionom **macro F1** (ne prema gubitku).

## Rad

Finalna verzija rada je `Report.pdf`. Izvorni kod (LaTeX, IEEE šablon, na srpskom) je u
`paper/main.tex`, a figure u `paper/figures/`. Rad se kompajlira sa `pdfLaTeX` (dva prolaza).
