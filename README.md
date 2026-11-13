# Srovnání point-estimate a pravděpodobnostních modelů ve Flux.jl

## Zadání

Porovnat alespoň 3 metody strojového učení ve Flux.jl. Nejprve jako klasické
modely s bodovým odhadem (point-estimate), poté rozšířit vybrané z nich
o pravděpodobnostní výstup, který zahrnuje odhad nejistoty predikce.

## Požadavky

- Srovnat alespoň 3 metody a vybrané rozšířit o predikci nejistoty.
- Implementovat alespoň 3 metriky, na kterých se modely porovnají. Některé
  metriky dávají smysl jen pro point-estimate modely, jiné jen pro
  pravděpodobnostní.
- Ke vhodným metrikám zkusit i odpovídající loss funkce při tréninku (kde to
  dává smysl).
- Využít multiple dispatch při evaluaci, a to na typu výstupu modelu
  i na typu metriky.
- Udržet kód přehledný a modulární. Modely se skládají z existujících vrstev
  Flux.jl, takže důraz je na struktuře kódu a využití dispatche.

## Data a úloha

Data jsou katalog zemětřesení [USGS](https://earthquake.usgs.gov/fdsnws/event/1/)
pro oblast kolem Japonska (24–46° s. š., 122–148° v. d.) v letech 1990–2025.
Po vyčištění zůstanou jen zemětřesení s magnitudem **ostře větším než 5.0**.

Z každých `k = 10` po sobě jdoucích událostí (čas od předchozí události, poloha,
hloubka, magnitudo) se predikuje magnitudo následující události. Okna se tvoří
nad celým katalogem a teprve potom se chronologicky dělí na trénovací (70 %),
validační (15 %) a testovací (15 %) část.

## Modely a metriky

| Model | Bodový odhad | S nejistotou |
|---|---|---|
| `MeanBaseline` | průměr trénovacích magnitud | empirické rozdělení (`SampleHead`) |
| `RandomForest` | průměr stromů | predikce jednotlivých stromů (`SampleHead`) |
| `MLP` | MSE loss | normální rozdělení, NLL loss (`GaussianHead`) |
| `LSTMModel` (+ `TemporalAttention`) | MSE loss | normální rozdělení, NLL loss (`GaussianHead`) |

Metriky: `MAE`, `RMSE`, `CRPS`, `NLL` a `Coverage` (pokrytí intervalu).
`evaluate(metrika, predikce, y)` se vybírá podle typu metriky i typu predikce
(`PointPrediction`, `SamplePrediction`, `GaussianPrediction`). Pokud metrika
pro daný typ nedává smysl, vrací `missing`.

## Instalace

Je potřeba Julia 1.13 nebo novější.

```bash
git clone git@github.com:B0B36JUL-FinalProjects-2026/B0B36JUL-nohavond-2026.git
cd B0B36JUL-nohavond-2026
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
```

Balíček (`src/`) má vlastní prostředí v kořeni repozitáře. Skripty ve složce
`scripts/` mají navíc vlastní prostředí s balíčky pro grafy a ukládání
(CairoMakie, JLD2).

## Spuštění

Všechny příkazy se spouštějí z kořene repozitáře.

**1. Stažení dat** (uloží se do `data/japan_usgs.csv`, složka není v gitu):

```bash
julia --project=. scripts/download_data.jl
```

**2. Trénink jednotlivých modelů.** Každý skript natrénuje bodovou variantu
i variantu s nejistotou, vypíše tabulku metrik a uloží výsledky:

```bash
julia --project=scripts scripts/train_baseline.jl
julia --project=scripts scripts/train_forest.jl
julia --project=scripts scripts/train_mlp.jl
julia --project=scripts scripts/train_lstm.jl
```

**3. Souhrnné srovnání** všech natrénovaných modelů (tabulka obou částí dat,
graf CRPS a kalibrace intervalů):

```bash
julia --project=scripts scripts/compare.jl
```

Výstupy:

- `results/<model>.txt` a `results/summary.txt`: tabulky metrik,
- `results/<model>.jld2`: predikce na validačních a testovacích datech,
- `models/<model>.jld2`: natrénované modely, jen lokálně (složka není v gitu);
  neuronové sítě se ukládají jen jako `Flux.state` a parametry škálování
  (viz `model_state` a `load_state!`),
- `figures/*.png`: grafy.

Uloženou síť lze obnovit takto (hyperparametry musí odpovídat tréninku):

```julia
using EarthquakePredictionJapan, JLD2
saved = load("models/lstm.jld2", "gauss_model")
model = load_state!(LSTMModel(head = GaussianHead(), hidden = 64, dense = 32), saved)
```

## Použití z REPL

```julia
using EarthquakePredictionJapan

windows = make_windows(load_catalog(); k = 10)
data = chronological_split(windows)

model = fit!(MLP(head = GaussianHead()), data.train; val = data.val)
p = predict(model, data.test)
evaluate(CRPS(), p, data.test.y)
evaluate(Coverage(0.9), p, data.test.y)
```

## Testy

```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```

Testy se spouštějí i automaticky na GitHubu po každém pushi (GitHub Actions).
Součástí testů je kontrola kvality balíčku pomocí [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl).

## Struktura repozitáře

```
├── Project.toml          závislosti balíčku
├── src/
│   ├── data/             načtení, okna, dělení a standardizace dat
│   ├── models/           modely (baseline, random forest, MLP, LSTM, attention)
│   ├── predictions.jl    typy predikcí
│   ├── metrics.jl        metriky a evaluate
│   ├── losses.jl         loss funkce pro jednotlivé hlavy
│   └── train_flux.jl     trénovací smyčka neuronových sítí
├── scripts/              stažení dat, trénink, souhrnné srovnání (vlastní Project.toml)
├── results/              tabulky metrik a uložené predikce
├── models/               uložené modely (jen lokálně, není v gitu)
├── figures/              grafy
├── test/                 testy (runtests.jl, vlastní Project.toml)
└── .github/workflows     CI na GitHub Actions
```

## Licence

Viz soubor [LICENSE](LICENSE).
