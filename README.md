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

## Instalace
 
Je potřeba Julia 1.13 nebo novější.
 
```bash
git clone https://github.com/ORGANIZACE/B0B36JUL-nohavond-2026.git
cd B0B36JUL-nohavond-2026
julia --project=.
```
 
V Julia REPL nainstaluj závislosti:
 
```julia
julia> ]
pkg> instantiate
```
 
```julia
julia> using EarthquakePredictionJapan
```
 
## Testy
 
```julia
pkg> test
```
 
Testy se spouštějí i automaticky na GitHubu po každém pushi (GitHub Actions).
Součástí testů je kontrola kvality balíčku pomocí [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl).
 
## Struktura repozitáře
 
```
├── Project.toml      závislosti balíčku
├── src/              zdrojový kód balíčku
├── test/             testy (runtests.jl, vlastní Project.toml)
└── .github/workflows CI na GitHub Actions
```
 
## Licence
 
Viz soubor [LICENSE](LICENSE).