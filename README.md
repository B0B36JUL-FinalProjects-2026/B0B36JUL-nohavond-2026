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
