# Rad (IEEE, LaTeX)

`main.tex` — izvorni kod rada na srpskom u IEEE konferencijskom šablonu.
`figures/` — figure korišćene u radu (kopija iz `../results/`).

## Kompajliranje na Overleaf (preporučeno)
1. Otvori [overleaf.com](https://www.overleaf.com) (besplatan nalog).
2. **New Project → Upload Project** i otpremi ceo `paper/` folder
   (ili napravi prazan projekat pa dodaj `main.tex` + folder `figures/`).
3. Compiler ostavi na **pdfLaTeX**. Klikni **Recompile** → dobijaš `main.pdf`.
4. Preuzmi PDF i preimenuj u **`Report.pdf`**, pa stavi u koren repozitorijuma.

> Ako prijavi grešku oko `babel` (srpski jezik), u `main.tex` zameni liniju
> `\usepackage[serbian]{babel}` sa `\usepackage[english]{babel}` — sadržaj ostaje
> na srpskom, menja se samo automatsko prelamanje reči.

## Lokalno (alternativa)
Sa instaliranim TeX Live / MiKTeX:
```bash
pdflatex main.tex && pdflatex main.tex
```
(dva prolaza zbog referenci na slike i literaturu).
