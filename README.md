# Model oběhu Stirlingova motoru

Tento repozitář obsahuje zdrojové kódy pro interaktivní webovou aplikaci simulující oběh Stirlingova motoru typu Beta. Aplikace je napsaná v Pythonu s využitím frameworku Streamlit a provádí výpočty s uvažováním polytropických změn na teplé i studené straně motoru.

Živá verze aplikace je k dispozici zde: https://stirling-engine-model.streamlit.app/

## Co aplikace umožňuje
* Simulaci ideálního oběhu Stirlingova motoru beze ztrát a vykreslení p-V diagramů.
* Výpočet energetické bilance (indikovaný výkon, přivedené a odvedené teplo, účinnost, regenerované teplo).
* Citlivostní analýzu – výpočet toho, jak změna jednoho vybraného parametru (např. tlaku nebo mrtvých objemů) ovlivňuje celkový výkon a účinnost.
* Odhad reálného výkonu motoru pomocí Bealeova čísla na základě empirických dat G. Walkera.
* Animaci pohybu pístů na základě zadané geometrie a fázového posunu.

## Použité technologie
Skript využívá následující Python knihovny:
* streamlit
* numpy
* scipy
* pandas
* plotly
* matplotlib

## Spuštění na vlastním PC
Pokud si chcete aplikaci stáhnout a spustit u sebe, stačí použít následující příkazy:

1. Stažení repozitáře:
git clone https://github.com/vovota2/Stirling-model.git
cd Stirling-model

2. Instalace potřebných balíčků:
pip install -r requirements.txt

3. Spuštění:
streamlit run BETA_1.py

## Použitá literatura
Data pro výpočet odhadu výkonu a empirické křivky Bealeova čísla vychází z této publikace:
* MARTINI, William. Stirling engine design manual, 2004. Přetisk vydání z roku 1983. Honolulu: University press of the Pacific, ISBN: 1-4102-1604-7.

## Licence
Projekt je uvolněn pod svobodnou licencí GNU GPLv3. Můžete jej volně používat a upravovat, ale případné odvozené projekty musí být uvolněny pod stejnou licencí. Úplné znění najdete v souboru LICENSE.
