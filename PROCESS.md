# Dokumentacja procesu

Ten plik dokumentuje **jak** pracowałem/am nad mini-projektem — jakie narzędzia AI wykorzystałem, jakie prompty pisałem, jakie decyzje podjąłem i co nie zadziałało.

> **PROCESS.md jest tak samo ważny jak kod.** Prowadzący ocenia świadome korzystanie z narzędzi AI — to jest kurs o aspektach AI.

---

## Narzędzia AI

[Lista narzędzi AI użytych w projekcie]

| Narzędzie | Do czego używałem |
|-----------|-------------------|
| Gemini Pro | Zaprojektowanie eksperymentów, konsultacje, iteracyjne poprawianie kodu |
| ChatGPT | Dostarczenie wyników i porównanie wyciągniętych wniosków |

## Prompty

> Nie wklejaj outputu z AI — tylko prompty, które wpisywałeś/aś.

### [Kategoria 1, "Generowanie kodu"]

```
Jesteś specjalistą od AI, Grafowych Sieci Neuronowych, 
Wyjaśnialności oraz Neurologii i analizy sygnału EEG do detekcji padaczki.

Masz do dyspozycji wyternowany model oparty o ChebConv wyglądający w ten sposób:
[model]. Wytrenowany został na zbiorze Chb-Mit (pacjenci 1-20 train, 21-22 val, 23-24 test). 
Przyjmuje on grafy oparte o PLV [skrypt do tworzenia], które na węzłach - odpowiadających kanałom -  mają handcrafted features 
często używane w dziedzine detekcji napadów padaczkowych [skrypt do features], stworzone na 6s 
wycinkach z nagrań. Grafy mają labels 0 lub 1, w zależności od tego czy obejmują one okres nienapadowy czy napadowy. 
Zaprojektuj profesjonalną analizę wyjaśnialności tego modelu, którą będzie można przedstawić w Jupyter Notebook.

Tutaj masz dodatkowe instrukcje odnośnie tego projektu:
[Opis miniprojektu]
[AGENST.md]
[PROCESS.md]
```

**Kontekst:** Rozpoczęcie pracy nad projektem, później dalsze iteracyjne poprawianie eksperymentów

```
Na jakich danych powinno robić się wyjaśnialność? Treningowych czy testowych? 
```

**Kontekst:** (w tej samej sesji) LLM automatycznie zaproponował treningowe, jedak po zadaniu tego pytania sprostował że powinny być to dane testowe.

### [Kategoria 2, "Analiza wyników"]

```
co sądzisz o tych wynikach?: Są wzięte z tego kodu [kod]
Są one do zadania: Detekcja padaczki za pomocą eeg, 6s epochs, GNN (cheb conv), handcrafted features i wyjaśnialność za pomocą GNNExplainer.
```

**Kontekst:** Porównanie wniosków z tymi które automatycznie nasuwały się na myśl.

## Decyzje

1. **Zmiana SHAP na GNNExplainer** — Klasyczny SHAP jest niekoniecznie najlepszym rozwiązaniem dla GNNów. Rozważałem również nad GraphSHAP, ale koniec końców zdecydowałem się na aktualnie bardzo popularny GNNExplainer.
2. **Skupienie się na cechach** — Łatwiejsze w interpretacji + *Co nie zadziałało 2.*

## Co nie zadziałało

1. **Niekompatybilny SHAP** — W pierwszej iteracji LLM zaproponował (trzymając się oryginalnych wytycznych) eksperymenty związane z klasycznym SHAP'em. Niestety nie jest on najlepszym rozwiązaniem w przypadku GNNów, jak również zaproponowane eksperymenty nie były klinicznie jakościowe. Zdecydowałem się wtedy na dostosowaną do tego typu architektury metodę GNNExplainer.
2. **Problem z dokładniejszą ewaluacją krawędzi** — Zauważyłem że przy inicjalnie zaproponowanej liczbie epok (200) w algorytmnie GNNExplainer, podświetlane krawędzie potrafiły się różnić przy wywoływaniu tego samego kodu dla tej samej próbki. Zwiększenie epok do 1000 znacząco ustabilizowało ten proces, jednakże znacznie wydłużyło czas potrzebny do wygenerowania analizy. Stąd też postanowiłem skupić się na tworzonych cechach, których wpływ był zdecydowanie bardziej stabilny, nawet przy mniejszej ilości epok. 

## Iteracje

1. **v1** — "Naiwna" próba wykorzystania SHAP.
2. **v2** — Zmiana na GNNExplainer oraz dalsza analiza co na jego podstawie można zewaluować.
3. **vN** — Dokładna analiza istotności cech dla decyzji modelu, w zależności od jego wyniku.