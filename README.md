# [Tytuł mini-projektu]

**Autor:** [Imię Nazwisko], nr indeksu: [XXXXXX]

**Temat:** [Numer z menu (1-29) lub "własny"] — [Nazwa tematu]

**Kurs:** Aspekty prawne, społeczne i etyczne w AI, PWr 2025/2026

> Lista tematów: [Zasady zaliczenia — Menu mini-projektów](https://github.com/laugustyniak/ai-ethics-law-course/blob/main/Zasady%20zaliczenia.md#menu-mini-projekt%C3%B3w)

---

## Quick Start

```bash
uv sync                        # zainstaluj zależności
cp .env.example .env           # skopiuj wzór zmiennych środowiskowych
# uzupełnij klucze API w .env

uv run src/example_openai.py   # sprawdź że działa (OpenAI)
uv run src/example_anthropic.py  # lub Anthropic
uv run src/example_gemini.py     # lub Gemini
```

---

## Cel projektu

[2-3 zdania: co projekt robi i po co. Jaki problem rozwiązuje / analizuje?]

## Powiązanie z projektem grupowym

[Jak mini-projekt wiąże się z Waszym projektem naukowo-wdrożeniowym? Jeśli nie — napisz dlaczego wybrałeś ten temat.]

## Wymagania

Projekt korzysta z [uv](https://docs.astral.sh/uv/) — szybkiego menedżera pakietów Python.

```bash
# Instalacja uv (jeśli nie masz)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalacja zależności
uv sync

# Z notebookami Jupyter
uv sync --extra notebooks
```

**Zmienne środowiskowe** — skopiuj plik `.env.example` i uzupełnij klucze API:

```bash
cp .env.example .env
# Uzupełnij klucze w .env (OpenAI / Anthropic / Google — w zależności od projektu)
```

## Uruchomienie

```bash
# Główny skrypt (zamień na swój po implementacji)
uv run src/main.py

# lub notebook
uv run jupyter notebook notebooks/analiza.ipynb
```

## Wyniki

[Najważniejsze wyniki — tabelki, wykresy, liczby. Wstaw bezpośrednio lub linkuj do plików w `wyniki/`.]

## Wnioski merytoryczne

[Kluczowa sekcja — co wynika z analizy w kontekście prawa / etyki / regulacji AI? Konkretne obserwacje i rekomendacje.]

## Ograniczenia

[Czego projekt nie robi? Co można by rozszerzyć? Bądź uczciwy.]

## Źródła

- [Nazwa źródła](URL) — krótki opis