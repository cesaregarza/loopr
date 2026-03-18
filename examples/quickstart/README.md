# Quickstart Dataset

This directory contains a small deterministic simulated league used by the
README quickstart.

The simulation is intentionally simple:

- latent player skills
- fixed event rosters with one bench player per team
- actual set appearances that rotate lineups
- best-of-5 set outcomes simulated from lineup strength

The ranking flow only needs:

- `matches.csv`
- `participants.csv`
- optional `appearances.csv`

Additional files are for orientation:

- `entities.csv`: entity IDs and display names
- `groups.csv`: event-local group labels

Regenerate the dataset with:

```bash
python -m loopr.example_data --output-dir examples/quickstart
```

Or:

```bash
PYTHONPATH=src .venv/bin/python examples/quickstart/generate_dataset.py
```
