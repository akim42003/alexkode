# Deep-ML 60

A TUI for machine learning and linear algebra problems. Browse by category, difficulty, or completion status---edit solutions in your preferred editor, run tests, and track progress, without leaving the terminal.

![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

## Prerequisites

- **Go** 1.21+
- **Python 3** with NumPy (`pip install numpy`)

## Setup

This project is split across two repositories — the TUI app and the problem sets.

```bash
# 1. Clone the TUI
git clone https://github.com/akim42003/alexkode.git
cd alexkode

# 2. Clone the problems into the TUI directory
git clone https://github.com/akim42003/mldl_problems.git
mv mldl_problems problems
```

Your directory should look like:

```
deep-ml-tui/
├── cmd/
├── internal/
├── problems/          <-- cloned from the problems repo
│   ├── 01-linear-algebra/
│   ├── 02-statistics-probability/
│   ├── 03-machine-learning/
│   ├── 04-deep-learning/
│   ├── 05-nlp/
│   └── 06-computer-vision/
├── config.json
├── go.mod
└── README.md
```

## Build

```bash
go mod init v0
go mod tidy
go build -o alexkode ./cmd/main.go
```

Or run directly:

```bash
go mod init v0
go mod tidy
go run ./cmd/main.go
```

## Usage

```bash
./alexkode
```

### Navigation

| Screen | Key | Action |
|---|---|---|
| **All** | `Ctrl+C` | Quit |
| **All** | `Esc` | Go back |
| **All** | `↑/↓` or `j/k` | Navigate / scroll |
| **Main Menu** | `Enter` | Select filter |
| **Main Menu** | `q` | Quit |
| **Problem List** | `Enter` | Open problem |
| **Problem Detail** | `e` | Edit your solution |
| **Problem Detail** | `r` | Run tests |
| **Problem Detail** | `s` | Toggle solution view |
| **Problem Detail** | `c` | Toggle completed |

### Workflow

1. Pick a problem from the menu (filter by category, difficulty, or status)
2. Read the problem description
3. Press `e` to open your editor — a template file is created automatically
4. Write your solution implementing the `solution()` function
5. Press `r` to run tests against the problem's test cases
6. Tests that all pass will auto-mark the problem as completed

Your code is saved in `user_code/` and your progress in `progress.json` — both are gitignored.

## Configuration

Create a `config.json` in the project root:

```json
{
    "editor": "nvim",
    "terminal": "ghostty"
}
```

| Field | Description | Default |
|---|---|---|
| `editor` | Editor command to open solution files | `$EDITOR`, then `nano` |

The editor is resolved in order: `config.json` > `$EDITOR` environment variable > `nano`.

## Project Structure

```
cmd/main.go                  Entry point
internal/
  config/config.go            User configuration (editor, etc.)
  problem/
    problem.go                Problem types, loader, filters
    runner.go                 Python test runner
  progress/progress.go        Completion tracking (progress.json)
  tui/
    model.go                  App state and initialization
    update.go                 Input handling and view transitions
    view.go                   View routing and header rendering
    styles.go                 Terminal color palette
    mainmenu.go               Main menu (filter type + filter value)
    problemlist.go             Scrollable problem list
    problemdetail.go           Problem detail / solution viewer
    testresults.go             Test results display
problems/                     Problem sets (separate repo)
user_code/                    Your solutions (gitignored)
progress.json                 Your progress (gitignored)
config.json                   Your config (gitignored)
```

## Problems

60 problems across 6 categories:

| Category | Count |
|---|---|
| Linear Algebra | 10 |
| Statistics & Probability | 8 |
| Machine Learning | 15 |
| Deep Learning | 15 |
| NLP | 6 |
| Computer Vision | 6 |

Each problem includes a description, multiple solution approaches, and test cases. Solutions use NumPy only (with nltk allowed for NLP problems).
